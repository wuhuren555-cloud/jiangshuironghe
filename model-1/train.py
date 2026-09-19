"""M1 terrain expert with calendar-safe Lead 0/1/3/5 and strict LOSO.

Training:
  python train.py train --csv input.csv --out results_m1
Prediction:
  python train.py predict --csv history.csv --model results_m1/lead_1 \
      --issue-date 2024-12-30 --out prediction.csv

Date is the precipitation accumulation day. Lead 0 is retrospective fusion;
positive leads are forecasts. By default gpm_delay_days=0 to preserve the
previous code's experiment. Set it to the product's real issue-time latency for
an operational experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

STATIC = ['DEM', 'Slope', 'lat', 'lon']


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_data(path, require_obs=True):
    df = pd.read_csv(path)
    date_col = next((c for c in ['Date', 'date', 'time', 'datetime', 'DATE'] if c in df), None)
    if date_col is None:
        raise ValueError('A real date column is required; dates must never be invented.')
    required = ['station_name', 'gpm_rain', 'Aspect'] + STATIC
    if require_obs:
        required += ['station_rain']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df['date'] = pd.to_datetime(df[date_col], errors='raise')
    if df['date'].isna().any() or df['station_name'].isna().any():
        raise ValueError('Missing station/date keys.')
    if df['date'].dt.tz is not None or (df['date'] != df['date'].dt.normalize()).any():
        raise ValueError('Use timezone-free daily dates after aligning accumulation windows.')
    df['station_name'] = df['station_name'].astype(str)
    if df.duplicated(['station_name', 'date']).any():
        raise ValueError('Duplicate station/date keys must be resolved upstream.')
    for col in ['gpm_rain', 'Aspect'] + STATIC + (['station_rain'] if 'station_rain' in df else []):
        df[col] = pd.to_numeric(df[col], errors='raise')
        if np.isinf(df[col]).any():
            raise ValueError(f'Infinite values in {col}.')
    for col in ['gpm_rain', 'station_rain']:
        if col in df and (df[col].dropna() < 0).any():
            raise ValueError(f'Negative {col}; convert documented missing sentinels upstream.')
    if df[STATIC + ['Aspect']].isna().any().any():
        raise ValueError('Static terrain fields must be complete; do not silently fill with zero.')
    if not df['lat'].between(-90, 90).all() or not df['lon'].between(-180, 180).all():
        raise ValueError('Invalid coordinates.')
    if not df['Slope'].between(0, 90).all() or not df['Aspect'].between(0, 360).all():
        raise ValueError('Slope/aspect must be degrees; resolve flat-surface nodata upstream.')
    if (df.groupby('station_name')[STATIC + ['Aspect']].nunique() > 1).any().any():
        raise ValueError('Static fields change within a station/grid cell.')
    return df.sort_values(['station_name', 'date']).reset_index(drop=True)


def build_features(raw, lead, delay, history, issue_dates=None):
    """Exact calendar lookups, never shift rows of a table with missing dates.

    Targets are not features. Also usable without station_rain at inference.
    """
    if min(lead, delay) < 0 or history < 1:
        raise ValueError('Nonnegative lead/delay and positive history are required.')
    if issue_dates is None:
        frame = raw[['station_name', 'date']].rename(columns={'date': 'valid_date'}).copy()
        frame['issue_date'] = frame.valid_date - pd.to_timedelta(lead, unit='D')
    else:
        frame = pd.MultiIndex.from_product(
            [raw.station_name.unique(), pd.to_datetime(issue_dates)],
            names=['station_name', 'issue_date']).to_frame(index=False)
        frame['valid_date'] = frame.issue_date + pd.to_timedelta(lead, unit='D')
    meta = raw.groupby('station_name')[STATIC + ['Aspect']].first()
    frame = frame.join(meta, on='station_name', validate='many_to_one')
    series = raw.set_index(['station_name', 'date'])['gpm_rain']
    terrain_features = STATIC.copy()
    frame['aspect_sin'] = np.sin(np.deg2rad(frame.Aspect))
    frame['aspect_cos'] = np.cos(np.deg2rad(frame.Aspect))
    terrain_features += ['aspect_sin', 'aspect_cos']
    frame['latest_gpm_date'] = frame.issue_date - pd.to_timedelta(delay, unit='D')
    lag_cols = []
    for lag in range(history):
        name = f'gpm_lag_{lag}'
        dates = frame.latest_gpm_date - pd.to_timedelta(lag, unit='D')
        keys = pd.MultiIndex.from_arrays([frame.station_name, dates])
        frame[name] = series.reindex(keys).to_numpy()
        lag_cols.append(name)
    # Preserve the previous feature contract while replacing row shifts with
    # exact calendar lookups. Missing calendar days remain missing and the row
    # is removed below instead of silently becoming a different lead time.
    if history < 4:
        raise ValueError('history_days must be at least 4 for gpm_curr + three lags.')
    frame['gpm_mean_3'] = frame[lag_cols[:3]].mean(axis=1, skipna=False)
    frame['month'] = frame.valid_date.dt.month
    frame['month_sin'] = np.sin(2 * np.pi * (frame.month - 1) / 12)
    frame['month_cos'] = np.cos(2 * np.pi * (frame.month - 1) / 12)
    calendar_features = ['month_sin', 'month_cos']
    if 'station_rain' in raw:
        obs = raw.set_index(['station_name', 'date']).station_rain
        frame['obs'] = obs.reindex(pd.MultiIndex.from_arrays(
            [frame.station_name, frame.valid_date])).to_numpy()
    if lead == 0:
        features = [lag_cols[0]] + terrain_features + calendar_features
    else:
        features = lag_cols[:4] + ['gpm_mean_3'] + terrain_features + calendar_features
    frame = frame.dropna(subset=['obs'] + features if 'obs' in frame else features).copy()
    assert (frame.latest_gpm_date <= frame.issue_date).all()
    return frame, features


def divide(a, b):
    return float(a / b) if b != 0 else np.nan


def continuous(obs, pred):
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    if len(o) == 0:
        return dict.fromkeys(['MAE', 'MSE', 'RMSE', 'R2', 'NSE', 'CC', 'KGE', 'PBIAS_pct'], np.nan)
    mse = float(np.mean((p-o)**2))
    nse = 1 - divide(np.sum((p-o)**2), np.sum((o-o.mean())**2))
    r = float(np.corrcoef(o, p)[0, 1]) if len(o) > 1 and o.std() > 0 and p.std() > 0 else np.nan
    kge = 1-np.sqrt((r-1)**2+(divide(p.std(), o.std())-1)**2+(divide(p.mean(), o.mean())-1)**2)
    return dict(MAE=float(np.mean(abs(p-o))), MSE=mse, RMSE=np.sqrt(mse),
                R2=nse, NSE=nse, CC=r, KGE=kge, PBIAS_pct=100*divide(np.sum(p-o), np.sum(o)))


def categorical(observed_event, predicted_event):
    a, b = np.asarray(observed_event, bool), np.asarray(predicted_event, bool)
    h, f, m, c = [int(x.sum()) for x in (a & b, ~a & b, a & ~b, ~a & ~b)]
    random_hits = divide((h+f)*(h+m), len(a))
    return dict(H=h, F=f, M=m, CN=c, POD=divide(h,h+m), FAR=divide(f,h+f),
                CSI=divide(h,h+f+m), ETS=divide(h-random_hits,h+f+m-random_hits),
                FBIAS=divide(h+f,h+m), F1=divide(2*h,2*h+f+m))


def metrics(obs, pred, q90, wet=0.1):
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    q = np.broadcast_to(np.asarray(q90, float), o.shape)
    valid = np.isfinite(o) & np.isfinite(p) & np.isfinite(q)
    n_total = len(o)
    o, p, q = o[valid], p[valid], q[valid]
    # The requested P90 scenario is conditioned only on observed rainfall >=
    # the station threshold. Predicted extremes outside those dates are not
    # part of this conditional scenario table.
    heavy = o >= q
    result = dict(n_total=n_total, n_valid=len(o), n_excluded=n_total-len(o), n_extreme=int(heavy.sum()))
    result.update({'all_'+k:v for k,v in continuous(o,p).items()})
    result.update({'wet_'+k:v for k,v in categorical(o >= wet,p >= wet).items()})
    result.update({'extreme_'+k:v for k,v in continuous(o[heavy],p[heavy]).items()})
    # FHV-like precipitation high-volume bias, paired on observed P90 dates.
    phv = 100*divide(np.sum(p[heavy]-o[heavy]),np.sum(o[heavy]))
    result['PHV90_pct'] = phv
    result['HB90_pct'] = phv  # backward-compatible alias for prior result files
    result['PEAK90_bias_pct'] = (100*divide(np.max(p[heavy])-np.max(o[heavy]),np.max(o[heavy]))
                                 if heavy.any() else np.nan)
    result['Alpha90'] = (divide(np.std(p[heavy]), np.std(o[heavy]))
                         if heavy.any() and np.std(o[heavy]) > 0 else np.nan)
    result['POD90'] = divide(np.sum(p[heavy] >= q[heavy]),np.sum(heavy))
    return result


def partition(frame, args):
    dates = [pd.Timestamp(x) for x in [args.train_start,args.train_end,args.val_end,args.test_end]]
    start, tr, va, te = dates
    if not start < tr < va < te:
        raise ValueError('Split boundaries must be strictly increasing.')
    valid_obs = np.isfinite(frame.obs)
    # A fixed model/tuned configuration must exist by the issue time.
    masks = dict(train=(frame.valid_date >= start)&(frame.valid_date <= tr),
                 val=(frame.valid_date > tr)&(frame.valid_date <= va)&(frame.issue_date >= tr),
                 test=(frame.valid_date > va)&(frame.valid_date <= te)&(frame.issue_date >= va))
    result = {k:frame.loc[v & valid_obs].copy() for k,v in masks.items()}
    if any(x.empty for x in result.values()):
        raise ValueError('Empty train/validation/test split.')
    return result


def fit_pair(train, features, params, wet, seed, jobs):
    import xgboost as xgb
    shared = dict(tree_method='hist',random_state=seed,n_jobs=jobs,verbosity=0,
                  n_estimators=params['n_estimators'],max_depth=params['max_depth'],
                  learning_rate=params['learning_rate'],subsample=params['subsample'],
                  colsample_bytree=params['colsample_bytree'],
                  min_child_weight=params['min_child_weight'],reg_alpha=params['reg_alpha'],
                  reg_lambda=params['reg_lambda'])
    y = (train.obs >= wet).astype(int)
    if y.nunique() != 2:
        raise ValueError('Training requires both wet and dry observations.')
    classifier = xgb.XGBClassifier(**shared,objective='binary:logistic',eval_metric='logloss')
    classifier.fit(train[features], y)
    regressor = xgb.XGBRegressor(**shared,objective='reg:squarederror')
    rainy = train.loc[y == 1]
    # M1 remains a general terrain-correction expert. Q90 labels are deliberately
    # excluded from fitting; extremes are handled by the separate extreme expert.
    regressor.fit(rainy[features],rainy.obs)
    return classifier, regressor


def prediction(pair, data, features, threshold):
    prob = pair[0].predict_proba(data[features])[:,1]
    amount = np.maximum(pair[1].predict(data[features]),0.0)
    return np.where(prob >= threshold,amount,0.0), prob


def tune(train, val, features, args):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Scale only balances the four stations. P90 labels/metrics are not used for
    # model selection, so M1 does not absorb the role of the extreme expert.
    scales = train.groupby('station_name').obs.std(ddof=0).clip(lower=1.0)
    def objective(trial):
        params = dict(n_estimators=trial.suggest_int('n_estimators',100,300,step=50),
                      max_depth=trial.suggest_int('max_depth',2,5),
                      learning_rate=trial.suggest_float('learning_rate',0.02,0.12,log=True),
                      subsample=trial.suggest_float('subsample',0.7,1.0),
                      colsample_bytree=trial.suggest_float('colsample_bytree',0.7,1.0),
                      min_child_weight=trial.suggest_float('min_child_weight',3,20),
                      reg_alpha=trial.suggest_float('reg_alpha',0.001,10,log=True),
                      reg_lambda=trial.suggest_float('reg_lambda',1,30,log=True))
        pair = fit_pair(train,features,params,args.wet_threshold,args.seed,args.jobs)
        prob = pair[0].predict_proba(val[features])[:,1]
        amount = np.maximum(pair[1].predict(val[features]),0.0)
        best = (float('inf'),0.5)
        for threshold in np.linspace(0.1,0.9,17):
            pred = np.where(prob >= threshold,amount,0.0)
            scores = []
            for station in val.station_name.unique():
                ix = (val.station_name == station).to_numpy()
                o, p = val.obs.to_numpy()[ix], pred[ix]
                full_rmse = np.sqrt(np.mean((p-o)**2))
                scores.append(full_rmse/scales[station])
            score = float(np.mean(scores))
            if score < best[0]:
                best = score,float(threshold)
        trial.set_user_attr('threshold',best[1])
        return best[0]
    study = optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective,n_trials=args.trials,n_jobs=1)
    return study


def evaluate(predictions, wet, models=None):
    if models is None:
        models = [m for m in ['M1','GPM_latest','monthly_climatology'] if m in predictions]
    rows = []
    for (lead,split), group in predictions.groupby(['lead_days','split']):
        for model in models:
            for station, sub in group.groupby('station_name'):
                rows.append(dict(lead_days=lead,split=split,model=model,station=station,
                    **metrics(sub.obs,sub[model],sub.q90,wet)))
            rows.append(dict(lead_days=lead,split=split,model=model,station='__pooled__',
                **metrics(group.obs,group[model],group.q90,wet)))
    table = pd.DataFrame(rows)
    station_rows = table[table.station != '__pooled__']
    cols = [c for c in table if c not in ['lead_days','split','model','station']]
    macro = station_rows.groupby(['lead_days','split','model'])[cols].mean().reset_index()
    macro['station'] = '__macro_mean__'
    # Counts in a macro row mean total counts; all score columns are station means.
    count_cols = [c for c in cols if c.startswith('n_') or c.endswith(('_H','_F','_M','_CN'))]
    sums = station_rows.groupby(['lead_days','split','model'])[count_cols].sum().reset_index()
    macro = macro.drop(columns=count_cols).merge(sums,on=['lead_days','split','model'])
    return pd.concat([table,macro],ignore_index=True)


def p90_table(table):
    """Compact conditional table using only observed-rain >= station Q90 dates."""
    identity = ['lead_days','split','model','station','n_total','n_valid','n_excluded','n_extreme']
    diagnostics = ['extreme_MAE','extreme_MSE','extreme_RMSE','extreme_R2','extreme_NSE',
                   'extreme_CC','extreme_KGE','extreme_PBIAS_pct','PHV90_pct',
                   'PEAK90_bias_pct','Alpha90','POD90']
    return table[[c for c in identity+diagnostics if c in table.columns]].copy()


def overall_table(table):
    """Compact full-sample table, separate from the conditional P90 diagnostics."""
    identity = ['lead_days','split','model','station','n_total','n_valid','n_excluded']
    diagnostics = [c for c in table.columns if c.startswith('all_') or c.startswith('wet_')]
    return table[[c for c in identity+diagnostics if c in table.columns]].copy()


def write_metric_views(table, out, suffix=''):
    """Preserve the complete table and emit two unambiguous reporting views."""
    suffix = f'_{suffix}' if suffix else ''
    table.to_csv(out/f'metrics{suffix}.csv',index=False,encoding='utf-8-sig')
    overall_table(table).to_csv(out/f'overall_metrics{suffix}.csv',index=False,encoding='utf-8-sig')
    p90_table(table).to_csv(out/f'p90_metrics{suffix}.csv',index=False,encoding='utf-8-sig')


OVERALL_REPORT_METRICS = [
    ('样本总天数 (Valid Days)', 'n_valid', '{:.0f}'),
    ('平均绝对误差 MAE (mm/d)', 'all_MAE', '{:.3f}'),
    ('均方根误差 RMSE (mm/d)', 'all_RMSE', '{:.3f}'),
    ('纳什效率 NSE (R²)', 'all_NSE', '{:.3f}'),
    ('相关系数 CC', 'all_CC', '{:.3f}'),
    ('Kling-Gupta 效率 KGE', 'all_KGE', '{:.3f}'),
    ('水量相对偏差 PBIAS (%)', 'all_PBIAS_pct', '{:+.2f}%'),
    ('降雨命中率 POD (≥0.1mm)', 'wet_POD', '{:.3f}'),
    ('降雨空报率 FAR (≥0.1mm)', 'wet_FAR', '{:.3f}'),
    ('临界成功指数 CSI (≥0.1mm)', 'wet_CSI', '{:.3f}'),
]

# Every score below is evaluated only on dates where observed rain >= station Q90.
# FAR90/CSI90/SEDI are intentionally absent: they need observed non-events and
# therefore are not conditional-P90-only scores.
P90_REPORT_METRICS = [
    ('P90情景样本数 (Obs≥Q90)', 'n_extreme', '{:.0f}'),
    ('暴雨平均绝对误差 MAE (mm/d)', 'extreme_MAE', '{:.3f}'),
    ('暴雨均方根误差 RMSE (mm/d)', 'extreme_RMSE', '{:.3f}'),
    ('暴雨相关系数 CC', 'extreme_CC', '{:.3f}'),
    ('暴雨纳什效率 NSE', 'extreme_NSE', '{:.3f}'),
    ('暴雨 Kling-Gupta KGE', 'extreme_KGE', '{:.3f}'),
    ('暴雨水量容积偏差 PHV90 (%)', 'PHV90_pct', '{:+.2f}%'),
    ('暴雨峰值相对误差 PEAK (%)', 'PEAK90_bias_pct', '{:+.2f}%'),
    ('暴雨变异比 Alpha90 (σs/σo)', 'Alpha90', '{:.3f}'),
    ('条件命中率 POD90', 'POD90', '{:.3f}'),
]


def build_vertical_table(data, key, labels, definitions):
    result = pd.DataFrame({'评估指标项 (Evaluation Metric)': [x[0] for x in definitions]})
    for value, label in labels.items():
        matched = data[data[key] == value]
        if matched.empty:
            result[label] = '—'
            continue
        row = matched.iloc[0]
        values = []
        for _, field, fmt in definitions:
            number = row.get(field, np.nan)
            values.append('—' if pd.isna(number) or np.isinf(number) else fmt.format(number))
        result[label] = values
    return result


def print_vertical_reports(temporal_metrics, loso_metrics, out):
    """Print the four legacy-style reports plus one temporal-vs-LOSO table."""
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1400)
    pd.set_option('display.unicode.east_asian_width', True)
    main = temporal_metrics[(temporal_metrics.model == 'M1')].copy()
    macro = main[main.station == '__macro_mean__'].copy()

    split_labels = {
        'train': '训练集 (2012-2019)',
        'val': '验证集 (2020-2021)',
        'test': '独立测试集 (2022-2024)',
    }
    table1 = build_vertical_table(macro[macro.lead_days == 0], 'split',
                                  split_labels, OVERALL_REPORT_METRICS)
    test_macro = macro[macro.split == 'test']
    leads = sorted(test_macro.lead_days.unique())
    lead_labels = {lead: ('Lead 0 (融合基线)' if lead == 0 else f'+{int(lead)}天预报')
                   for lead in leads}
    table2 = build_vertical_table(test_macro, 'lead_days', lead_labels, OVERALL_REPORT_METRICS)
    table3 = build_vertical_table(test_macro, 'lead_days', lead_labels, P90_REPORT_METRICS)

    station_data = main[(main.lead_days == 0) & (main.split == 'test')]
    stations = sorted(x for x in station_data.station.unique() if not x.startswith('__'))
    station_labels = {x: f'【{x}站】' for x in stations}
    station_labels['__macro_mean__'] = '★ 全流域均值'
    table4 = build_vertical_table(station_data, 'station', station_labels,
                                  OVERALL_REPORT_METRICS)

    loso_macro = loso_metrics[(loso_metrics.model == 'M1') &
                              (loso_metrics.station == '__macro_mean__') &
                              (loso_metrics.split == 'loso_test')]
    left = test_macro[['lead_days','n_valid','all_RMSE','all_CC','all_PBIAS_pct',
                       'PHV90_pct']].rename(columns={
        'n_valid':'时间盲测样本数','all_RMSE':'时间盲测RMSE','all_CC':'时间盲测CC',
        'all_PBIAS_pct':'时间盲测PBIAS(%)','PHV90_pct':'时间盲测PHV90(%)'})
    right = loso_macro[['lead_days','n_valid','all_RMSE','all_CC','all_PBIAS_pct',
                        'PHV90_pct']].rename(columns={
        'n_valid':'LOSO样本数','all_RMSE':'LOSO_RMSE','all_CC':'LOSO_CC',
        'all_PBIAS_pct':'LOSO_PBIAS(%)','PHV90_pct':'LOSO_PHV90(%)'})
    table5 = left.merge(right, on='lead_days', how='outer').sort_values('lead_days')
    table5['RMSE变化(LOSO-时间盲测)'] = table5['LOSO_RMSE'] - table5['时间盲测RMSE']
    table5 = table5[['lead_days','时间盲测样本数','LOSO样本数','时间盲测RMSE','LOSO_RMSE',
                     'RMSE变化(LOSO-时间盲测)','时间盲测CC','LOSO_CC',
                     '时间盲测PBIAS(%)','LOSO_PBIAS(%)','时间盲测PHV90(%)','LOSO_PHV90(%)']]

    reports = [
        ('表 1：Lead 0 全量数据三级时段评估', table1),
        ('表 2：测试期全量数据多预见期对照', table2),
        ('表 3：测试期 P90 条件情景（仅 Obs≥Q90 日期）', table3),
        ('表 4：Lead 0 测试期各站点空间精度', table4),
        ('表 5：原时间盲测与严格 LOSO 空间外推对照', table5),
    ]
    for title, table in reports:
        print('\n' + '='*110)
        print(title)
        print('='*110)
        print(table.to_string(index=False))
    print('='*110)

    table1.to_csv(out/'table1_splits_vertical.csv', index=False, encoding='utf-8-sig')
    table2.to_csv(out/'table2_leads_vertical.csv', index=False, encoding='utf-8-sig')
    table3.to_csv(out/'table3_p90_conditional_vertical.csv', index=False, encoding='utf-8-sig')
    table4.to_csv(out/'table4_stations_vertical.csv', index=False, encoding='utf-8-sig')
    table5.to_csv(out/'table5_temporal_vs_loso.csv', index=False, encoding='utf-8-sig')


def common_across_leads(predictions):
    """Keep identical station/valid-day support across leads AND baselines."""
    available = predictions[np.isfinite(predictions.GPM_latest)].copy()
    n_leads = predictions.lead_days.nunique()
    keys = ['split','station_name','valid_date']
    counts = available.groupby(keys).lead_days.transform('nunique')
    return available[counts == n_leads]


def loso_fold(splits, heldout_station):
    """Strict spatial-temporal fold: held-out station is absent from fit/tuning."""
    return {
        'train': splits['train'][splits['train'].station_name != heldout_station].copy(),
        'val': splits['val'][splits['val'].station_name != heldout_station].copy(),
        'test': splits['test'][splits['test'].station_name == heldout_station].copy(),
    }


def run_loso(raw, thresholds, args, out):
    """Run nested LOSO diagnostics in addition to the primary temporal experiment.

    For each held-out station, hyperparameters use only the other stations'
    2012-2019 training and 2020-2021 validation samples. Evaluation uses the
    held-out station in 2022-2024. Its training-period Q90 is evaluation metadata
    only and never enters fitting or tuning.
    """
    stations = sorted(raw.station_name.unique())
    if len(stations) < 2:
        raise ValueError('LOSO requires at least two stations.')
    predictions = []
    for lead in sorted(set(args.leads)):
        frame, features = build_features(raw,lead,args.gpm_delay_days,args.history_days)
        frame['q90'] = frame.station_name.map(thresholds)
        base_splits = partition(frame,args)
        for heldout in stations:
            fold = loso_fold(base_splits,heldout)
            if any(x.empty for x in fold.values()):
                raise ValueError(f'Empty LOSO fold for held-out station {heldout}, lead {lead}.')
            study = tune(fold['train'],fold['val'],features,args)
            threshold = study.best_trial.user_attrs['threshold']
            pair = fit_pair(fold['train'],features,study.best_params,
                            args.wet_threshold,args.seed,args.jobs)
            safe_station = str(heldout).replace('/','_').replace('\\','_')
            directory = out/'loso'/f'lead_{lead}'/f'heldout_{safe_station}'
            directory.mkdir(parents=True,exist_ok=True)
            pair[0].save_model(directory/'classifier.json')
            pair[1].save_model(directory/'regressor.json')
            save_json(directory/'config.json',dict(
                features=features,threshold=threshold,lead_days=lead,
                heldout_station=str(heldout),gpm_delay_days=args.gpm_delay_days,
                history_days=args.history_days,params=study.best_params,
                validation_score=study.best_value,seed=args.seed,
                q90_note='Held-out station training-period Q90; evaluation only'))
            study.trials_dataframe().to_csv(directory/'trials.csv',index=False)
            sub = fold['test']
            pred, prob = prediction(pair,sub,features,threshold)
            result = sub[['station_name','issue_date','valid_date','latest_gpm_date','obs','q90']].copy()
            result['M1'] = pred
            result['wet_probability'] = prob
            result['GPM_latest'] = sub.gpm_lag_0
            result['n_missing_gpm_lags'] = sub[[f'gpm_lag_{i}' for i in range(args.history_days)]].isna().sum(axis=1)
            result['lead_days'] = lead
            result['split'] = 'loso_test'
            result['heldout_station'] = heldout
            predictions.append(result)
            print(f'LOSO lead {lead}, held out {heldout}: validation score={study.best_value:.4f}; wet gate={threshold:.2f}',flush=True)
    result = pd.concat(predictions,ignore_index=True)
    result.to_csv(out/'predictions_loso.csv',index=False,encoding='utf-8-sig')
    metrics_loso = evaluate(result,args.wet_threshold,models=['M1','GPM_latest'])
    write_metric_views(metrics_loso,out,'loso')
    shared = common_across_leads(result)
    shared_metrics = evaluate(shared,args.wet_threshold,models=['M1','GPM_latest'])
    write_metric_views(shared_metrics,out,'loso_common_all_leads')
    return metrics_loso


def train_command(args):
    if args.trials < 1 or args.jobs < 1:
        raise ValueError('trials and jobs must be positive.')
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('Output already contains a run; use a new output directory.')
    raw = read_data(args.csv)
    train_obs = raw[raw.date.between(args.train_start,args.train_end)].dropna(subset=['station_rain'])
    thresholds = train_obs.groupby('station_name').station_rain.quantile(.9)
    if set(raw.station_name)-set(thresholds.index):
        raise ValueError('Every evaluated station must have training-period observations for Q90.')
    if (thresholds <= 0).any():
        warnings.warn('Some all-day Q90 thresholds are zero; report these as relative extremes, not absolute heavy rain.')
    thresholds.rename('q90_mm_day').to_csv(out/'thresholds.csv',encoding='utf-8-sig')
    # Same baseline and Q90 period across leads; neither uses validation/test labels.
    climatology = train_obs.assign(month=train_obs.date.dt.month).groupby(['station_name','month']).station_rain.mean()
    annual = train_obs.groupby('station_name').station_rain.mean()
    audit = []
    for station, sub in raw.groupby('station_name'):
        audit.append(dict(station=station,n=len(sub),first_date=str(sub.date.min().date()),
                          last_date=str(sub.date.max().date()),missing_calendar_days=int((sub.date.max()-sub.date.min()).days+1-len(sub)),
                          missing_gpm=int(sub.gpm_rain.isna().sum()),missing_obs=int(sub.station_rain.isna().sum())))
    save_json(out/'data_audit.json',audit)
    manifest = dict(config=vars(args),input_sha256=hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
                    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','xgboost','scikit-learn','optuna']},
                    interpretation='Historical availability-assumed hindcast; not operational verification',
                    quantile='Station training-period ALL valid days, linear Q90, conditional scenario obs >= threshold',
                    target='Single valid-day precipitation, not lead-period accumulated precipitation')
    save_json(out/'manifest.json',manifest)
    all_predictions = []
    for lead in sorted(set(args.leads)):
        directory = out/f'lead_{lead}'
        directory.mkdir(exist_ok=True)
        frame, features = build_features(raw,lead,args.gpm_delay_days,args.history_days)
        frame['q90'] = frame.station_name.map(thresholds)
        splits = partition(frame,args)
        if any(splits['train'][c].notna().sum() == 0 for c in features):
            raise ValueError('A training feature is entirely missing; check available history and delay.')
        print(f'Lead {lead}: '+str({k:len(v) for k,v in splits.items()}),flush=True)
        study = tune(splits['train'],splits['val'],features,args)
        threshold = study.best_trial.user_attrs['threshold']
        pair = fit_pair(splits['train'],features,study.best_params,args.wet_threshold,args.seed,args.jobs)
        pair[0].save_model(directory/'classifier.json')
        pair[1].save_model(directory/'regressor.json')
        save_json(directory/'config.json',dict(features=features,threshold=threshold,lead_days=lead,
             gpm_delay_days=args.gpm_delay_days,history_days=args.history_days,params=study.best_params,
             train_end=args.train_end,val_end=args.val_end,wet_threshold=args.wet_threshold,
             validation_score=study.best_value,seed=args.seed))
        study.trials_dataframe().to_csv(directory/'trials.csv',index=False)
        for split, sub in splits.items():
            pred, prob = prediction(pair,sub,features,threshold)
            result = sub[['station_name','issue_date','valid_date','latest_gpm_date','obs','q90']].copy()
            result['M1'] = pred
            result['wet_probability'] = prob
            result['GPM_latest'] = sub.gpm_lag_0
            keys = pd.MultiIndex.from_arrays([sub.station_name,sub.month])
            climate_values = climatology.reindex(keys).to_numpy()
            result['monthly_climatology'] = np.where(np.isfinite(climate_values),climate_values,sub.station_name.map(annual))
            result['n_missing_gpm_lags'] = sub[[f'gpm_lag_{i}' for i in range(args.history_days)]].isna().sum(axis=1)
            result['lead_days'],result['split'] = lead,split
            all_predictions.append(result)
        print(f'Lead {lead}: validation score={study.best_value:.4f}; wet gate={threshold:.2f}',flush=True)
    predictions = pd.concat(all_predictions,ignore_index=True)
    predictions.to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    full_metrics = evaluate(predictions,args.wet_threshold)
    write_metric_views(full_metrics,out)
    # Fair baseline comparison when the legacy cleaned CSV has missing GPM days.
    common = predictions[np.isfinite(predictions.GPM_latest)]
    common_metrics = evaluate(common,args.wet_threshold)
    write_metric_views(common_metrics,out,'common_support')
    shared = common_across_leads(predictions)
    shared_metrics = evaluate(shared,args.wet_threshold)
    write_metric_views(shared_metrics,out,'common_all_leads')
    loso_metrics = None
    if not args.skip_loso:
        loso_metrics = run_loso(raw,thresholds,args,out)
        print_vertical_reports(full_metrics,loso_metrics,out)
    else:
        warnings.warn('LOSO was skipped; legacy metric CSV files were still written.')
    print(f'Completed: {out.resolve()}',flush=True)


def predict_command(args):
    import xgboost as xgb
    directory = Path(args.model)
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    if pd.Timestamp(args.issue_date) < pd.Timestamp(config['val_end']):
        raise ValueError('Issue date predates availability of the tuned configuration.')
    raw = read_data(args.csv,require_obs=False)
    frame, features = build_features(raw,config['lead_days'],config['gpm_delay_days'],config['history_days'],[args.issue_date])
    if features != config['features']:
        raise ValueError('Feature schema mismatch.')
    if frame[[f'gpm_lag_{i}' for i in range(config['history_days'])]].isna().all(axis=1).any():
        raise ValueError('A location has no GPM history for this issue date.')
    pair = xgb.XGBClassifier(),xgb.XGBRegressor()
    pair[0].load_model(directory/'classifier.json')
    pair[1].load_model(directory/'regressor.json')
    pred, prob = prediction(pair,frame,features,config['threshold'])
    result = frame[['station_name','issue_date','valid_date','latest_gpm_date']].copy()
    result['prediction_mm_day'],result['wet_probability'] = pred,prob
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.out,index=False,encoding='utf-8-sig')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command',required=True)
    t = commands.add_parser('train')
    t.add_argument('--csv',required=True)
    t.add_argument('--out',required=True)
    t.add_argument('--leads',type=int,nargs='+',default=[0,1,3,5])
    t.add_argument('--gpm-delay-days',type=int,default=0,
                   help='GPM availability delay at issue time; 0 preserves the previous experiment.')
    t.add_argument('--history-days',type=int,default=4)
    t.add_argument('--train-start',default='2012-01-01')
    t.add_argument('--train-end',default='2019-12-31')
    t.add_argument('--val-end',default='2021-12-31')
    t.add_argument('--test-end',default='2024-12-31')
    t.add_argument('--wet-threshold',type=float,default=0.1)
    t.add_argument('--trials',type=int,default=25)
    t.add_argument('--seed',type=int,default=42)
    t.add_argument('--jobs',type=int,default=4)
    t.add_argument('--skip-loso',action='store_true',
                   help='Skip strict station-held-out diagnostics (LOSO runs by default).')
    q = commands.add_parser('predict')
    q.add_argument('--csv',required=True)
    q.add_argument('--model',required=True)
    q.add_argument('--issue-date',required=True)
    q.add_argument('--out',required=True)
    return p


def button_run_arguments():
    """Build defaults when VS Code's Run Python File button passes no arguments."""
    script_dir = Path(__file__).resolve().parent
    file_name = 'qingxiduiqi_2012_2024.csv'
    candidates = [
        script_dir/'shicezhandianshuju'/file_name,
        script_dir.parent/'shicezhandianshuju'/file_name,
        script_dir.parent.parent/'shicezhandianshuju'/file_name,
        Path.cwd()/'shicezhandianshuju'/file_name,
        Path.cwd()/'jiangshuironghe-MOE'/'shicezhandianshuju'/file_name,
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        searched = '\n'.join(f'  - {p}' for p in candidates)
        raise FileNotFoundError(
            '点击运行按钮后未自动找到 qingxiduiqi_2012_2024.csv。\n'
            '已检查以下位置：\n' + searched +
            '\n请将数据放在项目的 shicezhandianshuju 文件夹中。')

    base_out = script_dir/'results_m1_0135_calendar_p90_loso'
    # Never destroy an earlier experiment. A repeated button click receives a
    # timestamped directory while the first run keeps the concise base name.
    if base_out.exists():
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = script_dir/f'{base_out.name}_{stamp}'
    else:
        out_path = base_out

    print('='*110)
    print('VS Code 运行按钮模式：未检测到命令行参数，自动执行完整训练。')
    print(f'输入数据：{csv_path}')
    print(f'输出目录：{out_path}')
    print('预见期：Lead 0、1、3、5；LOSO：开启；Optuna：25次。')
    print('训练完成后将在本终端依次打印表1—表5。')
    print('='*110, flush=True)
    return parser().parse_args([
        'train', '--csv', str(csv_path), '--out', str(out_path),
    ])


if __name__ == '__main__':
    args = button_run_arguments() if len(sys.argv) == 1 else parser().parse_args()
    if args.command == 'train':
        train_command(args)
    else:
        predict_command(args)
