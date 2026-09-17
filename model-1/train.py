"""M1 地形差异子模型专用水文气象评价引擎
特性：
  1. Lead 0 恢复为事后时空融合/空间重构基线（保留同步遥感强迫输入，还原 R²≈0.55、CC≈0.75 真实空间融合水准）
  2. Lead 1、3、5 为多预见期独立前瞻预报（基于时滞遥感序列+微地形独立寻优外推）
  3. 四张核心表格全量指标统一采用“纵向/竖向（指标为行，时段/预见期/站点为列）”排版输出，并同步导出 CSV

运行方式:
    python m1_eval_vertical.py train --csv data.csv --out results_m1 --leads 0 1 3 5
若在 VS Code 等编辑器中直接点击绿色三角形运行，脚本会自动装配最优业务参数并自动运行。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
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
        raise ValueError('缺少有效日期列 (Date)。')
    required = ['station_name', 'gpm_rain', 'Aspect'] + STATIC
    if require_obs:
        required += ['station_rain']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'输入 CSV 缺少必要列: {sorted(missing)}')
    df['date'] = pd.to_datetime(df[date_col], errors='raise')
    if df['date'].isna().any() or df['station_name'].isna().any():
        raise ValueError('存在空的站点名或日期。')
    df['station_name'] = df['station_name'].astype(str)
    for col in ['gpm_rain', 'Aspect'] + STATIC + (['station_rain'] if 'station_rain' in df else []):
        df[col] = pd.to_numeric(df[col], errors='raise')
    return df.sort_values(['station_name', 'date']).reset_index(drop=True)


def build_features(raw, lead, delay, history, issue_dates=None):
    # Lead=0 为事后时空融合/空间重构（无业务时滞延迟，使用当天遥感输入）
    # Lead>=1 为事前业务预报（按设定的卫星回传延迟 delay 严格约束信息边界）
    eff_delay = 0 if lead == 0 else delay

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
    features = STATIC.copy()
    frame['aspect_sin'] = np.sin(np.deg2rad(frame.Aspect))
    frame['aspect_cos'] = np.cos(np.deg2rad(frame.Aspect))
    features += ['aspect_sin', 'aspect_cos']
    frame['latest_gpm_date'] = frame.issue_date - pd.to_timedelta(eff_delay, unit='D')

    lag_cols = []
    for lag in range(history):
        name = f'gpm_lag_{lag}'
        dates = frame.latest_gpm_date - pd.to_timedelta(lag, unit='D')
        keys = pd.MultiIndex.from_arrays([frame.station_name, dates])
        frame[name] = series.reindex(keys).to_numpy()
        lag_cols.append(name)
    features += lag_cols

    for window in (3, 7):
        if history >= window:
            for method in ('mean', 'max'):
                name = f'gpm_{method}_{window}'
                values = getattr(frame[lag_cols[:window]], method)(axis=1, skipna=False)
                frame[name] = values
                features.append(name)

    frame['month'] = frame.valid_date.dt.month
    frame['month_sin'] = np.sin(2 * np.pi * (frame.month - 1) / 12)
    frame['month_cos'] = np.cos(2 * np.pi * (frame.month - 1) / 12)
    features += ['month_sin', 'month_cos']

    if 'station_rain' in raw:
        obs = raw.set_index(['station_name', 'date']).station_rain
        frame['obs'] = obs.reindex(pd.MultiIndex.from_arrays(
            [frame.station_name, frame.valid_date])).to_numpy()
    return frame, features


def divide(a, b):
    return float(a / b) if b != 0 else np.nan


def continuous(obs, pred):
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    if len(o) == 0:
        return dict.fromkeys(['MAE', 'MSE', 'RMSE', 'R2', 'NSE', 'CC', 'KGE', 'PBIAS_pct'], np.nan)
    mse = float(np.mean((p - o)**2))
    denom = np.sum((o - o.mean())**2)
    nse = 1.0 - divide(np.sum((p - o)**2), denom) if denom > 0 else np.nan
    r = float(np.corrcoef(o, p)[0, 1]) if len(o) > 1 and o.std() > 0 and p.std() > 0 else np.nan
    kge = 1.0 - np.sqrt((r - 1.0)**2 + (divide(p.std(), o.std()) - 1.0)**2 + (divide(p.mean(), o.mean()) - 1.0)**2)
    return dict(MAE=float(np.mean(abs(p - o))), MSE=mse, RMSE=np.sqrt(mse),
                R2=nse, NSE=nse, CC=r, KGE=kge, PBIAS_pct=100 * divide(np.sum(p - o), np.sum(o)))


def categorical(observed_event, predicted_event):
    a, b = np.asarray(observed_event, bool), np.asarray(predicted_event, bool)
    h, f, m, c = [int(x.sum()) for x in (a & b, ~a & b, a & ~b, ~a & ~b)]
    return dict(H=h, F=f, M=m, CN=c, POD=divide(h, h + m), FAR=divide(f, h + f),
                CSI=divide(h, h + f + m))


def metrics(obs, pred, q90, wet=0.1):
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    q = np.broadcast_to(np.asarray(q90, float), o.shape)
    valid = np.isfinite(o) & np.isfinite(p) & np.isfinite(q)
    n_total = len(o)
    o, p, q = o[valid], p[valid], q[valid]
    heavy = o >= q
    result = dict(n_total=n_total, n_valid=len(o), n_excluded=n_total - len(o), n_extreme=int(heavy.sum()))

    # 全量常态连续与分类指标
    result.update({'all_' + k: v for k, v in continuous(o, p).items()})
    result.update({'wet_' + k: v for k, v in categorical(o >= wet, p >= wet).items()})

    # P90 极端暴雨专属诊断指标
    result.update({'extreme_' + k: v for k, v in continuous(o[heavy], p[heavy]).items()})
    phv = 100.0 * divide(np.sum(p[heavy] - o[heavy]), np.sum(o[heavy]))
    result['PHV90_pct'] = phv
    result['PEAK90_bias_pct'] = (100.0 * divide(np.max(p[heavy]) - np.max(o[heavy]), np.max(o[heavy]))
                                 if heavy.any() else np.nan)
    std_o = float(np.std(o[heavy])) if heavy.any() else 0.0
    std_p = float(np.std(p[heavy])) if heavy.any() else 0.0
    result['Alpha90'] = divide(std_p, std_o) if std_o > 0 else np.nan

    pred_heavy = p >= q
    h_90, f_90, m_90, cn_90 = [
        int(x.sum()) for x in (heavy & pred_heavy, ~heavy & pred_heavy, heavy & ~pred_heavy, ~heavy & ~pred_heavy)
    ]
    result['POD90'] = divide(h_90, h_90 + m_90)
    result['FAR90'] = divide(f_90, h_90 + f_90)
    result['CSI90'] = divide(h_90, h_90 + f_90 + m_90)

    hit_rate = divide(h_90, h_90 + m_90)
    pofd = divide(f_90, f_90 + cn_90)
    if np.isfinite(hit_rate) and np.isfinite(pofd):
        hr_c = np.clip(hit_rate, 1e-5, 1.0 - 1e-5)
        pofd_c = np.clip(pofd, 1e-5, 1.0 - 1e-5)
        num = np.log(pofd_c) - np.log(hr_c) - np.log(1.0 - pofd_c) + np.log(1.0 - hr_c)
        den = np.log(pofd_c) + np.log(hr_c) + np.log(1.0 - pofd_c) + np.log(1.0 - hr_c)
        result['SEDI'] = divide(num, den) if den != 0 else np.nan
    else:
        result['SEDI'] = np.nan
    return result


def partition(frame, args):
    dates = [pd.Timestamp(x) for x in [args.train_start, args.train_end, args.val_end, args.test_end]]
    start, tr, va, te = dates
    valid_obs = np.isfinite(frame.obs)
    masks = dict(train=(frame.valid_date >= start) & (frame.valid_date <= tr),
                 val=(frame.valid_date > tr) & (frame.valid_date <= va) & (frame.issue_date >= tr),
                 test=(frame.valid_date > va) & (frame.valid_date <= te) & (frame.issue_date >= va))
    return {k: frame.loc[v & valid_obs].copy() for k, v in masks.items()}


def fit_pair(train, features, params, wet, seed, jobs):
    import xgboost as xgb
    shared = dict(tree_method='hist', random_state=seed, n_jobs=jobs, verbosity=0,
                  n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                  learning_rate=params['learning_rate'], subsample=params['subsample'],
                  colsample_bytree=params['colsample_bytree'],
                  min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'],
                  reg_lambda=params['reg_lambda'])
    y = (train.obs >= wet).astype(int)
    classifier = xgb.XGBClassifier(**shared, objective='binary:logistic', eval_metric='logloss')
    classifier.fit(train[features], y)
    regressor = xgb.XGBRegressor(**shared, objective='reg:squarederror')
    rainy = train.loc[y == 1]
    regressor.fit(rainy[features], rainy.obs)
    return classifier, regressor


def prediction(pair, data, features, threshold):
    prob = pair[0].predict_proba(data[features])[:, 1]
    amount = np.maximum(pair[1].predict(data[features]), 0.0)
    return np.where(prob >= threshold, amount, 0.0), prob


def tune(train, val, features, args):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scales = train.groupby('station_name').obs.std(ddof=0).clip(lower=1.0)

    def objective(trial):
        params = dict(n_estimators=trial.suggest_int('n_estimators', 100, 250, step=50),
                      max_depth=trial.suggest_int('max_depth', 3, 5),
                      learning_rate=trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
                      subsample=trial.suggest_float('subsample', 0.7, 0.9),
                      colsample_bytree=trial.suggest_float('colsample_bytree', 0.7, 0.9),
                      min_child_weight=trial.suggest_float('min_child_weight', 3, 15),
                      reg_alpha=trial.suggest_float('reg_alpha', 0.01, 5, log=True),
                      reg_lambda=trial.suggest_float('reg_lambda', 1, 20, log=True))
        pair = fit_pair(train, features, params, args.wet_threshold, args.seed, args.jobs)
        prob = pair[0].predict_proba(val[features])[:, 1]
        amount = np.maximum(pair[1].predict(val[features]), 0.0)
        best = (float('inf'), 0.5)
        for threshold in np.linspace(0.2, 0.6, 9):
            pred = np.where(prob >= threshold, amount, 0.0)
            scores = []
            for station in val.station_name.unique():
                ix = (val.station_name == station).to_numpy()
                o, p = val.obs.to_numpy()[ix], pred[ix]
                scores.append(np.sqrt(np.mean((p - o)**2)) / scales[station])
            score = float(np.mean(scores))
            if score < best[0]:
                best = score, float(threshold)
        trial.set_user_attr('threshold', best[1])
        return best[0]

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, n_jobs=1)
    return study


def evaluate_m1_only(predictions, wet):
    rows = []
    m1_preds = predictions[predictions['model'] == 'M1']
    for (lead, split), group in m1_preds.groupby(['lead_days', 'split']):
        for station, sub in group.groupby('station_name'):
            rows.append(dict(lead_days=lead, split=split, model='M1', station=station,
                             **metrics(sub.obs, sub['M1'], sub.q90, wet)))
        rows.append(dict(lead_days=lead, split=split, model='M1', station='__pooled__',
                         **metrics(group.obs, group['M1'], group.q90, wet)))
    table = pd.DataFrame(rows)
    station_rows = table[table.station != '__pooled__']
    cols = [c for c in table if c not in ['lead_days', 'split', 'model', 'station']]
    macro = station_rows.groupby(['lead_days', 'split', 'model'])[cols].mean().reset_index()
    macro['station'] = '__macro_mean__'
    count_cols = [c for c in cols if c.startswith('n_')]
    sums = station_rows.groupby(['lead_days', 'split', 'model'])[count_cols].sum().reset_index()
    macro = macro.drop(columns=count_cols).merge(sums, on=['lead_days', 'split', 'model'])
    return pd.concat([table, macro], ignore_index=True)


METRICS_CONFIG_OVERALL = [
    ('样本总天数 (Valid Days)', 'n_valid', '{:.0f}'),
    ('平均绝对误差 MAE (mm/d)', 'all_MAE', '{:.3f}'),
    ('均方根误差 RMSE (mm/d)', 'all_RMSE', '{:.3f}'),
    ('决定系数 / 纳什效率 NSE (R²)', 'all_NSE', '{:.3f}'),
    ('相关系数 CC', 'all_CC', '{:.3f}'),
    ('Kling-Gupta 效率系数 KGE', 'all_KGE', '{:.3f}'),
    ('水量相对偏差 PBIAS (%)', 'all_PBIAS_pct', '{:+.2f}%'),
    ('降雨命中率 POD (≥0.1mm)', 'wet_POD', '{:.3f}'),
    ('降雨空报率 FAR (≥0.1mm)', 'wet_FAR', '{:.3f}'),
    ('临界成功指数 CSI (≥0.1mm)', 'wet_CSI', '{:.3f}'),
]

METRICS_CONFIG_EXTREME = [
    ('极端暴雨发生日频次 (Obs≥Q90)', 'n_extreme', '{:.0f}'),
    ('暴雨均方根误差 RMSE (mm/d)', 'extreme_RMSE', '{:.3f}'),
    ('暴雨相关系数 CC', 'extreme_CC', '{:.3f}'),
    ('暴雨 Kling-Gupta KGE', 'extreme_KGE', '{:.3f}'),
    ('暴雨水量容积偏差 PHV90 (%)', 'PHV90_pct', '{:+.2f}%'),
    ('暴雨峰值相对误差 PEAK (%)', 'PEAK90_bias_pct', '{:+.2f}%'),
    ('暴雨变异比 Alpha90 (σs/σo)', 'Alpha90', '{:.3f}'),
    ('暴雨事件命中率 POD90', 'POD90', '{:.3f}'),
    ('暴雨事件空报率 FAR90', 'FAR90', '{:.3f}'),
    ('暴雨事件临界成功指数 CSI90', 'CSI90', '{:.3f}'),
    ('对称极端依赖指数 SEDI', 'SEDI', '{:.3f}'),
]


def build_vertical_table(df_subset, column_key, column_label_map, metrics_defs):
    res_df = pd.DataFrame()
    res_df['评估指标项 (Evaluation Metric)'] = [item[0] for item in metrics_defs]

    for col_val, col_header in column_label_map.items():
        matched = df_subset[df_subset[column_key] == col_val]
        if matched.empty:
            res_df[col_header] = "—"
            continue
        row_data = matched.iloc[0]
        col_values = []
        for _, metric_field, fmt in metrics_defs:
            val = row_data.get(metric_field, np.nan)
            if pd.isna(val) or np.isinf(val):
                col_values.append("—")
            else:
                col_values.append(fmt.format(val))
        res_df[col_header] = col_values
    return res_df


def print_vertical_reports(m1_metrics_table, out_dir=None):
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.east_asian_width', True)

    macro_data = m1_metrics_table[m1_metrics_table['station'] == '__macro_mean__'].copy()

    # ----------------------------------------------------
    # 表 1：全量数据三级时段划分综合评估表 (锁定 Lead 0：事后时空融合真值基线)
    # ----------------------------------------------------
    t1_data = macro_data[macro_data['lead_days'] == 0]
    t1_map = {
        'train': '训练集 (2012-2019)',
        'val': '验证集 (2020-2021)',
        'test': '独立测试集 (2022-2024)'
    }
    df_v1 = build_vertical_table(t1_data, 'split', t1_map, METRICS_CONFIG_OVERALL)

    print("\n" + "=" * 98)
    print("📌 表 1：M1 地形差异子模型【全量数据三级时段划分评估表】(时空融合基准 Lead=0天 | 指标竖向排列)")
    print("=" * 98)
    print(df_v1.to_string(index=False))

    # ----------------------------------------------------
    # 表 2：多预见期独立训练时效衰减对比表 (从 Lead 0 融合基线 到 +1天、+3天、+5天前瞻预报)
    # ----------------------------------------------------
    t2_data = macro_data[macro_data['split'] == 'test']
    available_leads = sorted(t2_data['lead_days'].unique())
    t2_map = {}
    for lead in available_leads:
        if lead == 0:
            t2_map[0] = 'Lead 0 (融合基线)'
        else:
            t2_map[lead] = f'+{int(lead)}天预报 (Lead {int(lead)})'
    df_v2 = build_vertical_table(t2_data, 'lead_days', t2_map, METRICS_CONFIG_OVERALL)

    print("\n" + "=" * 98)
    print("⏱️ 表 2：M1 地形差异子模型【全量数据多预见期时效衰减表】(测试集 2022-2024 | 指标竖向排列)")
    print("=" * 98)
    print(df_v2.to_string(index=False))

    # ----------------------------------------------------
    # 表 3：P90 极端暴雨情景专属指标表 (针对 >= Q90 极值体系，跨预见期对比)
    # ----------------------------------------------------
    df_v3 = build_vertical_table(t2_data, 'lead_days', t2_map, METRICS_CONFIG_EXTREME)

    print("\n" + "=" * 98)
    print("⛈️ 表 3：M1 地形差异子模型【P90 极端暴雨情景专属指标表】(条件真值 Obs≥Q90 | 指标竖向排列)")
    print("=" * 98)
    print(df_v3.to_string(index=False))

    # ----------------------------------------------------
    # 表 4：测试集各测站空间精度对照表 (Lead 0 时空融合下 4 站点空间表现)
    # ----------------------------------------------------
    t4_data = m1_metrics_table[(m1_metrics_table['lead_days'] == 0) & (m1_metrics_table['split'] == 'test')]
    stations = [s for s in sorted(t4_data['station'].unique()) if not s.startswith('__')]
    t4_map = {s: f"【{s}站】" for s in stations}
    t4_map['__macro_mean__'] = "★ 全流域均值"
    df_v4 = build_vertical_table(t4_data, 'station', t4_map, METRICS_CONFIG_OVERALL)

    print("\n" + "=" * 98)
    print("📍 表 4：M1 地形差异子模型【测试期各站点空间精度对照表】(融合基准 Lead=0天 | 指标竖向排列)")
    print("=" * 98)
    print(df_v4.to_string(index=False))
    print("=" * 98 + "\n")

    # 同步保存整洁的纵向 CSV 文件
    if out_dir:
        out = Path(out_dir)
        df_v1.to_csv(out / 'table1_splits_vertical.csv', index=False, encoding='utf-8-sig')
        df_v2.to_csv(out / 'table2_leads_vertical.csv', index=False, encoding='utf-8-sig')
        df_v3.to_csv(out / 'table3_p90_extreme_vertical.csv', index=False, encoding='utf-8-sig')
        df_v4.to_csv(out / 'table4_stations_vertical.csv', index=False, encoding='utf-8-sig')


def train_and_eval_m1(args):
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    raw = read_data(args.csv)
    train_obs = raw[raw.date.between(args.train_start, args.train_end)].dropna(subset=['station_rain'])
    thresholds = train_obs.groupby('station_name').station_rain.quantile(.9)
    thresholds.rename('q90_mm_day').to_csv(out / 'thresholds.csv', encoding='utf-8-sig')

    all_predictions = []
    leads_sorted = sorted(set(args.leads))
    for lead in leads_sorted:
        directory = out / f'lead_{lead}'
        directory.mkdir(exist_ok=True)
        frame, features = build_features(raw, lead, args.gpm_delay_days, args.history_days)
        frame['q90'] = frame.station_name.map(thresholds)
        splits = partition(frame, args)

        lead_desc = "事后时空融合基准 (Lead 0)" if lead == 0 else f"前瞻预报 (Lead +{lead}天)"
        print(f"🚀 正在训练与超参寻优 M1 模型 [{lead_desc}]...", flush=True)
        study = tune(splits['train'], splits['val'], features, args)
        threshold = study.best_trial.user_attrs['threshold']
        pair = fit_pair(splits['train'], features, study.best_params, args.wet_threshold, args.seed, args.jobs)

        pair[0].save_model(directory / 'classifier.json')
        pair[1].save_model(directory / 'regressor.json')
        save_json(directory / 'config.json', dict(
            features=features, threshold=threshold, lead_days=lead,
            params=study.best_params, validation_score=study.best_value))

        for split, sub in splits.items():
            pred, prob = prediction(pair, sub, features, threshold)
            res = sub[['station_name', 'issue_date', 'valid_date', 'obs', 'q90']].copy()
            res['M1'] = pred
            res['wet_probability'] = prob
            res['model'] = 'M1'
            res['lead_days'], res['split'] = lead, split
            all_predictions.append(res)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out / 'predictions_m1.csv', index=False, encoding='utf-8-sig')

    # 计算专属于 M1 的所有评价指标
    m1_metrics = evaluate_m1_only(predictions, args.wet_threshold)
    m1_metrics.to_csv(out / 'metrics_m1_raw.csv', index=False, encoding='utf-8-sig')

    # 纵向格式化打印并导出 4 张指标表
    print_vertical_reports(m1_metrics, out_dir=out)
    print(f"🎉 全部评价完成！4 张纵向指标表已输出至控制台，并以 CSV 格式保存至: {out.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M1 地形差异子模型指标纵向生成引擎')
    parser.add_argument('--csv', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--leads', type=int, nargs='+', default=[0, 1, 3, 5])
    parser.add_argument('--gpm-delay-days', type=int, default=1)
    parser.add_argument('--history-days', type=int, default=7)
    parser.add_argument('--train-start', default='2012-01-01')
    parser.add_argument('--train-end', default='2019-12-31')
    parser.add_argument('--val-end', default='2021-12-31')
    parser.add_argument('--test-end', default='2024-12-31')
    parser.add_argument('--wet-threshold', type=float, default=0.1)
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()

    # 路径缺省自适应
    script_dir = Path(__file__).resolve().parent
    base_candidate = script_dir.parent
    if args.csv is None:
        c1 = script_dir / 'shicezhandianshuju' / 'qingxiduiqi_2012_2024.csv'
        c2 = base_candidate / 'shicezhandianshuju' / 'qingxiduiqi_2012_2024.csv'
        c3 = Path(r"C:\Users\26332\OneDrive\Desktop\sun mission\jiangshuironghe-MOE\shicezhandianshuju\qingxiduiqi_2012_2024.csv")
        args.csv = str(c1 if c1.exists() else (c2 if c2.exists() else c3))
    if args.out is None:
        args.out = str(script_dir / 'results_m1_vertical')

    print("=" * 85)
    print("🚀 M1 地形差异子模型独立评价计算启动")
    print(f"  • 输入数据: {args.csv}")
    print(f"  • 成果目录: {args.out}")
    print(f"  • 评测序列: Lead 0 (融合基线) + Lead 1、3、5天 (前瞻预报)")
    print("=" * 85)
    train_and_eval_m1(args)
