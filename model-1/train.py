"""M1 地形差异子模型专用水文气象评价引擎 (改进版)
核心修正：
  1. 纠正预见期偏移机制：基于台站时序平移 (shift(-lead)) 构建监督样本对，规范 Lead 0/1/3/5 逻辑
  2. 剔除多余时滞损耗：起报日 t 严格获取至 t 日已知卫星观测，真实评估未来 1、3、5 天前瞻性能
  3. 保留四张标准纵向指标表生成与 CSV 导出功能
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

STATIC_COLS = ['DEM', 'Slope', 'lat', 'lon']


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_data(path):
    df = pd.read_csv(path)
    date_col = next((c for c in ['Date', 'date', 'time', 'datetime', 'DATE'] if c in df), None)
    if date_col is None:
        raise ValueError('输入 CSV 缺少有效日期列 (Date)。')
    
    required = ['station_name', 'gpm_rain', 'Aspect', 'station_rain'] + STATIC_COLS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'输入 CSV 缺少必要列: {sorted(missing)}')
        
    df['date'] = pd.to_datetime(df[date_col], errors='raise')
    df['station_name'] = df['station_name'].astype(str)
    
    for col in ['gpm_rain', 'Aspect', 'station_rain'] + STATIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='raise')
        
    return df.sort_values(['station_name', 'date']).reset_index(drop=True)


def build_lead_dataset(raw_df, lead):
    """
    根据预见期 lead 构建特征集与预测目标：
    - 起报日 t：获取 t 日当天的 gpm_rain 以及前期滞后水汽记忆
    - 目标日 t + lead：实测降水标签 obs = station_rain(t + lead)
    """
    station_dfs = []
    
    for station, group in raw_df.groupby('station_name'):
        g = group.sort_values('date').copy()
        
        # 1. 明确时间锚点
        g['issue_date'] = g['date']
        g['valid_date'] = g['date'] + pd.Timedelta(days=lead)
        
        # 2. 预测目标标签向后位移 lead 天（即当天的行拟合未来第 lead 天的实测雨量）
        g['obs'] = g['station_rain'].shift(-lead)
        
        # 3. 构造起报日已知的遥感特征与时序滞后记忆
        g['gpm_curr'] = g['gpm_rain']
        g['gpm_lag_1'] = g['gpm_rain'].shift(1)
        g['gpm_lag_2'] = g['gpm_rain'].shift(2)
        g['gpm_lag_3'] = g['gpm_rain'].shift(3)
        g['gpm_mean_3'] = g[['gpm_curr', 'gpm_lag_1', 'gpm_lag_2']].mean(axis=1)
        
        # 4. 目标预报日的气候月份特征
        g['target_month'] = g['valid_date'].dt.month
        g['month_sin'] = np.sin(2 * np.pi * (g['target_month'] - 1) / 12)
        g['month_cos'] = np.cos(2 * np.pi * (g['target_month'] - 1) / 12)
        
        # 5. 坡向正余弦物理分解
        g['aspect_sin'] = np.sin(np.deg2rad(g['Aspect']))
        g['aspect_cos'] = np.cos(np.deg2rad(g['Aspect']))
        
        # 剔除末尾由于 shift 导致的无效样本
        g = g.dropna(subset=['obs', 'gpm_curr', 'gpm_lag_1', 'gpm_lag_2', 'gpm_lag_3'])
        station_dfs.append(g)
        
    combined = pd.concat(station_dfs, ignore_index=True)
    
    # 确定输入特征列表
    if lead == 0:
        # Lead 0 (事后时空融合)：聚焦于当天同步遥感强迫与微地形的校正
        features = ['gpm_curr', 'DEM', 'Slope', 'aspect_sin', 'aspect_cos', 'lat', 'lon', 'month_sin', 'month_cos']
    else:
        # Lead >= 1 (前瞻预报)：引入前期水汽滞后记忆，捕捉演变动量
        features = ['gpm_curr', 'gpm_lag_1', 'gpm_lag_2', 'gpm_lag_3', 'gpm_mean_3',
                    'DEM', 'Slope', 'aspect_sin', 'aspect_cos', 'lat', 'lon', 'month_sin', 'month_cos']
                    
    return combined, features


def partition(frame, args):
    start = pd.Timestamp(args.train_start)
    tr = pd.Timestamp(args.train_end)
    va = pd.Timestamp(args.val_end)
    te = pd.Timestamp(args.test_end)
    
    valid_obs = np.isfinite(frame['obs'])
    
    masks = {
        'train': (frame['valid_date'] >= start) & (frame['valid_date'] <= tr),
        'val':   (frame['valid_date'] > tr) & (frame['valid_date'] <= va) & (frame['issue_date'] >= tr),
        'test':  (frame['valid_date'] > va) & (frame['valid_date'] <= te) & (frame['issue_date'] >= va)
    }
    return {k: frame.loc[v & valid_obs].copy() for k, v in masks.items()}


def fit_pair(train, features, params, wet, seed, jobs):
    shared = dict(tree_method='hist', random_state=seed, n_jobs=jobs, verbosity=0,
                  n_estimators=params['n_estimators'], max_depth=params['max_depth'],
                  learning_rate=params['learning_rate'], subsample=params['subsample'],
                  colsample_bytree=params['colsample_bytree'],
                  min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'],
                  reg_lambda=params['reg_lambda'])
                  
    y = (train['obs'] >= wet).astype(int)
    classifier = xgb.XGBClassifier(**shared, objective='binary:logistic', eval_metric='logloss')
    classifier.fit(train[features], y)
    
    regressor = xgb.XGBRegressor(**shared, objective='reg:squarederror')
    rainy = train.loc[y == 1]
    regressor.fit(rainy[features], rainy['obs'])
    
    return classifier, regressor


def tune(train, val, features, args):
    scales = train.groupby('station_name')['obs'].std(ddof=0).clip(lower=1.0)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 80, 200, step=40),
            max_depth=trial.suggest_int('max_depth', 3, 6),
            learning_rate=trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
            subsample=trial.suggest_float('subsample', 0.65, 0.9),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.65, 0.9),
            min_child_weight=trial.suggest_float('min_child_weight', 2, 12),
            reg_alpha=trial.suggest_float('reg_alpha', 0.01, 5, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 0.5, 15, log=True)
        )
        pair = fit_pair(train, features, params, args.wet_threshold, args.seed, args.jobs)
        prob = pair[0].predict_proba(val[features])[:, 1]
        amount = np.maximum(pair[1].predict(val[features]), 0.0)
        
        best = (float('inf'), 0.5)
        for threshold in np.linspace(0.25, 0.55, 7):
            pred = np.where(prob >= threshold, amount, 0.0)
            scores = []
            for station in val['station_name'].unique():
                ix = (val['station_name'] == station).to_numpy()
                o, p = val['obs'].to_numpy()[ix], pred[ix]
                scores.append(np.sqrt(np.mean((p - o)**2)) / scales[station])
            score = float(np.mean(scores))
            if score < best[0]:
                best = score, float(threshold)
                
        trial.set_user_attr('threshold', best[1])
        return best[0]

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.trials, n_jobs=1)
    return study


def prediction(pair, data, features, threshold):
    prob = pair[0].predict_proba(data[features])[:, 1]
    amount = np.maximum(pair[1].predict(data[features]), 0.0)
    return np.where(prob >= threshold, amount, 0.0), prob


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
    return dict(H=h, F=f, M=m, CN=c, POD=divide(h, h + m), FAR=divide(f, h + f), CSI=divide(h, h + f + m))


def metrics(obs, pred, q90, wet=0.1):
    o, p = np.asarray(obs, float), np.asarray(pred, float)
    q = np.broadcast_to(np.asarray(q90, float), o.shape)
    valid = np.isfinite(o) & np.isfinite(p) & np.isfinite(q)
    n_total = len(o)
    o, p, q = o[valid], p[valid], q[valid]
    heavy = o >= q
    result = dict(n_total=n_total, n_valid=len(o), n_excluded=n_total - len(o), n_extreme=int(heavy.sum()))

    result.update({'all_' + k: v for k, v in continuous(o, p).items()})
    result.update({'wet_' + k: v for k, v in categorical(o >= wet, p >= wet).items()})

    result.update({'extreme_' + k: v for k, v in continuous(o[heavy], p[heavy]).items()})
    result['PHV90_pct'] = 100.0 * divide(np.sum(p[heavy] - o[heavy]), np.sum(o[heavy]))
    result['PEAK90_bias_pct'] = (100.0 * divide(np.max(p[heavy]) - np.max(o[heavy]), np.max(o[heavy])) if heavy.any() else np.nan)
    std_o = float(np.std(o[heavy])) if heavy.any() else 0.0
    std_p = float(np.std(p[heavy])) if heavy.any() else 0.0
    result['Alpha90'] = divide(std_p, std_o) if std_o > 0 else np.nan

    pred_heavy = p >= q
    h_90, f_90, m_90, cn_90 = [int(x.sum()) for x in (heavy & pred_heavy, ~heavy & pred_heavy, heavy & ~pred_heavy, ~heavy & ~pred_heavy)]
    result['POD90'] = divide(h_90, h_90 + m_90)
    result['FAR90'] = divide(f_90, h_90 + f_90)
    result['CSI90'] = divide(h_90, h_90 + f_90 + m_90)

    hit_rate = divide(h_90, h_90 + m_90)
    pofd = divide(f_90, f_90 + cn_90)
    if np.isfinite(hit_rate) and np.isfinite(pofd):
        hr_c, pofd_c = np.clip(hit_rate, 1e-5, 1.0 - 1e-5), np.clip(pofd, 1e-5, 1.0 - 1e-5)
        num = np.log(pofd_c) - np.log(hr_c) - np.log(1.0 - pofd_c) + np.log(1.0 - hr_c)
        den = np.log(pofd_c) + np.log(hr_c) + np.log(1.0 - pofd_c) + np.log(1.0 - hr_c)
        result['SEDI'] = divide(num, den) if den != 0 else np.nan
    else:
        result['SEDI'] = np.nan
    return result


def evaluate_m1_only(predictions, wet):
    rows = []
    m1_preds = predictions[predictions['model'] == 'M1']
    for (lead, split), group in m1_preds.groupby(['lead_days', 'split']):
        for station, sub in group.groupby('station_name'):
            rows.append(dict(lead_days=lead, split=split, model='M1', station=station,
                             **metrics(sub['obs'], sub['M1'], sub['q90'], wet)))
        rows.append(dict(lead_days=lead, split=split, model='M1', station='__pooled__',
                         **metrics(group['obs'], group['M1'], group['q90'], wet)))
                         
    table = pd.DataFrame(rows)
    station_rows = table[table['station'] != '__pooled__']
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

    # 表 1：三级时段划分评估表 (Lead 0)
    t1_data = macro_data[macro_data['lead_days'] == 0]
    t1_map = {'train': '训练集 (2012-2019)', 'val': '验证集 (2020-2021)', 'test': '独立测试集 (2022-2024)'}
    df_v1 = build_vertical_table(t1_data, 'split', t1_map, METRICS_CONFIG_OVERALL)

    print("\n" + "=" * 98)
    print("📌 表 1：M1 地形差异子模型【全量数据三级时段划分评估表】(时空融合基准 Lead=0天 | 指标竖向排列)")
    print("=" * 98)
    print(df_v1.to_string(index=False))

    # 表 2：多预见期时效衰减表 (测试集)
    t2_data = macro_data[macro_data['split'] == 'test']
    available_leads = sorted(t2_data['lead_days'].unique())
    t2_map = {lead: ('Lead 0 (融合基线)' if lead == 0 else f'+{int(lead)}天预报 (Lead {int(lead)})') for lead in available_leads}
    df_v2 = build_vertical_table(t2_data, 'lead_days', t2_map, METRICS_CONFIG_OVERALL)

    print("\n" + "=" * 98)
    print("⏱️ 表 2：M1 地形差异子模型【全量数据多预见期时效衰减表】(测试集 2022-2024 | 指标竖向排列)")
    print("=" * 98)
    print(df_v2.to_string(index=False))

    # 表 3：P90 极端暴雨情景专属指标表
    df_v3 = build_vertical_table(t2_data, 'lead_days', t2_map, METRICS_CONFIG_EXTREME)

    print("\n" + "=" * 98)
    print("⛈️ 表 3：M1 地形差异子模型【P90 极端暴雨情景专属指标表】(条件真值 Obs≥Q90 | 指标竖向排列)")
    print("=" * 98)
    print(df_v3.to_string(index=False))

    # 表 4：各测站空间精度对照表 (Lead 0 测试期)
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
    train_obs = raw[raw['date'].between(args.train_start, args.train_end)].dropna(subset=['station_rain'])
    thresholds = train_obs.groupby('station_name')['station_rain'].quantile(.9)
    thresholds.rename('q90_mm_day').to_csv(out / 'thresholds.csv', encoding='utf-8-sig')

    all_predictions = []
    leads_sorted = sorted(set(args.leads))
    
    for lead in leads_sorted:
        directory = out / f'lead_{lead}'
        directory.mkdir(exist_ok=True)
        
        # 严格基于预见期偏移构造数据集
        frame, features = build_lead_dataset(raw, lead)
        frame['q90'] = frame['station_name'].map(thresholds)
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

    m1_metrics = evaluate_m1_only(predictions, args.wet_threshold)
    m1_metrics.to_csv(out / 'metrics_m1_raw.csv', index=False, encoding='utf-8-sig')

    print_vertical_reports(m1_metrics, out_dir=out)
    print(f"🎉 全部评估完成！4 张修正后的纵向指标表已输出至控制台并保存在: {out.resolve()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M1 地形差异子模型修正评价引擎')
    parser.add_argument('--csv', default=None)
    parser.add_argument('--out', default=None)
    parser.add_argument('--leads', type=int, nargs='+', default=[0, 1, 3, 5])
    parser.add_argument('--train-start', default='2012-01-01')
    parser.add_argument('--train-end', default='2019-12-31')
    parser.add_argument('--val-end', default='2021-12-31')
    parser.add_argument('--test-end', default='2024-12-31')
    parser.add_argument('--wet-threshold', type=float, default=0.1)
    parser.add_argument('--trials', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()

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
    print("🚀 M1 地形差异子模型独立评价计算启动 (预见期逻辑已严格对齐)")
    print(f"  • 输入数据: {args.csv}")
    print(f"  • 成果目录: {args.out}")
    print(f"  • 评测序列: Lead 0 (融合基线) + Lead 1、3、5天 (独立前瞻预报)")
    print("=" * 85)
    train_and_eval_m1(args)
