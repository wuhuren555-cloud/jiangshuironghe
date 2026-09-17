import os
import joblib
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 静默 Optuna 中间迭代日志
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 1. 读取数据并进行【训练集 / 验证集 / 测试集】时序划分
# =====================================================================
base_dir = r"C:\Users\26332\OneDrive\Desktop\sun mission\jiangshuironghe-MOE"
csv_path = os.path.join(base_dir, "shicezhandianshuju", "qingxiduiqi_2012_2024.csv")
df = pd.read_csv(csv_path)

# 识别并解析日期列
date_col = next((c for c in ['Date', 'date', 'time', 'datetime'] if c in df.columns), None)
if date_col is not None:
    df['Date'] = pd.to_datetime(df[date_col])
else:
    stations_count = df['station_name'].nunique()
    days_per_station = len(df) // stations_count
    date_range = pd.date_range("2012-01-01", periods=days_per_station, freq="D")
    df['Date'] = np.tile(date_range, stations_count)

# 按站点与日期严格升序排序
df = df.sort_values(by=['station_name', 'Date']).reset_index(drop=True)

feature_cols = ['gpm_rain', 'DEM', 'Slope', 'Aspect', 'lat', 'lon', 'Month']
df['is_wet'] = (df['station_rain'] >= 0.1).astype(int)
stations = df['station_name'].unique()

# 划分时间节点 (L=0 实时空间融合)
train_mask = (df['Date'] >= "2012-01-01") & (df['Date'] <= "2019-12-31")
val_mask   = (df['Date'] >= "2020-01-01") & (df['Date'] <= "2021-12-31")
test_mask  = (df['Date'] >= "2022-01-01") & (df['Date'] <= "2024-12-31")

train_df = df[train_mask].copy()
val_df   = df[val_mask].copy()
test_df  = df[test_mask].copy()

train_wet = train_df[train_df['station_rain'] >= 0.1].copy()
val_wet   = val_df[val_df['station_rain'] >= 0.1].copy()

# 定义各站点历史基准期专属 P90 极值门槛 (mm/d)
P90_STANDARD = {'连州': 17.02, '韶关': 18.29, '佛冈': 21.51, '连平': 17.27}
p90_map = {}
for st in stations:
    if st in P90_STANDARD:
        p90_map[st] = P90_STANDARD[st]
    else:
        p90_map[st] = float(train_df[train_df['station_name'] == st]['station_rain'].quantile(0.90))

print("=======================================================================================")
print("📊 数据集时序划分完成：")
print(f"  • 训练集 (Train Set): 2012-01-01 ~ 2019-12-31 | 样本数: {len(train_df):>6} 条")
print(f"  • 验证集 (Val Set)  : 2020-01-01 ~ 2021-12-31 | 样本数: {len(val_df):>6} 条 (用于超参调优&阈值搜索)")
print(f"  • 测试集 (Test Set) : 2022-01-01 ~ 2024-12-31 | 样本数: {len(test_df):>6} 条 (独立时序盲测)")
print(f"  • 各站点 P90 暴雨极值门槛: {p90_map}")
print("=======================================================================================")

# =====================================================================
# 2. 综合评估指标体系 (全量指标 + 四维正交暴雨极值指标)
# =====================================================================
def calc_all_metrics(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    
    valid = ~np.isnan(obs) & ~np.isnan(pred)
    o, p = obs[valid], pred[valid]
    
    if len(o) == 0:
        return {}
        
    # 1. 基础误差
    mse = mean_squared_error(o, p)
    mae = mean_absolute_error(o, p)
    rmse = np.sqrt(mse)
    
    # 2. 拟合优度与效率指标
    ss_tot = np.sum((o - np.mean(o))**2)
    ss_res = np.sum((o - p)**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    nse = r2
    
    # 3. KGE
    std_o, std_p = np.std(o), np.std(p)
    mean_o, mean_p = np.mean(o), np.mean(p)
    if std_o > 0 and std_p > 0 and mean_o != 0:
        r = np.corrcoef(o, p)[0, 1]
        alpha = std_p / std_o
        beta = mean_p / mean_o
        kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    else:
        r = 0.0
        kge = -999.0
        
    # 4. 干湿事件分类指标 (0.1 mm/d 门槛)
    H = np.sum((o >= 0.1) & (p >= 0.1))
    F = np.sum((o < 0.1) & (p >= 0.1))
    M = np.sum((o >= 0.1) & (p < 0.1))
    
    pod = H / (H + M) if (H + M) > 0 else 0.0
    far = F / (H + F) if (H + F) > 0 else 0.0
    csi = H / (H + F + M) if (H + F + M) > 0 else 0.0
    
    return {
        'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R²': r2, 'NSE': nse,
        'KGE': kge, 'CC': r, 'POD': pod, 'FAR': far, 'CSI': csi
    }

def calc_extreme_comprehensive_metrics(all_obs, all_pred, p90_threshold):
    """
    四维正交极端降水诊断体系：
    1. 容积水量: PHV (对标水文 FHV)
    2. 极值锐度: PE_peak (最大峰值偏差), Alpha_90 (变异比/方差保持度)
    3. 过程相位: CC_90 (极值相关系数), RMSE_90, MAE_90, MSE_90, R2_90
    4. 极端分类: POD_90, FAR_90, CSI_90, SEDI (对称极端依赖指数)
    """
    all_obs = np.asarray(all_obs, dtype=float)
    all_pred = np.asarray(all_pred, dtype=float)
    
    # 提取超阈值暴雨子集
    ext_mask = (all_obs >= p90_threshold)
    o_ext = all_obs[ext_mask]
    p_ext = all_pred[ext_mask]
    
    if len(o_ext) == 0:
        return {}
        
    # 1. 基础回归与拟合指标 (在 P90 极值子集上)
    mse_90 = mean_squared_error(o_ext, p_ext)
    mae_90 = mean_absolute_error(o_ext, p_ext)
    rmse_90 = np.sqrt(mse_90)
    
    ss_tot = np.sum((o_ext - np.mean(o_ext))**2)
    ss_res = np.sum((o_ext - p_ext)**2)
    r2_90 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    nse_90 = r2_90
    
    std_o = np.std(o_ext)
    std_p = np.std(p_ext)
    mean_o = np.mean(o_ext)
    mean_p = np.mean(p_ext)
    
    if std_o > 1e-5 and std_p > 1e-5:
        cc_90 = np.corrcoef(o_ext, p_ext)[0, 1]
        alpha = std_p / std_o
        beta = mean_p / mean_o if mean_o != 0 else 1.0
        kge_90 = 1.0 - np.sqrt((cc_90 - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    else:
        cc_90 = 0.0
        kge_90 = -999.0
        
    # 2. 暴雨专属水文与极值指标
    sum_o = np.sum(o_ext)
    phv = ((np.sum(p_ext) - sum_o) / sum_o * 100.0) if sum_o > 0 else 0.0
    
    max_o = np.max(o_ext)
    max_p = np.max(p_ext)
    pe_peak = ((max_p - max_o) / max_o * 100.0) if max_o > 0 else 0.0
    
    alpha_90 = (std_p / std_o) if std_o > 0 else 0.0
    
    # 3. 列联表与极端分类判别 (基于全序列以 P90 门槛划分事件)
    H = np.sum((all_obs >= p90_threshold) & (all_pred >= p90_threshold))
    F = np.sum((all_obs < p90_threshold) & (all_pred >= p90_threshold))
    M = np.sum((all_obs >= p90_threshold) & (all_pred < p90_threshold))
    CR = np.sum((all_obs < p90_threshold) & (all_pred < p90_threshold))
    
    pod_90 = H / (H + M) if (H + M) > 0 else 0.0
    far_90 = F / (H + F) if (H + F) > 0 else 0.0
    csi_90 = H / (H + F + M) if (H + F + M) > 0 else 0.0
    
    # 4. 对称极端依赖指数 SEDI
    hit_rate = np.clip(pod_90, 1e-5, 1.0 - 1e-5)
    pofd = F / (F + CR) if (F + CR) > 0 else 1e-5
    pofd = np.clip(pofd, 1e-5, 1.0 - 1e-5)
    
    num = np.log(pofd) - np.log(hit_rate) - np.log(1 - pofd) + np.log(1 - hit_rate)
    den = np.log(pofd) + np.log(hit_rate) + np.log(1 - pofd) + np.log(1 - hit_rate)
    sedi = num / den if den != 0 else 0.0
    
    return {
        '场次': len(o_ext), 'MSE': mse_90, 'MAE': mae_90, 'RMSE': rmse_90,
        'R²': r2_90, 'NSE': nse_90, 'KGE': kge_90, 'CC': cc_90,
        'POD': pod_90, 'FAR': far_90, 'CSI': csi_90,
        'PHV(%)': phv, 'PE_peak(%)': pe_peak, 'Alpha_90': alpha_90, 'SEDI': sedi
    }

# =====================================================================
# 3. 阶段一：干湿分类器 & 动态决策阈值寻优 (在 Train 训练，在 Val 验证)
# =====================================================================
print("\n🚀 阶段一：干湿分类器 (XGBClassifier) 超参及阈值在验证集寻优 (50轮)...")

def objective_cls(trial):
    threshold = trial.suggest_float('threshold', 0.20, 0.60)
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 80, 200, step=20),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 2.5),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(train_df[feature_cols], train_df['is_wet'])
    
    probs = model.predict_proba(val_df[feature_cols])[:, 1]
    pred = (probs >= threshold).astype(int)
    obs = val_df['is_wet'].values
    
    H = np.sum((obs == 1) & (pred == 1))
    F = np.sum((obs == 0) & (pred == 1))
    M = np.sum((obs == 1) & (pred == 0))
    csi = H / (H + F + M) if (H + F + M) > 0 else 0.0
    return csi

study_cls = optuna.create_study(direction="maximize")
study_cls.optimize(objective_cls, n_trials=50, show_progress_bar=True)

best_cls_params = study_cls.best_params
best_threshold = best_cls_params.pop('threshold')

print(f"✅ 分类器寻优完成！验证集最优 CSI: {study_cls.best_value:.4f}")
print(f"🎯 选定最佳降水概率决策阈值 (Best Threshold): {best_threshold:.4f}")

# =====================================================================
# 4. 阶段二：雨量回归器 (XGBRegressor) 超参数寻优
# =====================================================================
print("\n🚀 阶段二：雨量回归器 (XGBRegressor) 在验证集超参寻优 (50轮)...")

def objective_reg(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=30),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(train_wet[feature_cols], train_wet['station_rain'])
    
    pred = model.predict(val_wet[feature_cols])
    pred = np.clip(pred, 0, None)
    obs = val_wet['station_rain'].values
    
    rmse = np.sqrt(mean_squared_error(obs, pred))
    return rmse

study_reg = optuna.create_study(direction="minimize")
study_reg.optimize(objective_reg, n_trials=50, show_progress_bar=True)

best_reg_params = study_reg.best_params
print(f"✅ 回归器寻优完成！验证集最低 RMSE: {study_reg.best_value:.4f} mm/d")

# =====================================================================
# 5. 模型最终拟合与全量预测 (L=0 实时融合基准)
# =====================================================================
best_cls_params['random_state'] = 42
best_reg_params['random_state'] = 42

final_cls = xgb.XGBClassifier(**best_cls_params)
final_cls.fit(train_df[feature_cols], train_df['is_wet'])

final_reg = xgb.XGBRegressor(**best_reg_params)
final_reg.fit(train_wet[feature_cols], train_wet['station_rain'])

def predict_pipeline(data_split, cls_mod, reg_mod, thresh):
    probs = cls_mod.predict_proba(data_split[feature_cols])[:, 1]
    is_wet_pred = (probs >= thresh).astype(int)
    rain_pred = reg_mod.predict(data_split[feature_cols])
    final_pred = np.where(is_wet_pred == 1, rain_pred, 0.0)
    return np.clip(final_pred, 0, None)

train_df['pred_rain'] = predict_pipeline(train_df, final_cls, final_reg, best_threshold)
val_df['pred_rain']   = predict_pipeline(val_df, final_cls, final_reg, best_threshold)
test_df['pred_rain']  = predict_pipeline(test_df, final_cls, final_reg, best_threshold)

# =====================================================================
# 6. 输出：全量样本评估表 (表 1 & 表 2)
# =====================================================================
m_train = calc_all_metrics(train_df['station_rain'], train_df['pred_rain'])
m_val   = calc_all_metrics(val_df['station_rain'], val_df['pred_rain'])
m_test  = calc_all_metrics(test_df['station_rain'], test_df['pred_rain'])

print("\n" + "=" * 90)
print("🏆 表 1：降水融合子模型 1 在【训练集 / 验证集 / 独立测试集】全指标评估表 (全量样本)")
print("=" * 90)
print(f"{'评估指标 (Metrics)':<24} | {'训练集 (2012-2019)':<16} | {'验证集 (2020-2021)':<16} | {'测试集 (2022-2024)':<16}")
print("-" * 90)
for k in ['MSE', 'MAE', 'RMSE', 'R²', 'NSE', 'KGE', 'CC', 'POD', 'FAR', 'CSI']:
    print(f"{k:<24} | {m_train[k]:<16.3f} | {m_val[k]:<16.3f} | {m_test[k]:<16.3f}")
print("=" * 90)

print("\n" + "=" * 95)
print("📍 表 2：【独立测试集 (2022-2024 盲测)】4 个站点逐站多维指标明细 (全量样本)")
print("=" * 95)
print(f"{'站点':<6} | {'MSE':<7} | {'MAE':<6} | {'RMSE':<6} | {'R²':<6} | {'NSE':<6} | {'KGE':<6} | {'POD':<5} | {'FAR':<5} | {'CSI':<5}")
print("-" * 95)
for site in stations:
    sub = test_df[test_df['station_name'] == site]
    m_s = calc_all_metrics(sub['station_rain'], sub['pred_rain'])
    print(f"{site:<6} | {m_s['MSE']:<7.2f} | {m_s['MAE']:<6.2f} | {m_s['RMSE']:<6.2f} | {m_s['R²']:<6.3f} | {m_s['NSE']:<6.3f} | {m_s['KGE']:<6.3f} | {m_s['POD']:<5.2f} | {m_s['FAR']:<5.2f} | {m_s['CSI']:<5.2f}")
print("=" * 95)

# =====================================================================
# 7. 输出：P90 暴雨极值全套指标评估表 (表 3 & 表 4)
# =====================================================================
# 针对多站计算加权/全局 P90 评估
def calc_dataset_extreme_metrics(df_split):
    records = []
    for site in stations:
        sub = df_split[df_split['station_name'] == site]
        p90_val = p90_map[site]
        m = calc_extreme_comprehensive_metrics(sub['station_rain'], sub['pred_rain'], p90_val)
        records.append(m)
    df_rec = pd.DataFrame(records)
    # 计算均值汇总
    mean_series = df_rec.mean(numeric_only=True)
    mean_series['场次'] = int(df_rec['场次'].sum())
    return mean_series

ext_train = calc_dataset_extreme_metrics(train_df)
ext_val   = calc_dataset_extreme_metrics(val_df)
ext_test  = calc_dataset_extreme_metrics(test_df)

print("\n" + "=" * 95)
print("⛈️ 表 3：降水融合子模型 1 在【P90 暴雨极值样本】全套指标评估表 (四维正交极值体系)")
print("=" * 95)
print(f"{'极值评估指标 (Metrics)':<24} | {'训练集 (≥P90)':<16} | {'验证集 (≥P90)':<16} | {'测试集 (≥P90)':<16}")
print("-" * 95)
for k in ['MSE', 'MAE', 'RMSE', 'R²', 'NSE', 'KGE', 'CC', 'POD', 'FAR', 'CSI']:
    print(f"{k:<24} | {ext_train[k]:<16.3f} | {ext_val[k]:<16.3f} | {ext_test[k]:<16.3f}")
print(f"{'PHV (暴雨高水容积偏差%)':<24} | {ext_train['PHV(%)']:<+16.2f} | {ext_val['PHV(%)']:<+16.2f} | {ext_test['PHV(%)']:<+16.2f}")
print(f"{'PE_peak (极值峰值偏差%)':<24} | {ext_train['PE_peak(%)']:<+16.2f} | {ext_val['PE_peak(%)']:<+16.2f} | {ext_test['PE_peak(%)']:<+16.2f}")
print(f"{'Alpha_90 (变异比/保真度)':<24} | {ext_train['Alpha_90']:<16.3f} | {ext_val['Alpha_90']:<16.3f} | {ext_test['Alpha_90']:<16.3f}")
print(f"{'SEDI (对称极端依赖指数)':<24} | {ext_train['SEDI']:<16.3f} | {ext_val['SEDI']:<16.3f} | {ext_test['SEDI']:<16.3f}")
print("=" * 95)

print("\n" + "=" * 125)
print("🎯 表 4：【独立测试集 (2022-2024 盲测)】4 站点在 P90 暴雨极值下的逐站指标明细")
print("=" * 125)
print(f"{'站点':<5} | {'门槛':<6} | {'场次':<4} | {'RMSE':<6} | {'R²':<6} | {'CC':<5} | {'CSI':<5} | {'PHV(%)':<8} | {'PE_peak(%)':<10} | {'Alpha_90':<8} | {'SEDI':<6}")
print("-" * 125)
for site in stations:
    sub = test_df[test_df['station_name'] == site]
    p90_val = p90_map[site]
    m_s = calc_extreme_comprehensive_metrics(sub['station_rain'], sub['pred_rain'], p90_val)
    print(f"{site:<6} | {p90_val:<6.2f} | {m_s['场次']:<4} | {m_s['RMSE']:<6.2f} | {m_s['R²']:<6.3f} | {m_s['CC']:<5.3f} | {m_s['CSI']:<5.2f} | {m_s['PHV(%)']:<+8.1f} | {m_s['PE_peak(%)']:<+10.1f} | {m_s['Alpha_90']:<8.3f} | {m_s['SEDI']:<6.3f}")
print("=" * 125)

# =====================================================================
# 8. 多预见期 (Lead Time = 1, 2, 3 天) 时空外推实验与评估 (表 5 & 表 6)
# =====================================================================
def run_lead_time_experiments(df_in, lead_days=[1, 2, 3]):
    lead_summary = []
    lead_station_details = []
    
    for L in lead_days:
        df_lead = df_in.copy()
        # 严格按站点进行目标值时空错位平移 (以 t 日特征预测 t+L 日降水)
        df_lead['target_rain'] = df_lead.groupby('station_name')['station_rain'].shift(-L)
        df_lead['target_date'] = df_lead.groupby('station_name')['Date'].shift(-L)
        df_lead['Month'] = df_lead['target_date'].dt.month
        df_lead['is_wet'] = (df_lead['target_rain'] >= 0.1).astype(int)
        df_lead = df_lead.dropna(subset=['target_rain', 'target_date']).reset_index(drop=True)
        
        # 划分各预见期下的时序区间
        tr_L = df_lead[(df_lead['target_date'] >= "2012-01-01") & (df_lead['target_date'] <= "2019-12-31")].copy()
        te_L = df_lead[(df_lead['target_date'] >= "2022-01-01") & (df_lead['target_date'] <= "2024-12-31")].copy()
        tr_wet_L = tr_L[tr_L['target_rain'] >= 0.1].copy()
        
        # 拟合多预见期模型 (复用验证集最优超参架构)
        m_cls_L = xgb.XGBClassifier(**best_cls_params)
        m_cls_L.fit(tr_L[feature_cols], tr_L['is_wet'])
        
        m_reg_L = xgb.XGBRegressor(**best_reg_params)
        m_reg_L.fit(tr_wet_L[feature_cols], tr_wet_L['target_rain'])
        
        # 预测独立测试集
        te_L['pred_rain'] = predict_pipeline(te_L, m_cls_L, m_reg_L, best_threshold)
        
        # 1. 全量样本多维指标
        m_all_L = calc_all_metrics(te_L['target_rain'], te_L['pred_rain'])
        
        # 2. P90 暴雨极值专属指标汇总
        records_ext = []
        for site in stations:
            sub_s = te_L[te_L['station_name'] == site]
            p90_val = p90_map[site]
            m_s = calc_extreme_comprehensive_metrics(sub_s['target_rain'], sub_s['pred_rain'], p90_val)
            records_ext.append(m_s)
            
            lead_station_details.append({
                'Lead_Time': f"+{L}天",
                '站点': site,
                '场次': m_s['场次'],
                'RMSE_90': m_s['RMSE'],
                'CC_90': m_s['CC'],
                'CSI_90': m_s['CSI'],
                'PHV(%)': m_s['PHV(%)'],
                'PE_peak(%)': m_s['PE_peak(%)'],
                'Alpha_90': m_s['Alpha_90'],
                'SEDI': m_s['SEDI']
            })
            
        df_rec_ext = pd.DataFrame(records_ext).mean(numeric_only=True)
        
        lead_summary.append({
            'Lead_Time': f"+{L}天",
            'All_RMSE': m_all_L['RMSE'],
            'All_R²': m_all_L['R²'],
            'All_KGE': m_all_L['KGE'],
            'All_CSI': m_all_L['CSI'],
            'P90_RMSE': df_rec_ext['RMSE'],
            'P90_CC': df_rec_ext['CC'],
            'P90_CSI': df_rec_ext['CSI'],
            'P90_PHV': df_rec_ext['PHV(%)'],
            'P90_PE_peak': df_rec_ext['PE_peak(%)'],
            'P90_Alpha': df_rec_ext['Alpha_90'],
            'P90_SEDI': df_rec_ext['SEDI']
        })
        
    return pd.DataFrame(lead_summary), pd.DataFrame(lead_station_details)

print("\n🚀 正在执行多预见期 (Lead Time = 1, 2, 3 天) 时空外推演算与衰减分析...")
df_lead_sum, df_lead_sites = run_lead_time_experiments(df, lead_days=[1, 2, 3])

print("\n" + "=" * 125)
print("📡 表 5：【多预见期 (1~3天)】测试集 (2022-2024) 全量与 P90 极值演化总表")
print("=" * 125)
print(f"{'预见期':<6} | {'全量RMSE':<8} | {'全量R²':<7} | {'全量KGE':<7} | {'极值RMSE':<8} | {'极值CC':<6} | {'极值CSI':<7} | {'PHV(%)':<8} | {'PE_peak(%)':<10} | {'Alpha_90':<8} | {'SEDI':<6}")
print("-" * 125)
for _, r in df_lead_sum.iterrows():
    print(f"{r['Lead_Time']:<8} | {r['All_RMSE']:<10.2f} | {r['All_R²']:<9.3f} | {r['All_KGE']:<9.3f} | {r['P90_RMSE']:<10.2f} | {r['P90_CC']:<8.3f} | {r['P90_CSI']:<9.2f} | {r['P90_PHV']:<+8.1f} | {r['P90_PE_peak']:<+10.1f} | {r['P90_Alpha']:<8.3f} | {r['P90_SEDI']:<6.3f}")
print("=" * 125)

print("\n" + "=" * 120)
print("📍 表 6：【多预见期 (1~3天)】4 站点逐站 P90 暴雨极值指标衰减明细表")
print("=" * 120)
print(f"{'预见期':<6} | {'站点':<5} | {'场次':<4} | {'RMSE_90':<8} | {'CC_90':<6} | {'CSI_90':<6} | {'PHV(%)':<8} | {'PE_peak(%)':<10} | {'Alpha_90':<8} | {'SEDI':<6}")
print("-" * 120)
for _, r in df_lead_sites.iterrows():
    print(f"{r['Lead_Time']:<8} | {r['站点']:<6} | {r['场次']:<4} | {r['RMSE_90']:<8.2f} | {r['CC_90']:<6.3f} | {r['CSI_90']:<6.2f} | {r['PHV(%)']:<+8.1f} | {r['PE_peak(%)']:<+10.1f} | {r['Alpha_90']:<8.3f} | {r['SEDI']:<6.3f}")
print("=" * 120)

# =====================================================================
# 9. 模型资产持久化导出
# =====================================================================
model_dir = os.path.join(base_dir, "wanggeshuju")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(final_cls, os.path.join(model_dir, "dixingchayi-jiangshuiyuzhiyouhua_cls_model.pkl"))
joblib.dump(final_reg, os.path.join(model_dir, "dixingchayi-jiangshuiyuzhiyouhua_reg_model.pkl"))

config_info = {
    'best_threshold': best_threshold,
    'feature_cols': feature_cols,
    'p90_thresholds': p90_map
}
joblib.dump(config_info, os.path.join(model_dir, "model_config.pkl"))

print(f"\n🎉 升级版全套评估、极值检验与多预见期演算全部完成！资产已导出至: {model_dir}")
