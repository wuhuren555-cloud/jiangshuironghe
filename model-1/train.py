import os
import joblib
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 静默 Optuna 中间过程输出
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

feature_cols = ['gpm_rain', 'DEM', 'Slope', 'Aspect', 'lat', 'lon', 'Month']
df['is_wet'] = (df['station_rain'] >= 0.1).astype(int)
stations = df['station_name'].unique()

# 划分时间节点
train_mask = (df['Date'] >= "2012-01-01") & (df['Date'] <= "2019-12-31")
val_mask   = (df['Date'] >= "2020-01-01") & (df['Date'] <= "2021-12-31")
test_mask  = (df['Date'] >= "2022-01-01") & (df['Date'] <= "2024-12-31")

train_df = df[train_mask].copy()
val_df   = df[val_mask].copy()
test_df  = df[test_mask].copy()

train_wet = train_df[train_df['station_rain'] >= 0.1].copy()
val_wet   = val_df[val_df['station_rain'] >= 0.1].copy()

print("=======================================================================================")
print("📊 数据集时序划分完成：")
print(f"  • 训练集 (Train Set): 2012-01-01 ~ 2019-12-31 | 样本数: {len(train_df):>6} 条")
print(f"  • 验证集 (Val Set)  : 2020-01-01 ~ 2021-12-31 | 样本数: {len(val_df):>6} 条 (用于调参&阈值寻优)")
print(f"  • 测试集 (Test Set) : 2022-01-01 ~ 2024-12-31 | 样本数: {len(test_df):>6} 条 (独立盲测验证)")
print("=======================================================================================")

# =====================================================================
# 2. 多维综合评估指标函数 (MSE, MAE, R², NSE, KGE, POD, FAR, CSI 等)
# =====================================================================
def calc_all_metrics(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    
    valid = ~np.isnan(obs) & ~np.isnan(pred)
    o, p = obs[valid], pred[valid]
    
    if len(o) == 0:
        return {}
        
    # 1. 基础绝对误差
    mse = mean_squared_error(o, p)
    mae = mean_absolute_error(o, p)
    rmse = np.sqrt(mse)
    
    # 2. 拟合与效率指标
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
        
    # 4. 干湿事件探测分类指标 (POD, FAR, CSI)
    H = np.sum((o >= 0.1) & (p >= 0.1))  # 命中 (Hits)
    F = np.sum((o < 0.1) & (p >= 0.1))   # 空报 (False alarms)
    M = np.sum((o >= 0.1) & (p < 0.1))   # 漏报 (Misses)
    
    pod = H / (H + M) if (H + M) > 0 else 0.0
    far = F / (H + F) if (H + F) > 0 else 0.0
    csi = H / (H + F + M) if (H + F + M) > 0 else 0.0
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'NSE': nse,
        'KGE': kge,
        'CC': r,
        'POD': pod,
        'FAR': far,
        'CSI': csi
    }

# =====================================================================
# 3. 阶段一：干湿分类器 & 动态概率阈值寻优 (在 Train 训练，在 Val 验证)
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
# 5. 模型拟合与三数据集预测
# =====================================================================
best_cls_params['random_state'] = 42
best_reg_params['random_state'] = 42

final_cls = xgb.XGBClassifier(**best_cls_params)
final_cls.fit(train_df[feature_cols], train_df['is_wet'])

final_reg = xgb.XGBRegressor(**best_reg_params)
final_reg.fit(train_wet[feature_cols], train_wet['station_rain'])

def predict_pipeline(data_split):
    probs = final_cls.predict_proba(data_split[feature_cols])[:, 1]
    is_wet_pred = (probs >= best_threshold).astype(int)
    rain_pred = final_reg.predict(data_split[feature_cols])
    final_pred = np.where(is_wet_pred == 1, rain_pred, 0.0)
    return np.clip(final_pred, 0, None)

train_df['pred_rain'] = predict_pipeline(train_df)
val_df['pred_rain']   = predict_pipeline(val_df)
test_df['pred_rain']  = predict_pipeline(test_df)

# =====================================================================
# 6. 🌟 输出全数据集多维评估指标总表 (已补全 POD 与 FAR)
# =====================================================================
m_train = calc_all_metrics(train_df['station_rain'], train_df['pred_rain'])
m_val   = calc_all_metrics(val_df['station_rain'], val_df['pred_rain'])
m_test  = calc_all_metrics(test_df['station_rain'], test_df['pred_rain'])

print("\n" + "=" * 85)
print("🏆 表 1：降水融合子模型 1 在【训练集 / 验证集 / 独立测试集】全指标评估表")
print("=" * 85)
print(f"{'评估指标 (Metrics)':<24} | {'训练集 (2012-2019)':<16} | {'验证集 (2020-2021)':<16} | {'测试集 (2022-2024)':<16}")
print("-" * 85)
print(f"{'MSE (均方误差)':<24} | {m_train['MSE']:<16.3f} | {m_val['MSE']:<16.3f} | {m_test['MSE']:<16.3f}")
print(f"{'MAE (平均绝对误差)':<24} | {m_train['MAE']:<16.3f} | {m_val['MAE']:<16.3f} | {m_test['MAE']:<16.3f}")
print(f"{'RMSE (均方根误差)':<24} | {m_train['RMSE']:<16.3f} | {m_val['RMSE']:<16.3f} | {m_test['RMSE']:<16.3f}")
print(f"{'R² (决定系数)':<24} | {m_train['R²']:<16.3f} | {m_val['R²']:<16.3f} | {m_test['R²']:<16.3f}")
print(f"{'NSE (纳什效率系数)':<24} | {m_train['NSE']:<16.3f} | {m_val['NSE']:<16.3f} | {m_test['NSE']:<16.3f}")
print(f"{'KGE (Kling-Gupta)':<24} | {m_train['KGE']:<16.3f} | {m_val['KGE']:<16.3f} | {m_test['KGE']:<16.3f}")
print(f"{'CC (相关系数)':<24} | {m_train['CC']:<16.3f} | {m_val['CC']:<16.3f} | {m_test['CC']:<16.3f}")
print(f"{'POD (命中率/探测率)':<24} | {m_train['POD']:<16.3f} | {m_val['POD']:<16.3f} | {m_test['POD']:<16.3f}")
print(f"{'FAR (空报率/虚警率)':<24} | {m_train['FAR']:<16.3f} | {m_val['FAR']:<16.3f} | {m_test['FAR']:<16.3f}")
print(f"{'CSI (临界成功指数)':<24} | {m_train['CSI']:<16.3f} | {m_val['CSI']:<16.3f} | {m_test['CSI']:<16.3f}")
print("=" * 85)

# =====================================================================
# 7. 🌟 输出【独立测试集 (2022-2024 盲测)】4 站点逐站多维明细 (整齐紧凑排版)
# =====================================================================
print("\n" + "=" * 90)
print("📍 表 2：【独立测试集 (2022-2024 盲测)】4 个站点逐站多维指标明细")
print("=" * 90)
print(f"{'站点':<6} | {'MSE':<7} | {'MAE':<6} | {'RMSE':<6} | {'R²':<6} | {'NSE':<6} | {'KGE':<6} | {'POD':<5} | {'FAR':<5} | {'CSI':<5}")
print("-" * 90)

for site in stations:
    sub = test_df[test_df['station_name'] == site]
    m_s = calc_all_metrics(sub['station_rain'], sub['pred_rain'])
    print(f"{site:<6} | {m_s['MSE']:<7.2f} | {m_s['MAE']:<6.2f} | {m_s['RMSE']:<6.2f} | {m_s['R²']:<6.3f} | {m_s['NSE']:<6.3f} | {m_s['KGE']:<6.3f} | {m_s['POD']:<5.2f} | {m_s['FAR']:<5.2f} | {m_s['CSI']:<5.2f}")
print("=" * 90)

# =====================================================================
# 8. 模型资产持久化导出
# =====================================================================
model_dir = os.path.join(base_dir, "wanggeshuju")
joblib.dump(final_cls, os.path.join(model_dir, "dixingchayi-jiangshuiyuzhiyouhua_cls_model.pkl"))
joblib.dump(final_reg, os.path.join(model_dir, "dixingchayi-jiangshuiyuzhiyouhua_reg_model.pkl"))

config_info = {
    'best_threshold': best_threshold,
    'feature_cols': feature_cols
}
joblib.dump(config_info, os.path.join(model_dir, "model_config.pkl"))

print(f"\n🎉 训练、三集划分验证及评估全部完成！模型与元信息已导出至: {model_dir}")
