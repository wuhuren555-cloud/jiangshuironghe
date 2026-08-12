import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import joblib

# 静默 Optuna 中间日志
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. 读取第一步生成的对齐数据
base_dir = r"C:\Users\26332\OneDrive\Desktop\sun mission\jiangshuironghe-MOE"
csv_path = os.path.join(base_dir, "shicezhandianshuju", "qingxiduiqi_2012_2024.csv")
df = pd.read_csv(csv_path)

feature_cols = ['gpm_rain', 'DEM', 'Slope', 'Aspect', 'lat', 'lon', 'Month']
df['is_wet'] = (df['station_rain'] >= 0.1).astype(int)
stations = df['station_name'].unique()

print("==================================================")
print("🚀 阶段一：干湿分类器 (XGBClassifier) 超参数寻优中 (50轮)...")
print("==================================================")

def objective_cls(trial):
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
    
    csi_list = []
    for test_site in stations:
        train_df = df[df['station_name'] != test_site]
        test_df = df[df['station_name'] == test_site]
        
        model = xgb.XGBClassifier(**params)
        model.fit(train_df[feature_cols], train_df['is_wet'])
        
        pred = model.predict(test_df[feature_cols])
        obs = test_df['is_wet'].values
        
        H = np.sum((obs == 1) & (pred == 1))
        F = np.sum((obs == 0) & (pred == 1))
        M = np.sum((obs == 1) & (pred == 0))
        
        csi = H / (H + F + M) if (H + F + M) > 0 else 0
        csi_list.append(csi)
        
    return np.mean(csi_list)

study_cls = optuna.create_study(direction="maximize")
study_cls.optimize(objective_cls, n_trials=50, show_progress_bar=True)

best_cls_params = study_cls.best_params
print(f"✅ 分类器寻找完成！搜寻最高平均 CSI: {study_cls.best_value:.4f}")

print("\n==================================================")
print("🚀 阶段二：雨量回归器 (XGBRegressor) 超参数寻优中 (50轮)...")
print("==================================================")

df_wet = df[df['station_rain'] >= 0.1].copy()

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
    
    rmse_list = []
    for test_site in stations:
        train_df = df_wet[df_wet['station_name'] != test_site]
        test_df = df_wet[df_wet['station_name'] == test_site]
        
        model = xgb.XGBRegressor(**params)
        model.fit(train_df[feature_cols], train_df['station_rain'])
        
        pred = model.predict(test_df[feature_cols])
        pred = np.clip(pred, 0, None)
        obs = test_df['station_rain'].values
        
        rmse = np.sqrt(mean_squared_error(obs, pred))
        rmse_list.append(rmse)
        
    return np.mean(rmse_list)

study_reg = optuna.create_study(direction="minimize")
study_reg.optimize(objective_reg, n_trials=50, show_progress_bar=True)

best_reg_params = study_reg.best_params
print(f"✅ 回归器寻找完成！搜寻最低平均 RMSE: {study_reg.best_value:.4f} mm/d")

# =================================================="
# 🌟🌟 关键补全：使用调优后的黄金参数重新跑 LOSO-CV 打印各站点详细指标 🌟🌟
# =================================================="
print("\n==================================================")
print("📊 调参后【双核组合模型】盲测站点详细指标复盘 (LOSO-CV)")
print("==================================================")

best_cls_params['random_state'] = 42
best_reg_params['random_state'] = 42

eval_summary = []

for test_site in stations:
    train_df = df[df['station_name'] != test_site]
    test_df = df[df['station_name'] == test_site].copy()
    
    # 训练分类器
    m_cls = xgb.XGBClassifier(**best_cls_params)
    m_cls.fit(train_df[feature_cols], train_df['is_wet'])
    
    # 训练回归器（仅针对有雨样本）
    train_wet = train_df[train_df['station_rain'] >= 0.1]
    m_reg = xgb.XGBRegressor(**best_reg_params)
    m_reg.fit(train_wet[feature_cols], train_wet['station_rain'])
    
    # 组合预测
    pred_is_wet = m_cls.predict(test_df[feature_cols])
    pred_rain_raw = m_reg.predict(test_df[feature_cols])
    final_pred = np.where(pred_is_wet == 1, pred_rain_raw, 0.0)
    final_pred = np.clip(final_pred, 0, None)
    
    test_df['pred_rain'] = final_pred
    
    # 评估计算
    obs = test_df['station_rain'].values
    pred = test_df['pred_rain'].values
    
    rmse = np.sqrt(mean_squared_error(obs, pred))
    r2 = r2_score(obs, pred)
    cc = np.corrcoef(obs, pred)[0, 1]
    
    H = np.sum((obs >= 0.1) & (pred >= 0.1))
    F = np.sum((obs < 0.1) & (pred >= 0.1))
    M = np.sum((obs >= 0.1) & (pred < 0.1))
    
    pod = H / (H + M) if (H + M) > 0 else 0
    far = F / (H + F) if (H + F) > 0 else 0
    csi = H / (H + F + M) if (H + F + M) > 0 else 0
    
    eval_summary.append({
        '站点': test_site,
        'RMSE(mm/d)': rmse,
        'R²': r2,
        'CC': cc,
        'POD': pod,
        'FAR': far,
        'CSI': csi
    })
    
    print(f"📍 盲测站点【{test_site:^4}】| RMSE: {rmse:6.2f} mm/d | R²: {r2:5.3f} | CC: {cc:5.3f} | POD: {pod:4.2f} | FAR: {far:4.2f} | CSI: {csi:4.2f}")

# 输出 4 站平均结果
df_summary = pd.DataFrame(eval_summary)
mean_rmse = df_summary['RMSE(mm/d)'].mean()
mean_r2 = df_summary['R²'].mean()
mean_cc = df_summary['CC'].mean()
mean_pod = df_summary['POD'].mean()
mean_far = df_summary['FAR'].mean()
mean_csi = df_summary['CSI'].mean()

print("-" * 75)
print(f"🏆 【4 站全局平均】 | RMSE: {mean_rmse:6.2f} mm/d | R²: {mean_r2:5.3f} | CC: {mean_cc:5.3f} | POD: {mean_pod:4.2f} | FAR: {mean_far:4.2f} | CSI: {mean_csi:4.2f}")
print("=" * 75)

# =================================================="
# 保存调优后的最终模型
# =================================================="
final_cls_model = xgb.XGBClassifier(**best_cls_params)
final_cls_model.fit(df[feature_cols], df['is_wet'])

final_reg_model = xgb.XGBRegressor(**best_reg_params)
final_reg_model.fit(df_wet[feature_cols], df_wet['station_rain'])

model_dir = os.path.join(base_dir, "wanggeshuju")
joblib.dump(final_cls_model, os.path.join(model_dir, "dixingchayi-tiaocan_cls_model.pkl"))
joblib.dump(final_reg_model, os.path.join(model_dir, "dixingchayi-tiaocan_reg_model.pkl"))

print(f"\n🎉 调参和全套评估完成！调优后的模型权重已导出至: {model_dir}")
