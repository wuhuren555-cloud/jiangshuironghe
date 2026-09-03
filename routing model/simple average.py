import os
import random
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import torch

# =========================================================================
# 🔒 1. 锁死全局随机种子
# =========================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(42)

DRIZZLE_THRESHOLD = 0.15

# =========================================================================
# 📌 2. 路径与设备配置
# =========================================================================
base_dir = r"C:\Users\26332\OneDrive\Desktop\sun mission\jiangshuironghe-MOE"

router_dir          = os.path.join(base_dir, "路由模型")
expert_preds_pt     = os.path.join(router_dir, "expert_predictions_2012_2024.pt")

nc_gpm_path         = os.path.join(base_dir, "wanggeshuju", "juhe_gpm_daily_2012_2024.nc")
dem_tif             = os.path.join(base_dir, "边界文件两幅", "过程文件", "touying-caijian.tif")
station_csv_path    = os.path.join(base_dir, "shicezhandianshuju", "qingxiduiqi_2012_2024.csv")

txt_result_path     = os.path.join(router_dir, "结果_简单平均.txt")
nc_simple_out       = os.path.join(base_dir, "wanggeshuju", "quanliuyu_simple_average_2012_2024.nc")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ 计算设备: {DEVICE}")

# =========================================================================
# 📊 3. 全套学术评价指标计算函数
# =========================================================================
def calc_all_metrics(obs, pred, threshold=0.1):
    valid_mask = ~np.isnan(obs) & ~np.isnan(pred)
    o = np.array(obs)[valid_mask]
    p = np.array(pred)[valid_mask]
    if len(o) == 0:
        return {k: 0.0 for k in ['MAE', 'MSE', 'RMSE', 'R2', 'NSE', 'KGE', 'CC', 'POD', 'FAR', 'CSI']}

    mae = np.mean(np.abs(p - o))
    mse = np.mean((p - o) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    nse = 1.0 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 1e-6 else 0.0
    r2 = nse

    std_o, std_p = np.std(o), np.std(p)
    mean_o, mean_p = np.mean(o), np.mean(p)
    cc = np.corrcoef(o, p)[0, 1] if (std_o > 1e-6 and std_p > 1e-6) else 0.0

    # Kling-Gupta Efficiency (KGE)
    alpha = std_p / (std_o + 1e-8)
    beta = mean_p / (mean_o + 1e-8)
    kge = 1.0 - np.sqrt((cc - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

    # 降水事件分类统计 (阈值默认 0.1 mm/d)
    hits = np.sum((o >= threshold) & (p >= threshold))
    misses = np.sum((o >= threshold) & (p < threshold))
    false_alarms = np.sum((o < threshold) & (p >= threshold))

    pod = hits / (hits + misses + 1e-8)
    far = false_alarms / (hits + false_alarms + 1e-8)
    csi = hits / (hits + misses + false_alarms + 1e-8)

    return {
        'MAE': mae, 'MSE': mse, 'RMSE': rmse,
        'R2': r2, 'NSE': nse, 'KGE': kge, 'CC': cc,
        'POD': pod, 'FAR': far, 'CSI': csi
    }

# =========================================================================
# 🚀 4. 主程序：简单平均融合推演与分时段指标评估
# =========================================================================
def main():
    print("=" * 88)
    print("📌 正在执行【三专家模型简单平均（Simple Average Baseline）】精度评估...")

    # 1. 读取 3 个专家模型的预测张量 (N, 3, H, W)
    expert_preds = torch.load(expert_preds_pt)
    num_days, num_experts, H, W = expert_preds.shape
    print(f"📦 专家预测数据加载成功: 共 {num_days} 天, 专家数量 = {num_experts}")

    # 2. 读取地理坐标与流域掩膜
    ds_gpm = xr.open_dataset(nc_gpm_path, engine="netcdf4").sel(time=slice("2012-01-01", "2024-12-31"))
    times, lats, lons = ds_gpm.time.values, ds_gpm.lat.values, ds_gpm.lon.values
    gpm_dates_str = pd.to_datetime(times).strftime('%Y-%m-%d')

    match_template = ds_gpm['precipitation'].rename({"lon": "x", "lat": "y"}).rio.write_crs("EPSG:4326")
    da_dem = rioxarray.open_rasterio(dem_tif).rio.reproject_match(match_template)
    raw_dem_arr = da_dem.values[0]
    basin_mask_2d = ~np.isnan(raw_dem_arr) & (raw_dem_arr > -100)

    # 3. 读取站点实测真值
    df_station = pd.read_csv(station_csv_path)
    df_station['date'] = pd.to_datetime(df_station['date'])
    date_to_idx = {pd.Timestamp(d).strftime('%Y-%m-%d'): i for i, d in enumerate(times)}

    station_target = np.zeros((num_days, H, W), dtype=np.float32)
    station_coords = {}

    for _, row in df_station[['station_name', 'lat', 'lon']].drop_duplicates().iterrows():
        st_name = row['station_name']
        r_idx = int(np.abs(lats - row['lat']).argmin())
        c_idx = int(np.abs(lons - row['lon']).argmin())
        station_coords[st_name] = (r_idx, c_idx)

        st_data = df_station[df_station['station_name'] == st_name]
        for _, d_row in st_data.iterrows():
            d_str = d_row['date'].strftime('%Y-%m-%d')
            if d_str in date_to_idx:
                station_target[date_to_idx[d_str], r_idx, c_idx] = float(d_row['station_rain'])

    # 4. 执行等权重简单平均计算 (w1 = w2 = w3 = 1/3)
    # 并保持微雨门控一致性 (降水 < 0.15 mm/d 时归零)
    p_simple_tensor = torch.mean(expert_preds, dim=1) # (N, H, W)
    p_base_max = torch.max(expert_preds[:, 0, :, :], expert_preds[:, 1, :, :])
    wet_mask = (p_base_max >= DRIZZLE_THRESHOLD).float()
    p_simple_arr = (p_simple_tensor * wet_mask).numpy()

    # 5. 时段对齐划分 (训练期: 2012–2019, 验证期: 2020–2021, 测试期: 2022–2024)
    train_mask_idx = (gpm_dates_str >= '2012-01-01') & (gpm_dates_str <= '2019-12-31')
    val_mask_idx   = (gpm_dates_str >= '2020-01-01') & (gpm_dates_str <= '2021-12-31')
    test_mask_idx  = (gpm_dates_str >= '2022-01-01') & (gpm_dates_str <= '2024-12-31')

    train_indices = np.where(train_mask_idx)[0]
    val_indices   = np.where(val_mask_idx)[0]
    test_indices  = np.where(test_mask_idx)[0]

    def extract_pairs(split_indices):
        obs_all, pred_all = [], []
        for st_name, (r_idx, c_idx) in station_coords.items():
            obs_st = station_target[split_indices, r_idx, c_idx]
            pred_st = p_simple_arr[split_indices, r_idx, c_idx]
            obs_all.append(obs_st)
            pred_all.append(pred_st)
        return np.concatenate(obs_all), np.concatenate(pred_all)

    train_obs, train_pred = extract_pairs(train_indices)
    val_obs,   val_pred   = extract_pairs(val_indices)
    test_obs,  test_pred  = extract_pairs(test_indices)

    metrics_train = calc_all_metrics(train_obs, train_pred, threshold=0.1)
    metrics_val   = calc_all_metrics(val_obs,   val_pred,   threshold=0.1)
    metrics_test  = calc_all_metrics(test_obs,  test_pred,  threshold=0.1)

    # -------------------------------------------------------------------------
    # 📑 表 1：三期对比全景表 (简单平均 Baseline)
    # -------------------------------------------------------------------------
    metric_names = ['MAE', 'MSE', 'RMSE', 'R2', 'NSE', 'KGE', 'CC', 'POD', 'FAR', 'CSI']
    metric_units = {
        'MAE': 'mm/d', 'MSE': 'mm²/d²', 'RMSE': 'mm/d',
        'R2': '—', 'NSE': '—', 'KGE': '—', 'CC': '—',
        'POD': '—', 'FAR': '—', 'CSI': '—'
    }
    metric_display_names = {
        'MAE': 'MAE', 'MSE': 'MSE', 'RMSE': 'RMSE',
        'R2': 'R²', 'NSE': 'NSE', 'KGE': 'KGE', 'CC': 'CC',
        'POD': 'POD', 'FAR': 'FAR', 'CSI': 'CSI'
    }

    table1_lines = [
        "\n" + "=" * 88,
        "评估指标       | 单位         | 训练期时段 (2012–2019)| 验证期时段 (2020–2021)| 测试期时段 (2022–2024)",
        "-" * 88
    ]
    for m in metric_names:
        disp_m = metric_display_names[m]
        unit   = metric_units[m]
        v_tr   = metrics_train[m]
        v_va   = metrics_val[m]
        v_te   = metrics_test[m]
        table1_lines.append(f"{disp_m:<14} | {unit:<12} | {v_tr:<22.3f} | {v_va:<22.3f} | {v_te:<22.3f}")
    table1_lines.append("=" * 88)
    table1_text = "\n".join(table1_lines)

    # -------------------------------------------------------------------------
    # 📑 表 2：站点详情 (测试期: 2022–2024)
    # -------------------------------------------------------------------------
    table2_lines = [
        "\n" + "=" * 108,
        "📍【站点详情 (测试期: 2022–2024 - 简单平均 Baseline)】",
        "=" * 108,
        f"{'站点':<8} | {'MSE(mm²/d²)':<12} | {'MAE(mm/d)':<10} | {'RMSE(mm/d)':<11} | {'R²':<8} | {'NSE':<8} | {'KGE':<8} | {'POD':<7} | {'FAR':<7} | {'CSI':<7}",
        "-" * 108
    ]
    for st_name, (r_idx, c_idx) in station_coords.items():
        obs_st  = station_target[test_indices, r_idx, c_idx]
        pred_st = p_simple_arr[test_indices, r_idx, c_idx]
        m_st = calc_all_metrics(obs_st, pred_st, threshold=0.1)
        table2_lines.append(
            f"{st_name:<8} | {m_st['MSE']:<12.2f} | {m_st['MAE']:<10.2f} | {m_st['RMSE']:<11.2f} | "
            f"{m_st['R2']:<8.3f} | {m_st['NSE']:<8.3f} | {m_st['KGE']:<8.3f} | "
            f"{m_st['POD']:<7.3f} | {m_st['FAR']:<7.3f} | {m_st['CSI']:<7.3f}"
        )
    table2_lines.append("=" * 108)
    table2_text = "\n".join(table2_lines)

    # 打印至控制台
    print(table1_text)
    print(table2_text)

    # 写入结果文本文件保存
    full_output_log = table1_text + "\n" + table2_text + "\n"
    with open(txt_result_path, "w", encoding="utf-8") as f:
        f.write(full_output_log)
    print(f"\n📝 简单平均评估报告已保存至: {txt_result_path}")

    # =========================================================================
    # 💾 5. 全流域掩膜与 NetCDF 导出
    # =========================================================================
    p_simple_arr[:, ~basin_mask_2d] = np.nan

    ds_simple = xr.Dataset(
        data_vars={
            'simple_average_precipitation': (['time', 'lat', 'lon'], p_simple_arr)
        },
        coords={
            'time': ds_gpm.time,
            'lat': ds_gpm.lat,
            'lon': ds_gpm.lon
        },
        attrs={
            'description': 'Equal-Weighted Simple Average of 3 Experts (P = (P1 + P2 + P3) / 3)',
            'spatial_mask': 'Clipped to irregular watershed boundary using DEM mask'
        }
    )
    ds_simple.to_netcdf(nc_simple_out)
    print(f"🎉 全流域简单平均融合降水 NetCDF 导出完成: {nc_simple_out}")

if __name__ == "__main__":
    main()
