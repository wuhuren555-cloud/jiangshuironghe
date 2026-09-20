import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random

# =========================================================================
# 🔒 1. 锁定全局随机种子，确保 100% 实验可复现
# =========================================================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# =========================================================================
# 📌 2. 基础路径配置
# =========================================================================
base_dir = r"C:\Users\26332\OneDrive\Desktop\sun mission\jiangshuironghe-MOE"

# 2.1 GPM 卫星降水网格 (25 x 30 基准)
nc_path = os.path.join(base_dir, "wanggeshuju", "juhe_gpm_daily_2012_2024.nc")

# 2.2 ERA5 气象张量 (.pt)
era5_pt_path = os.path.join(base_dir, "processed_asset2", "era5_met_features_2012_2024.pt")
if not os.path.exists(era5_pt_path):
    era5_pt_path = os.path.join(base_dir, "ERA5数据", "风向", "era5_met_features_2012_2024.pt")
if not os.path.exists(era5_pt_path):
    era5_pt_path = os.path.join(base_dir, "ERA5数据", "era5_met_features_2012_2024.pt")

# 2.3 边界文件与未裁剪的原始 DEM TIF (供 MOE 门控网络使用)
dem_tif = os.path.join(base_dir, "边界文件两幅", "过程文件", "touying.tif")
if not os.path.exists(dem_tif):
    dem_tif = os.path.join(base_dir, "边界文件两幅", "touying.tif")

# 2.4 实测站点 CSV 路径
station_csv_path = os.path.join(base_dir, "shicezhandianshuju", "qingxiduiqi_2012_2024.csv")

# 2.5 资产 2 输出路径
out_dir = os.path.join(base_dir, "时空非平稳-考虑天气系统子模型")
os.makedirs(out_dir, exist_ok=True)
nc_asset2_out = os.path.join(base_dir, "wanggeshuju", "juxing_asset2_3day_2012_2024.nc")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ 当前计算设备: {DEVICE}")

# =========================================================================
# 📊 3. 完整水文气象 10 大核心指标评估函数
# =========================================================================
def compute_all_metrics(obs, pred, threshold=0.1):
    """
    计算图片中规定的完整水文与统计指标:
    MAE (mm/d), MSE (mm²/d²), RMSE (mm/d), R², NSE, KGE, CC, POD, FAR, CSI
    """
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    obs = obs[mask]
    pred = pred[mask]
    
    if len(obs) == 0:
        return {k: np.nan for k in ["MAE", "MSE", "RMSE", "R2", "NSE", "KGE", "CC", "POD", "FAR", "CSI"]}

    diff = pred - obs
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)
    
    # R² 与 NSE
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8)) if ss_tot != 0 else np.nan
    nse = r2  # 单一序列下与决定系数数学等价
    
    # CC 与 KGE
    std_obs = np.std(obs)
    std_pred = np.std(pred)
    mean_obs = np.mean(obs)
    mean_pred = np.mean(pred)
    
    if std_obs > 1e-6 and std_pred > 1e-6:
        cc = np.corrcoef(obs, pred)[0, 1]
    else:
        cc = 0.0
        
    alpha = std_pred / (std_obs + 1e-8)
    beta = mean_pred / (mean_obs + 1e-8)
    kge = 1.0 - np.sqrt((cc - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)

    # 降水事件分类指标 (阈值 0.1 mm/d)
    obs_rain = obs >= threshold
    pred_rain = pred >= threshold

    hits = np.sum(obs_rain & pred_rain)          
    misses = np.sum(obs_rain & ~pred_rain)       
    false_alarms = np.sum(~obs_rain & pred_rain) 

    pod = hits / (hits + misses + 1e-8)                          
    far = false_alarms / (hits + false_alarms + 1e-8)            
    csi = hits / (hits + misses + false_alarms + 1e-8)           

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "NSE": nse,
        "KGE": kge,
        "CC": cc,
        "POD": pod,
        "FAR": far,
        "CSI": csi
    }

# =========================================================================
# 🏗️ 4. 构建 ST-UNet 深度神经网络
# =========================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class STUNet(nn.Module):
    def __init__(self, in_channels=21, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 32)                             
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))     
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))    
        self.bottleneck = DoubleConv(128, 256)                             
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   
        self.conv_up2 = DoubleConv(192, 128)  
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    
        self.conv_up1 = DoubleConv(96, 64)    
        
        self.outc = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Softplus()  # 保证物理非负输出
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        b = self.bottleneck(x3)
        
        u2 = self.up2(b)
        u2 = F.interpolate(u2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        u2 = torch.cat([u2, x2], dim=1) 
        x_up2 = self.conv_up2(u2)
        
        u1 = self.up1(x_up2)
        u1 = F.interpolate(u1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        u1 = torch.cat([u1, x1], dim=1) 
        x_up1 = self.conv_up1(u1)
        
        return self.outc(x_up1)

# =========================================================================
# 🎯 5. 稀疏掩码损失函数 (Sparse Masked Loss)
# =========================================================================
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        diff = (pred - target) ** 2
        masked_diff = diff * mask
        loss = torch.sum(masked_diff) / (torch.sum(mask) * pred.shape[0] + 1e-8)
        return loss

# =========================================================================
# 📦 6. 时空 Dataset 组帧类
# =========================================================================
class SpatiotemporalPrecipDataset(Dataset):
    def __init__(self, era5_tensor, gpm_tensor, dem_tensor, station_target, mask):
        self.num_days = era5_tensor.shape[0]
        self.mask = mask
        
        dem_expanded = dem_tensor.unsqueeze(0).repeat(self.num_days, 1, 1, 1)
        self.daily_features = torch.cat([gpm_tensor, era5_tensor, dem_expanded], dim=1) 
        
        self.mean = self.daily_features.mean(dim=(0, 2, 3), keepdim=True)
        self.std = self.daily_features.std(dim=(0, 2, 3), keepdim=True) + 1e-6
        self.daily_features_norm = (self.daily_features - self.mean) / self.std
        
        self.target = station_target

    def __len__(self):
        return self.num_days - 2

    def __getitem__(self, idx):
        t = idx + 1
        f_prev = self.daily_features_norm[t-1]
        f_curr = self.daily_features_norm[t]
        f_next = self.daily_features_norm[t+1]
        
        x_3days = torch.cat([f_prev, f_curr, f_next], dim=0)
        y_curr = self.target[t]
        
        return x_3days, y_curr

# =========================================================================
# 🖨️ 7. 标准化学术表格打印函数
# =========================================================================
def print_metrics_table(title, results_dict):
    """打印规范的站点详细指标表格"""
    print("\n" + "=" * 115)
    print(f"📊 {title}")
    print("=" * 115)
    header = f"{'站点 (Station)':<14} | {'MAE (mm/d)':<10} | {'MSE (mm²/d²)':<12} | {'RMSE (mm/d)':<11} | {'R²':<7} | {'NSE':<7} | {'KGE':<7} | {'CC':<7} | {'POD':<6} | {'FAR':<6} | {'CSI':<6}"
    print(header)
    print("-" * 115)
    
    for st_name, m in results_dict.items():
        if st_name == "【4 站全局平均】":
            print("-" * 115)
        print(f"{st_name:<14} | {m['MAE']:<10.2f} | {m['MSE']:<12.2f} | {m['RMSE']:<11.2f} | {m['R2']:<7.3f} | {m['NSE']:<7.3f} | {m['KGE']:<7.3f} | {m['CC']:<7.3f} | {m['POD']:<6.2f} | {m['FAR']:<6.2f} | {m['CSI']:<6.2f}")
    print("=" * 115)

# =========================================================================
# 🚀 8. 主执行逻辑
# =========================================================================
def main():
    print("=" * 80)
    print("正在对齐空间网格，准备 ST-UNet 数据集 (2012-2019训练 / 2020-2021验证 / 2022-2024测试)...")
    
    # 8.1 载入 GPM NetCDF
    ds_gpm = xr.open_dataset(nc_path, engine="netcdf4").sel(time=slice("2012-01-01", "2024-12-31"))
    ds_gpm['precipitation'] = ds_gpm['precipitation'].transpose('time', 'lat', 'lon')
    
    times = ds_gpm.time.values
    lats = ds_gpm.lat.values
    lons = ds_gpm.lon.values
    
    gpm_arr = ds_gpm['precipitation'].values
    gpm_arr_clean = np.nan_to_num(gpm_arr, nan=0.0)
    gpm_tensor = torch.tensor(gpm_arr_clean, dtype=torch.float32).unsqueeze(1)
    num_days, _, H, W = gpm_tensor.shape
    print(f"📐 GPM 空间维度对齐标准: Height={H}, Width={W}, 总天数={num_days}")
    
    # 8.2 载入未裁剪 DEM TIF 并对齐
    match_template = ds_gpm['precipitation'].rename({"lon": "x", "lat": "y"}).rio.write_crs("EPSG:4326")
    da_dem = rioxarray.open_rasterio(dem_tif).rio.reproject_match(match_template)
    dem_arr = da_dem.values[0]
    
    dem_arr_clean = np.nan_to_num(dem_arr, nan=0.0)
    dem_norm = (dem_arr_clean - dem_arr_clean.mean()) / (dem_arr_clean.std() + 1e-6)
    dem_tensor = torch.tensor(dem_norm, dtype=torch.float32).unsqueeze(0)

    # 8.3 载入 ERA5 张量并强对齐时间
    era5_data = torch.load(era5_pt_path)
    gpm_date_strs = pd.to_datetime(times).strftime('%Y-%m-%d')
    era5_full_dates = pd.date_range("2012-01-01", "2024-12-31").strftime('%Y-%m-%d')
    era5_date_map = {d: i for i, d in enumerate(era5_full_dates)}
    
    matching_indices = [era5_date_map[d] for d in gpm_date_strs if d in era5_date_map]
    era5_data = era5_data[matching_indices]
    
    if era5_data.shape[2:] != (H, W):
        era5_data = F.interpolate(era5_data, size=(H, W), mode='bilinear', align_corners=True)

    # 8.4 映射 4 个实测雨量站
    df_station = pd.read_csv(station_csv_path)
    df_station['date'] = pd.to_datetime(df_station['date'])
    date_to_idx = {pd.Timestamp(d).strftime('%Y-%m-%d'): i for i, d in enumerate(times)}
    
    mask = torch.zeros((1, 1, H, W), dtype=torch.float32)
    station_target = torch.zeros((num_days, 1, H, W), dtype=torch.float32)
    station_coords = {}
    
    unique_stations = df_station[['station_name', 'lat', 'lon']].drop_duplicates()
    print(f"\n📍 识别到 {len(unique_stations)} 个实测雨量站，精准对齐网格坐标:")
    for _, row in unique_stations.iterrows():
        st_name = row['station_name']
        st_lat, st_lon = row['lat'], row['lon']
        r_idx = int(np.abs(lats - st_lat).argmin())
        c_idx = int(np.abs(lons - st_lon).argmin())
        
        mask[0, 0, r_idx, c_idx] = 1.0
        station_coords[st_name] = (r_idx, c_idx)
        print(f"   • 站点 [{st_name}]: 坐标 ({st_lat:.2f}, {st_lon:.2f}) ➔ 像素索引 [{r_idx}, {c_idx}]")
        
        st_data = df_station[df_station['station_name'] == st_name]
        for _, d_row in st_data.iterrows():
            d_str = d_row['date'].strftime('%Y-%m-%d')
            if d_str in date_to_idx:
                t_i = date_to_idx[d_str]
                station_target[t_i, 0, r_idx, c_idx] = float(d_row['station_rain'])

    # =========================================================================
    # 8.5 重新划分时段: 训练集 (2012–2019) | 验证集 (2020–2021) | 测试集 (2022–2024)
    # =========================================================================
    train_mask = (gpm_date_strs >= '2012-01-01') & (gpm_date_strs <= '2019-12-31')
    val_mask   = (gpm_date_strs >= '2020-01-01') & (gpm_date_strs <= '2021-12-31')
    test_mask  = (gpm_date_strs >= '2022-01-01') & (gpm_date_strs <= '2024-12-31')
    
    train_dataset = SpatiotemporalPrecipDataset(era5_data[train_mask], gpm_tensor[train_mask], dem_tensor, station_target[train_mask], mask)
    val_dataset   = SpatiotemporalPrecipDataset(era5_data[val_mask],   gpm_tensor[val_mask],   dem_tensor, station_target[val_mask],   mask)
    
    print(f"\n📊 新三段式划分: 训练集={len(train_dataset)}天 (2012-2019) | 验证集={len(val_dataset)}天 (2020-2021) | 独立盲测={np.sum(test_mask)-2}天 (2022-2024)")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    
    # 8.6 训练 ST-UNet 算法
    model = STUNet(in_channels=21, out_channels=1).to(DEVICE)
    criterion = MaskedMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    
    print(f"\n🏋️ 开始训练 ST-UNet 时空非平稳模型 (共 30 Epochs)...")
    best_loss = float('inf')
    best_model_path = os.path.join(out_dir, "best_stunet_asset2-3天.pth")
    
    for epoch in range(1, 31):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b, mask.to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_b.size(0)
            
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                pred = model(x_b)
                loss = criterion(pred, y_b, mask.to(DEVICE))
                val_loss += loss.item() * x_b.size(0)
        val_loss /= len(val_dataset)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/30] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ⭐ Best Val: {best_loss:.4f}")

    # 8.7 全矩形网格 2D 推理演算
    print("\n🌐 正在进行全矩形网格 (25x30) 2012-2024 全时间轴推理演算...")
    full_dataset = SpatiotemporalPrecipDataset(era5_data, gpm_tensor, dem_tensor, station_target, mask)
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
    
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for x_b, _ in full_loader:
            pred = model(x_b.to(DEVICE))
            predictions.append(pred.cpu().numpy())
            
    pred_all = np.concatenate(predictions, axis=0)[:, 0, :, :] 
    valid_dates = gpm_date_strs[1:-1]  # 扣除前后 1 天窗口

    # =========================================================================
    # 8.8 分别评估并输出：训练集、验证集、测试集的各站点详细指标表
    # =========================================================================
    split_configs = {
        "表 1：训练集 (2012–2019) 站点详细指标评估": (valid_dates >= '2012-01-01') & (valid_dates <= '2019-12-31'),
        "表 2：验证集 (2020–2021) 站点详细指标评估": (valid_dates >= '2020-01-01') & (valid_dates <= '2021-12-31'),
        "表 3：独立测试集 (2022–2024) 站点详细指标评估": (valid_dates >= '2022-01-01') & (valid_dates <= '2024-12-31')
    }
    
    summary_splits = {}

    for title, period_mask in split_configs.items():
        period_indices = np.where(period_mask)[0]
        results = {}
        
        for st_name, (r, c) in station_coords.items():
            st_data = df_station[df_station['station_name'] == st_name]
            st_date_map = {d.strftime('%Y-%m-%d'): v for d, v in zip(st_data['date'], st_data['station_rain'])}
            
            obs_list, pred_list = [], []
            for idx in period_indices:
                d_str = valid_dates[idx]
                if d_str in st_date_map:
                    obs_list.append(st_date_map[d_str])
                    pred_list.append(pred_all[idx, r, c])
                    
            metrics = compute_all_metrics(np.array(obs_list), np.array(pred_list))
            results[st_name] = metrics
            
        # 计算该时段的全局平均指标
        avg_m = {k: np.nanmean([results[st][k] for st in station_coords]) for k in results[st_name]}
        results["【4 站全局平均】"] = avg_m
        
        # 打印当前时段单表
        print_metrics_table(title, results)
        
        # 暂存全局平均用于汇总表
        split_short_name = title.split("：")[1].split(" ")[0]
        summary_splits[split_short_name] = avg_m

    # =========================================================================
    # 8.9 输出 表 4：三集全局平均指标横向对比汇总表
    # =========================================================================
    print_metrics_table("表 4：训练集、验证集与独立测试集【4 站全局平均】横向汇总对照表", summary_splits)

    # 8.10 导出 NetCDF
    ds_asset2 = xr.Dataset(
        data_vars=dict(
            asset2_precipitation=(["time", "lat", "lon"], pred_all)
        ),
        coords=dict(
            time=times[1:-1],
            lat=lats,
            lon=lons,
        ),
        attrs=dict(
            description="北江中上游基于 ST-UNet 结合 ERA5 与风场水汽的时空非平稳融合降水资产2 (全矩形网格未裁剪版)",
            spatial_resolution="0.1deg x 0.1deg",
            time_coverage="2012-01-02 to 2024-12-30",
            units="mm/d"
        )
    )
    ds_asset2.to_netcdf(nc_asset2_out)
    
    print("\n" + "=" * 80)
    print(f"🎉 🎉 🎉 全矩形未裁剪版核心资产 2 构建完工！已成功导出！")
    print(f"📦 NC 文件路径: {nc_asset2_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()
