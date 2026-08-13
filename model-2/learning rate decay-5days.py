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

# 开启全局种子锁定
seed_everything(42)

# =========================================================================
# 📌 1. 基础路径配置（与资产 1 保持绝对一致）
# =========================================================================
base_dir = r"C:\Users\26332\OneDrive\Desktop\sun mission\jiangshuironghe-MOE"

# 1.1 GPM 卫星降水网格 (25 x 30 基准)
nc_path = os.path.join(base_dir, "wanggeshuju", "juhe_gpm_daily_2012_2024.nc")

# 1.2 ERA5 气象张量 (.pt)
era5_pt_path = os.path.join(base_dir, "processed_asset2", "era5_met_features_2012_2024.pt")
if not os.path.exists(era5_pt_path):
    era5_pt_path = os.path.join(base_dir, "ERA5数据", "era5_met_features_2012_2024.pt")

# 1.3 边界文件与裁剪好的 DEM TIF
dem_tif = os.path.join(base_dir, "边界文件两幅", "过程文件", "touying-caijian.tif")

# 1.4 实测站点 CSV 路径
station_csv_path = os.path.join(base_dir, "shicezhandianshuju", "qingxiduiqi_2012_2024.csv")

# 1.5 资产 2 输出路径
out_dir = os.path.join(base_dir, "时空非平稳-考虑天气系统子模型")
os.makedirs(out_dir, exist_ok=True)
nc_asset2_out = os.path.join(base_dir, "wanggeshuju", "quanliuyu_asset2_nonstationarity_2012_2024.nc")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ 当前计算设备: {DEVICE}")

# =========================================================================
# 📊 2. 水文遥感核心指标评估函数
# =========================================================================
def compute_hydrological_metrics(obs, pred, threshold=0.1):
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    obs = obs[mask]
    pred = pred[mask]
    
    if len(obs) == 0:
        return {"RMSE": np.nan, "R2": np.nan, "CC": np.nan, "POD": np.nan, "FAR": np.nan, "CSI": np.nan}

    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8)) if ss_tot != 0 else np.nan
    
    if np.std(obs) > 1e-6 and np.std(pred) > 1e-6:
        cc = np.corrcoef(obs, pred)[0, 1]
    else:
        cc = 0.0

    obs_rain = obs >= threshold
    pred_rain = pred >= threshold

    hits = np.sum(obs_rain & pred_rain)          
    misses = np.sum(obs_rain & ~pred_rain)       
    false_alarms = np.sum(~obs_rain & pred_rain) 

    pod = hits / (hits + misses + 1e-8)                          
    far = false_alarms / (hits + false_alarms + 1e-8)            
    csi = hits / (hits + misses + false_alarms + 1e-8)           

    return {
        "RMSE": rmse,
        "R2": r2,
        "CC": cc,
        "POD": pod,
        "FAR": far,
        "CSI": csi
    }

# =========================================================================
# 🏗️ 3. 构建 ST-UNet 深度神经网络 (5 天窗口，35 通道输入)
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
    """
    Input: (B, 35, H, W) [5天窗口 * (1 GPM + 5 ERA5 + 1 DEM) = 35 通道]
    Output: (B, 1, H, W) [t时刻修正后的 2D 降水场]
    """
    def __init__(self, in_channels=35, out_channels=1):
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
            nn.Softplus() # 非负激活
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
        
        out = self.outc(x_up1)
        return out

# =========================================================================
# 🎯 4. 稀疏掩码损失函数 (Sparse Masked Loss)
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
# 📦 5. 5天滑动窗口 Dataset 组帧类
# =========================================================================
class SpatiotemporalPrecipDataset(Dataset):
    def __init__(self, era5_tensor, gpm_tensor, dem_tensor, station_target, mask):
        self.num_days = era5_tensor.shape[0]
        self.mask = mask
        
        dem_expanded = dem_tensor.unsqueeze(0).repeat(self.num_days, 1, 1, 1)
        self.daily_features = torch.cat([gpm_tensor, era5_tensor, dem_expanded], dim=1) # (Days, 7, H, W)
        
        self.mean = self.daily_features.mean(dim=(0, 2, 3), keepdim=True)
        self.std = self.daily_features.std(dim=(0, 2, 3), keepdim=True) + 1e-6
        self.daily_features_norm = (self.daily_features - self.mean) / self.std
        
        self.target = station_target

    def __len__(self):
        # 5 天滑动窗口，前后各扣除 2 天
        return self.num_days - 4

    def __getitem__(self, idx):
        t = idx + 2 # 中心时刻 t
        f0 = self.daily_features_norm[t-2]
        f1 = self.daily_features_norm[t-1]
        f2 = self.daily_features_norm[t]
        f3 = self.daily_features_norm[t+1]
        f4 = self.daily_features_norm[t+2]
        
        # 连续 5 天特征拼接为 35 通道张量
        x_5days = torch.cat([f0, f1, f2, f3, f4], dim=0)
        y_curr = self.target[t]
        
        return x_5days, y_curr

# =========================================================================
# 🚀 6. 主训练与微调评估逻辑
# =========================================================================
def main():
    print("=" * 70)
    print("正在加载 GPM, DEM 与 ERA5，构建 5 天滑动窗口 ST-UNet 数据集...")
    
    # 6.1 载入 GPM NetCDF
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
    
    # 6.2 载入 DEM TIF 并重采样对齐
    match_template = ds_gpm['precipitation'].rename({"lon": "x", "lat": "y"}).rio.write_crs("EPSG:4326")
    da_dem = rioxarray.open_rasterio(dem_tif).rio.reproject_match(match_template)
    dem_arr = da_dem.values[0]
    
    valid_mask_2d = ~np.isnan(dem_arr) & (dem_arr > -100)
    dem_arr_clean = np.nan_to_num(dem_arr, nan=0.0)
    dem_norm = (dem_arr_clean - dem_arr_clean.mean()) / (dem_arr_clean.std() + 1e-6)
    dem_tensor = torch.tensor(dem_norm, dtype=torch.float32).unsqueeze(0)

    # 6.3 载入 ERA5 pt 张量并强对齐
    era5_data = torch.load(era5_pt_path)
    
    gpm_date_strs = pd.to_datetime(times).strftime('%Y-%m-%d')
    era5_full_dates = pd.date_range("2012-01-01", "2024-12-31").strftime('%Y-%m-%d')
    era5_date_map = {d: i for i, d in enumerate(era5_full_dates)}
    
    matching_indices = [era5_date_map[d] for d in gpm_date_strs if d in era5_date_map]
    era5_data = era5_data[matching_indices]
    
    if era5_data.shape[2:] != (H, W):
        era5_data = F.interpolate(era5_data, size=(H, W), mode='bilinear', align_corners=True)

    # 6.4 映射实测雨量站
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

    # 6.5 三段式划分数据集
    train_mask = (gpm_date_strs >= '2012-01-01') & (gpm_date_strs <= '2017-12-31')
    val_mask   = (gpm_date_strs >= '2018-01-01') & (gpm_date_strs <= '2019-12-31')
    
    train_dataset = SpatiotemporalPrecipDataset(era5_data[train_mask], gpm_tensor[train_mask], dem_tensor, station_target[train_mask], mask)
    val_dataset   = SpatiotemporalPrecipDataset(era5_data[val_mask],   gpm_tensor[val_mask],   dem_tensor, station_target[val_mask],   mask)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    
    # 6.6 初始化 35 通道模型与微调优化器 (余弦退火学习率)
    model = STUNet(in_channels=35, out_channels=1).to(DEVICE)
    criterion = MaskedMSELoss()
    
    # 微调参数配置: AdamW + 5e-4 初始 LR + 余弦退火调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    
    print(f"\n🏋️ 开始训练 5 天窗口 ST-UNet 深度模型 (共 30 Epochs，含 Cosine LR 衰减)...")
    best_loss = float('inf')
    best_model_path = os.path.join(out_dir, "best_stunet_asset2.pth")
    
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
        
        # 验证集挑选最优模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                pred = model(x_b)
                loss = criterion(pred, y_b, mask.to(DEVICE))
                val_loss += loss.item() * x_b.size(0)
        val_loss /= len(val_dataset)
        
        # 学习率动态退火步进
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/30] | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ⭐ Best Val: {best_loss:.4f}")

    # 6.7 全流域 2D 时空推理演算
    print("\n🌐 正在全流域全时间轴 (35 通道 5 天窗口) 推理演算...")
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
    
    # 6.8 📊 盲测站点详细指标复盘 (2020-2024 测试集)
    valid_dates = gpm_date_strs[2:-2] # 扣除前后 2 天窗口
    test_date_indices = np.where(valid_dates >= '2020-01-01')[0]
    
    print("\n" + "=" * 75)
    print("📊【资产 2 ST-UNet 5天窗口模型】盲测站点详细指标复盘 (2020-2024)")
    print("=" * 75)
    
    st_results = []
    for st_name, (r, c) in station_coords.items():
        st_data = df_station[df_station['station_name'] == st_name]
        st_date_map = {d.strftime('%Y-%m-%d'): v for d, v in zip(st_data['date'], st_data['station_rain'])}
        
        obs_list, pred_list = [], []
        for idx in test_date_indices:
            d_str = valid_dates[idx]
            if d_str in st_date_map:
                obs_list.append(st_date_map[d_str])
                pred_list.append(pred_all[idx, r, c])
                
        metrics = compute_hydrological_metrics(np.array(obs_list), np.array(pred_list))
        st_results.append(metrics)
        
        print(f"📍 盲测站点【 {st_name:<3} 】| RMSE: {metrics['RMSE']:6.2f} mm/d | R²: {metrics['R2']:.3f} | CC: {metrics['CC']:.3f} | POD: {metrics['POD']:.2f} | FAR: {metrics['FAR']:.2f} | CSI: {metrics['CSI']:.2f}")

    avg_rmse = np.nanmean([m['RMSE'] for m in st_results])
    avg_r2   = np.nanmean([m['R2']   for m in st_results])
    avg_cc   = np.nanmean([m['CC']   for m in st_results])
    avg_pod  = np.nanmean([m['POD']  for m in st_results])
    avg_far  = np.nanmean([m['FAR']  for m in st_results])
    avg_csi  = np.nanmean([m['CSI']  for m in st_results])
    
    print("-" * 75)
    print(f"🏆 【4 站全局平均】 | RMSE: {avg_rmse:6.2f} mm/d | R²: {avg_r2:.3f} | CC: {avg_cc:.3f} | POD: {avg_pod:.2f} | FAR: {avg_far:.2f} | CSI: {avg_csi:.2f}")
    print("=" * 75 + "\n")

    # 6.9 掩膜流域外无效背景区域并导出 NetCDF 资产 2
    pred_all_masked = np.full_like(pred_all, np.nan)
    for d in range(pred_all.shape[0]):
        pred_all_masked[d][valid_mask_2d] = pred_all[d][valid_mask_2d]
    
    ds_asset2 = xr.Dataset(
        data_vars=dict(
            asset2_precipitation=(["time", "lat", "lon"], pred_all_masked)
        ),
        coords=dict(
            time=times[2:-2], # 对齐 5 天窗口时间轴
            lat=lats,
            lon=lons,
        ),
        attrs=dict(
            description="北江中上游基于 5 天窗口 ST-UNet 结合 ERA5 与风场水汽的时空非平稳融合降水资产2",
            spatial_resolution="0.1deg x 0.1deg",
            time_coverage="2012-01-03 to 2024-12-29",
            units="mm/d"
        )
    )
    
    ds_asset2.to_netcdf(nc_asset2_out)
    
    print("=" * 70)
    print(f"🎉 🎉 🎉 核心资产 2 (5 天窗口) 构建完工！已成功导出！")
    print(f"📦 NC 文件路径: {nc_asset2_out}")
    print("=" * 70)

if __name__ == "__main__":
    main()
