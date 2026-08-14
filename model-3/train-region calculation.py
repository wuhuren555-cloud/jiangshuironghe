import os
import random
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =========================================================================
# 🔒 1. 锁死全局随机种子 (确保 100% 可复现)
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

# 设置站点极值分位数 (取历史 90% 分位数 P90 作为该站极值门槛)
EXTREME_QUANTILE = 0.90 

# =========================================================================
# 📌 2. 路径与设备配置
# =========================================================================
base_dir = r"C:\Users\26332\OneDrive\Desktop\sun mission\jiangshuironghe-MOE"

nc_path          = os.path.join(base_dir, "wanggeshuju", "juhe_gpm_daily_2012_2024.nc")
era5_ext_pt_path = os.path.join(base_dir, "ERA5数据", "CAPE    TCWV", "era5_extreme_features_2012_2024.pt")
dem_tif          = os.path.join(base_dir, "边界文件两幅", "过程文件", "touying-caijian.tif")
station_csv_path = os.path.join(base_dir, "shicezhandianshuju", "qingxiduiqi_2012_2024.csv")

era5_met_pt_path = None
target_met_file = "era5_met_features_2012_2024.pt"
for root, dirs, files in os.walk(base_dir):
    if target_met_file in files:
        era5_met_pt_path = os.path.join(root, target_met_file)
        break

save_model_dir = os.path.join(base_dir, "极值降水子模型")
os.makedirs(save_model_dir, exist_ok=True)
pth_model_path = os.path.join(save_model_dir, "best_ea_resnet_asset3.pth")

txt_result_path = os.path.join(base_dir, "三-降雨极值子模型", "结果.txt")
nc_asset3_out   = os.path.join(base_dir, "wanggeshuju", "juxing_asset3_2012_2024.nc")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ 计算设备: {DEVICE}")

# =========================================================================
# 🏗️ 3. 改进版：双向残差 PINeuralGPDNet 与 Huber 极值 Loss
# =========================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale

class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.ca(out)
        out = self.sa(out)
        out += residual
        return self.relu(out)

class PINeuralGPDNet(nn.Module):
    """【双向残差极值网络】：支持正向增雨与负向削降包络，并采用零初始化"""
    def __init__(self, in_channels=28, out_channels=1):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.block1 = DilatedResBlock(64, dilation=1)
        self.block2 = DilatedResBlock(64, dilation=2)
        self.block3 = DilatedResBlock(64, dilation=4)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1) # 允许自由正负残差输出
        )
        
        # 零初始化：使模型初始输出 Delta=0，训练起点稳定锚定在 GPM 上
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)

    def forward(self, x, gpm_base):
        feat = self.in_conv(x)
        feat = self.block1(feat)
        feat = self.block2(feat)
        feat = self.block3(feat)
        delta_p = self.out_conv(feat) # 双向残差 Delta P
        
        # 物理输出：支持双向增减，最终以 ReLU 保证整体非负产雨
        return F.relu(gpm_base + delta_p)

class StationQuantileExtremeLoss(nn.Module):
    """【自适应 Huber 极值 Loss】：平滑大残差，非对称惩罚极值低估"""
    def __init__(self, delta=5.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, extreme_mask):
        valid_count = torch.sum(extreme_mask)
        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        err = target - pred
        abs_err = torch.abs(err)
        
        # Huber 损失计算
        huber_loss = torch.where(
            abs_err <= self.delta,
            0.5 * (err ** 2),
            self.delta * abs_err - 0.5 * (self.delta ** 2)
        )
        
        # 非对称物理权重：对“低估暴雨”施加 1.3 倍惩罚
        asym_loss = torch.where(err > 0, 1.3 * huber_loss, huber_loss)
        masked_loss = asym_loss * extreme_mask
        
        return torch.sum(masked_loss) / (valid_count + 1e-8)

# =========================================================================
# 📊 4. 评估指标计算函数
# =========================================================================
def calc_metrics_quantile(obs, pred, threshold):
    ext_mask = ~np.isnan(obs) & ~np.isnan(pred) & (obs >= threshold)
    o, p = obs[ext_mask], pred[ext_mask]
    
    if len(o) < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    rmse = np.sqrt(np.mean((p - o) ** 2))
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 1e-6 else 0.0
    cc = np.corrcoef(o, p)[0, 1] if (np.std(o) > 1e-6 and np.std(p) > 1e-6) else 0.0

    hits = np.sum((o >= threshold) & (p >= threshold))
    misses = np.sum((o >= threshold) & (p < threshold))
    false_alarms = np.sum((o < threshold) & (p >= threshold))

    pod = hits / (hits + misses + 1e-8)
    far = false_alarms / (hits + false_alarms + 1e-8)
    csi = hits / (hits + misses + false_alarms + 1e-8)

    return rmse, r2, cc, pod, far, csi

# =========================================================================
# 📦 5. 数据集定义
# =========================================================================
class ExtremePrecipDataset(Dataset):
    def __init__(self, gpm, met, ext, dem, target, extreme_mask):
        self.num_days = gpm.shape[0]
        self.extreme_mask = extreme_mask
        self.gpm_raw = gpm
        dem_exp = dem.unsqueeze(0).repeat(self.num_days, 1, 1, 1)
        
        cape = ext[:, 0:1, :, :]
        tcwv = ext[:, 1:2, :, :]
        coupling_index = (cape / 1000.0) * (tcwv / 50.0)
        
        daily_feat_raw = torch.cat([gpm, met, ext, coupling_index, dem_exp], dim=1)
        self.mean = daily_feat_raw.mean(dim=(0, 2, 3), keepdim=True)
        self.std = daily_feat_raw.std(dim=(0, 2, 3), keepdim=True) + 1e-6
        self.norm_feat = (daily_feat_raw - self.mean) / self.std
        self.target = target

    def __len__(self):
        return self.num_days - 2

    def __getitem__(self, idx):
        t = idx + 1
        f_t1 = self.norm_feat[t-1, :-1]
        f_t2 = self.norm_feat[t, :-1]
        f_t3 = self.norm_feat[t+1, :-1]
        dem_single = self.norm_feat[t, -1:]
        
        x_3days = torch.cat([f_t1, f_t2, f_t3, dem_single], dim=0)
        gpm_base = self.gpm_raw[t]
        return x_3days, gpm_base, self.target[t], self.extreme_mask[t]

# =========================================================================
# 🚀 6. 主训练与评估程序
# =========================================================================
def main():
    print("=" * 75)
    print(f"🔥 构建【资产 3：双向残差+零初始化 各站 P{int(EXTREME_QUANTILE*100)} 极值专家】...")

    ds_gpm = xr.open_dataset(nc_path, engine="netcdf4").sel(time=slice("2012-01-01", "2024-12-31"))
    times, lats, lons = ds_gpm.time.values, ds_gpm.lat.values, ds_gpm.lon.values
    
    gpm_arr = np.nan_to_num(ds_gpm['precipitation'].transpose('time', 'lat', 'lon').values, nan=0.0)
    gpm_tensor = torch.tensor(gpm_arr, dtype=torch.float32).unsqueeze(1)
    num_days, _, H, W = gpm_tensor.shape

    # 流域边界模板读取
    match_template = ds_gpm['precipitation'].rename({"lon": "x", "lat": "y"}).rio.write_crs("EPSG:4326")
    da_dem = rioxarray.open_rasterio(dem_tif).rio.reproject_match(match_template)
    raw_dem_arr = da_dem.values[0]

    dem_arr_clean = np.nan_to_num(raw_dem_arr, nan=0.0)
    dem_norm = (dem_arr_clean - dem_arr_clean.mean()) / (dem_arr_clean.std() + 1e-6)
    dem_tensor = torch.tensor(dem_norm, dtype=torch.float32).unsqueeze(0)

    # 对齐 5通道 气象场与 2通道 极值特征
    era5_met = torch.load(era5_met_pt_path)
    era5_full_dates = pd.date_range("2012-01-01", "2024-12-31").strftime('%Y-%m-%d')
    era5_date_map = {d: i for i, d in enumerate(era5_full_dates)}
    gpm_date_strs = pd.to_datetime(times).strftime('%Y-%m-%d')
    matching_indices = [era5_date_map[d] for d in gpm_date_strs if d in era5_date_map]
    era5_met = era5_met[matching_indices]

    if era5_met.shape[2:] != (H, W):
        era5_met = F.interpolate(era5_met, size=(H, W), mode='bilinear', align_corners=True)

    era5_ext = torch.load(era5_ext_pt_path)
    if era5_ext.shape[0] != num_days:
        era5_ext = era5_ext[:num_days]

    # 实测数据与 P90 动态门槛计算
    df_station = pd.read_csv(station_csv_path)
    df_station['date'] = pd.to_datetime(df_station['date'])
    date_to_idx = {pd.Timestamp(d).strftime('%Y-%m-%d'): i for i, d in enumerate(times)}
    
    station_target = torch.zeros((num_days, 1, H, W), dtype=torch.float32)
    station_coords = {}
    station_p90_thresholds = {}

    print("\n📊 各站点 2012-2019 年历史降水 P90 极值专属门槛:")
    print("-" * 65)

    for _, row in df_station[['station_name', 'lat', 'lon']].drop_duplicates().iterrows():
        st_name = row['station_name']
        r_idx = int(np.abs(lats - row['lat']).argmin())
        c_idx = int(np.abs(lons - row['lon']).argmin())
        station_coords[st_name] = (r_idx, c_idx)
        
        st_data = df_station[df_station['station_name'] == st_name]
        for _, d_row in st_data.iterrows():
            d_str = d_row['date'].strftime('%Y-%m-%d')
            if d_str in date_to_idx:
                station_target[date_to_idx[d_str], 0, r_idx, c_idx] = float(d_row['station_rain'])
        
        train_st_rain = st_data[st_data['date'] < '2020-01-01']['station_rain'].values
        p90_val = float(np.percentile(train_st_rain, EXTREME_QUANTILE * 100))
        station_p90_thresholds[st_name] = p90_val
        print(f"  📍 站点【 {st_name:<4} 】| 坐标: ({r_idx:2d},{c_idx:2d}) | P90 极值门槛: {p90_val:5.2f} mm/d")

    print("-" * 65)

    extreme_mask_4d = torch.zeros((num_days, 1, H, W), dtype=torch.float32)
    for st_name, (r_idx, c_idx) in station_coords.items():
        st_thresh = station_p90_thresholds[st_name]
        st_rain_series = station_target[:, 0, r_idx, c_idx]
        is_extreme = (st_rain_series >= st_thresh).float()
        extreme_mask_4d[:, 0, r_idx, c_idx] = is_extreme

    train_mask_idx = (gpm_date_strs < '2020-01-01')
    val_mask_idx   = (gpm_date_strs >= '2020-01-01')

    train_ds = ExtremePrecipDataset(gpm_tensor[train_mask_idx], era5_met[train_mask_idx], era5_ext[train_mask_idx], dem_tensor, station_target[train_mask_idx], extreme_mask_4d[train_mask_idx])
    val_ds   = ExtremePrecipDataset(gpm_tensor[val_mask_idx], era5_met[val_mask_idx], era5_ext[val_mask_idx], dem_tensor, station_target[val_mask_idx], extreme_mask_4d[val_mask_idx])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = PINeuralGPDNet(in_channels=28, out_channels=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    criterion = StationQuantileExtremeLoss(delta=5.0)

    best_val_loss = float('inf')
    epochs = 35

    print(f"\n⚡ 开始双向残差极值专攻神经网络训练...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, gpm_b, y_b, m_b in train_loader:
            x_b, gpm_b, y_b, m_b = x_b.to(DEVICE), gpm_b.to(DEVICE), y_b.to(DEVICE), m_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_b, gpm_b)
            loss = criterion(pred, y_b, m_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_b.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, gpm_b, y_b, m_b in val_loader:
                x_b, gpm_b, y_b, m_b = x_b.to(DEVICE), gpm_b.to(DEVICE), y_b.to(DEVICE), m_b.to(DEVICE)
                pred = model(x_b, gpm_b)
                loss = criterion(pred, y_b, m_b)
                val_loss += loss.item() * x_b.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss and val_loss > 0:
            best_val_loss = val_loss
            torch.save(model.state_dict(), pth_model_path)
            saved_mark = "⭐ 最优模型保存"
        else:
            saved_mark = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {saved_mark}")

    # 推理 2012-2024
    print("\n" + "=" * 75)
    print("🌐 正在执行全流域各站 P90 双向极值场物理推理...")
    full_ds = ExtremePrecipDataset(gpm_tensor, era5_met, era5_ext, dem_tensor, station_target, extreme_mask_4d)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=False)

    model.load_state_dict(torch.load(pth_model_path, map_location=DEVICE))
    model.eval()

    pred_list = []
    with torch.no_grad():
        for x_b, gpm_b, _, _ in full_loader:
            pred = model(x_b.to(DEVICE), gpm_b.to(DEVICE)).cpu().numpy()
            pred_list.append(pred)

    full_pred = np.concatenate(pred_list, axis=0)[:, 0, :, :]
    first_day = full_pred[0:1]
    last_day  = full_pred[-1:]
    full_pred_padded = np.concatenate([first_day, full_pred, last_day], axis=0) # (4747, 25, 30)

    # 盲测复盘
    val_indices = np.where(val_mask_idx)[0]
    
    log_lines = []
    log_lines.append("==================================================================================")
    log_lines.append(f"📊【资产 3 各站 P90 双向极值专家】盲测站点复盘 (2020-2024)")
    log_lines.append("==================================================================================")

    ext_metrics_list = []

    for st_name, (r_idx, c_idx) in station_coords.items():
        obs_st = station_target.numpy()[val_indices, 0, r_idx, c_idx]
        pred_st = full_pred_padded[val_indices, r_idx, c_idx]
        st_p90 = station_p90_thresholds[st_name]
        
        rmse, r2, cc, pod, far, csi = calc_metrics_quantile(obs_st, pred_st, threshold=st_p90)
        ext_metrics_list.append([rmse, r2, cc, pod, far, csi])

        log_lines.append(f"📍 盲测极值【 {st_name:<4} 】(门槛>={st_p90:5.2f}mm/d) | RMSE: {rmse:6.2f} mm/d | R²: {r2:5.3f} | CC: {cc:5.3f} | POD: {pod:4.2f} | FAR: {far:4.2f} | CSI: {csi:4.2f}")

    avg_e = np.mean(ext_metrics_list, axis=0)
    
    log_lines.append("----------------------------------------------------------------------------------")
    log_lines.append(f"🏆【4 站专属 P90 极值平均】 | RMSE: {avg_e[0]:6.2f} mm/d | R²: {avg_e[1]:5.3f} | CC: {avg_e[2]:5.3f} | POD: {avg_e[3]:4.2f} | FAR: {avg_e[4]:4.2f} | CSI: {avg_e[5]:4.2f}")
    log_lines.append("==================================================================================")

    full_log_text = "\n".join(log_lines)
    print("\n" + full_log_text)
    
    with open(txt_result_path, "w", encoding="utf-8") as f:
        f.write(full_log_text + "\n\nBest Val Loss: " + f"{best_val_loss:.4f}\n")

    # 🌟【关键修改】：取消流域裁剪，保留完整矩形区域 (25x30) 以便 MOE 门控网络训练
    print("\n🌐 保持【全矩阵矩形区域 (25x30)】预测结果导出 (已取消 NaN 裁切，供后续 MOE 门控路由网络训练使用)...")

    ds_asset3 = xr.Dataset(
        data_vars={
            'asset3_precipitation': (['time', 'lat', 'lon'], full_pred_padded)
        },
        coords={
            'time': ds_gpm.time,
            'lat': ds_gpm.lat,
            'lon': ds_gpm.lon
        },
        attrs={
            'description': 'Asset 3: Extreme Precipitation Expert (Bidirectional Residual + P90 Quantile Focused)',
            'spatial_mask': 'Full rectangular grid (25x30) preserved for MOE routing',
            'extreme_thresholds': str(station_p90_thresholds)
        }
    )
    
    ds_asset3.to_netcdf(nc_asset3_out)
    
    print("=" * 75)
    print(f"🎉 🎉 🎉 资产 3 (全矩阵矩形导出版) 构建完工！已保存至:\n  📦 {nc_asset3_out}")
    print("=" * 75)

if __name__ == "__main__":
    main()
