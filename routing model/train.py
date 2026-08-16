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
# 🔒 1. 锁死全局随机种子 (确保 100% 实验可复现)
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

DRIZZLE_THRESHOLD = 0.15

# =========================================================================
# 📌 2. 路径与设备配置
# =========================================================================
base_dir = r"C:\Users\26332\OneDrive\Desktop\sun mission\jiangshuironghe-MOE"

router_dir          = os.path.join(base_dir, "路由模型")
router_inputs_pt    = os.path.join(router_dir, "router_inputs_2012_2024.pt")
expert_preds_pt     = os.path.join(router_dir, "expert_predictions_2012_2024.pt")

nc_gpm_path         = os.path.join(base_dir, "wanggeshuju", "juhe_gpm_daily_2012_2024.nc")
dem_tif             = os.path.join(base_dir, "边界文件两幅", "过程文件", "touying-caijian.tif")
station_csv_path    = os.path.join(base_dir, "shicezhandianshuju", "qingxiduiqi_2012_2024.csv")

pth_router_path     = os.path.join(router_dir, "best_top2_router_model.pth")
txt_result_path     = os.path.join(router_dir, "结果.txt")
nc_moe_fusion_out   = os.path.join(base_dir, "wanggeshuju", "quanliuyu_moe_fusion_2012_2024.nc")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ 计算设备: {DEVICE}")

# =========================================================================
# 🏗️ 3. 深度情景感知门控路由器 (Deep Regime-Aware Router Network)
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

class PairwiseRankingRouter(nn.Module):
    """【成对排序门控路由器】：
    1. 39 维时空特征输入 (33 环境特征 + 3 预测 + 3 显式残差)；
    2. 双重 CBAM 注意力与扩张卷积决策；
    3. 输出未归一化 Logits (供 Margin Ranking Loss 直接约束次序) 与 Softmax 权重。
    """
    def __init__(self, in_channels=33, num_experts=3):
        super().__init__()
        total_in_channels = in_channels + num_experts + 3
        
        self.stem = nn.Sequential(
            nn.Conv2d(total_in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, num_experts, kernel_size=1)
        )

    def forward(self, x, exp_b):
        p1 = exp_b[:, 0:1, :, :]
        p2 = exp_b[:, 1:2, :, :]
        p3 = exp_b[:, 2:3, :, :]

        p_max = torch.max(exp_b, dim=1, keepdim=True)[0]
        diff_12 = F.relu(p1 - p2)
        diff_32 = F.relu(p3 - p2)

        feat_combined = torch.cat([x, exp_b, p_max, diff_12, diff_32], dim=1) # (B, 39, H, W)
        
        feat = self.stem(feat_combined)
        feat = self.ca(feat)
        feat = self.sa(feat)
        logits = self.head(feat) # (B, 3, H, W)

        # 保持处处可微的 Smooth Softmax (无硬截断，彻底避免梯度死锁)
        weights = F.softmax(logits, dim=1)
        return weights, logits

# =========================================================================
# 🌟 4. 显式 Top-2 成对对序边际损失 (Pairwise Top-2 Margin Ranking Loss)
# =========================================================================
class PairwiseTop2MarginLoss(nn.Module):
    """【显式 Top-2 对序边际损失】：
    1. 主预测加权 Huber Loss (保证总体拟合)；
    2. 显式成对对序 Hinge 损失：强迫最优前两名专家的 Logit 至少比最差专家高出 margin (m=1.2)。
    """
    def __init__(self, delta=3.0, rank_weight=0.8, margin=1.2):
        super().__init__()
        self.delta = delta
        self.rank_w = rank_weight
        self.margin = margin

    def forward(self, p_fusion, expert_preds, logits, target, station_mask):
        valid_count = torch.sum(station_mask)
        if valid_count == 0:
            return torch.tensor(0.0, device=p_fusion.device, requires_grad=True)

        # 1. 主预测加权 Huber Loss
        err = target - p_fusion
        abs_err = torch.abs(err)
        huber_loss = torch.where(
            abs_err <= self.delta,
            0.5 * (err ** 2),
            self.delta * abs_err - 0.5 * (self.delta ** 2)
        )
        sample_weights = 1.0 + (target / 15.0) ** 2
        weighted_huber = huber_loss * sample_weights * station_mask
        loss_pred = torch.sum(weighted_huber) / (torch.sum(station_mask * sample_weights) + 1e-8)

        # 🌟 2. 显式 Top-2 成对对序边际损失 (Pairwise Margin Loss)
        # 计算三大专家的实测绝对误差: (B, 3, H, W)
        expert_abs_errors = torch.abs(expert_preds - target)
        
        # 提取各个专家的误差与 Logits
        e1, e2, e3 = expert_abs_errors[:, 0:1, :, :], expert_abs_errors[:, 1:2, :, :], expert_abs_errors[:, 2:3, :, :]
        z1, z2, z3 = logits[:, 0:1, :, :], logits[:, 1:2, :, :], logits[:, 2:3, :, :]

        # 判定哪个专家是真实误差最大的“劣质专家 (Worst)”
        is_e1_worst = (e1 >= e2) & (e1 >= e3)
        is_e2_worst = (e2 >= e1) & (e2 >= e3)
        is_e3_worst = (e3 >= e1) & (e3 >= e2)

        # 当专家 1 最差时，惩罚 z2 - z1 < m 和 z3 - z1 < m
        loss_w1 = F.relu(self.margin - (z2 - z1)) + F.relu(self.margin - (z3 - z1))
        # 当专家 2 最差时，惩罚 z1 - z2 < m 和 z3 - z2 < m
        loss_w2 = F.relu(self.margin - (z1 - z2)) + F.relu(self.margin - (z3 - z2))
        # 当专家 3 最差时 (如中雨日)，惩罚 z1 - z3 < m 和 z2 - z3 < m
        loss_w3 = F.relu(self.margin - (z1 - z3)) + F.relu(self.margin - (z2 - z3))

        pairwise_rank_loss = (
            is_e1_worst.float() * loss_w1 +
            is_e2_worst.float() * loss_w2 +
            is_e3_worst.float() * loss_w3
        )

        weighted_rank = pairwise_rank_loss * sample_weights * station_mask
        loss_rank = torch.sum(weighted_rank) / (torch.sum(station_mask * sample_weights) + 1e-8)

        return loss_pred + self.rank_w * loss_rank

# =========================================================================
# 📊 5. 学术指标计算
# =========================================================================
def calc_metrics(obs, pred, threshold=0.1):
    valid_mask = ~np.isnan(obs) & ~np.isnan(pred)
    o, p = obs[valid_mask], pred[valid_mask]
    if len(o) == 0:
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

class MOEDataset(Dataset):
    def __init__(self, router_inputs, expert_preds, target):
        self.x = router_inputs
        self.experts = expert_preds
        self.target = target

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.experts[idx], self.target[idx]

# =========================================================================
# 🚀 6. 主训练与导出程序
# =========================================================================
def main():
    print("=" * 75)
    print("🔥 开始训练【成对排序约束 Top-2 优化 MOE 路由系统】...")

    router_inputs = torch.load(router_inputs_pt)
    expert_preds  = torch.load(expert_preds_pt)
    num_days, _, H, W = router_inputs.shape

    ds_gpm = xr.open_dataset(nc_gpm_path, engine="netcdf4").sel(time=slice("2012-01-01", "2024-12-31"))
    times, lats, lons = ds_gpm.time.values, ds_gpm.lat.values, ds_gpm.lon.values
    gpm_dates_str = pd.to_datetime(times).strftime('%Y-%m-%d')

    match_template = ds_gpm['precipitation'].rename({"lon": "x", "lat": "y"}).rio.write_crs("EPSG:4326")
    da_dem = rioxarray.open_rasterio(dem_tif).rio.reproject_match(match_template)
    raw_dem_arr = da_dem.values[0]
    basin_mask_2d = ~np.isnan(raw_dem_arr) & (raw_dem_arr > -100)

    df_station = pd.read_csv(station_csv_path)
    df_station['date'] = pd.to_datetime(df_station['date'])
    date_to_idx = {pd.Timestamp(d).strftime('%Y-%m-%d'): i for i, d in enumerate(times)}
    
    mask = torch.zeros((1, 1, H, W), dtype=torch.float32)
    station_target = torch.zeros((num_days, 1, H, W), dtype=torch.float32)
    station_coords = {}

    for _, row in df_station[['station_name', 'lat', 'lon']].drop_duplicates().iterrows():
        st_name = row['station_name']
        r_idx = int(np.abs(lats - row['lat']).argmin())
        c_idx = int(np.abs(lons - row['lon']).argmin())
        mask[0, 0, r_idx, c_idx] = 1.0
        station_coords[st_name] = (r_idx, c_idx)
        
        st_data = df_station[df_station['station_name'] == st_name]
        for _, d_row in st_data.iterrows():
            d_str = d_row['date'].strftime('%Y-%m-%d')
            if d_str in date_to_idx:
                station_target[date_to_idx[d_str], 0, r_idx, c_idx] = float(d_row['station_rain'])

    train_mask_idx = (gpm_dates_str < '2020-01-01')
    val_mask_idx   = (gpm_dates_str >= '2020-01-01')

    train_ds = MOEDataset(router_inputs[train_mask_idx], expert_preds[train_mask_idx], station_target[train_mask_idx])
    val_ds   = MOEDataset(router_inputs[val_mask_idx], expert_preds[val_mask_idx], station_target[val_mask_idx])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

    router_model = PairwiseRankingRouter(in_channels=33, num_experts=3).to(DEVICE)
    optimizer = torch.optim.AdamW(router_model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = PairwiseTop2MarginLoss(delta=3.0, rank_weight=0.8, margin=1.2)

    best_val_loss = float('inf')
    epochs = 40

    print("\n⚡ 开始成对排序 MOE 路由训练 (40 Epochs)...")
    for epoch in range(1, epochs + 1):
        router_model.train()
        train_loss = 0.0
        for x_b, exp_b, y_b in train_loader:
            x_b, exp_b, y_b = x_b.to(DEVICE), exp_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            
            weights, logits = router_model(x_b, exp_b)
            p_fusion = torch.sum(weights * exp_b, dim=1, keepdim=True)
            
            asset1_p = exp_b[:, 0:1, :, :]
            p_fusion_gated = torch.where(
                (asset1_p < DRIZZLE_THRESHOLD) | (p_fusion < DRIZZLE_THRESHOLD),
                torch.tensor(0.0, device=p_fusion.device),
                p_fusion
            )
            
            loss = criterion(p_fusion_gated, exp_b, logits, y_b, mask.to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_b.size(0)
        train_loss /= len(train_ds)

        router_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, exp_b, y_b in val_loader:
                x_b, exp_b, y_b = x_b.to(DEVICE), exp_b.to(DEVICE), y_b.to(DEVICE)
                weights, logits = router_model(x_b, exp_b)
                p_fusion = torch.sum(weights * exp_b, dim=1, keepdim=True)
                
                asset1_p = exp_b[:, 0:1, :, :]
                p_fusion_gated = torch.where(
                    (asset1_p < DRIZZLE_THRESHOLD) | (p_fusion < DRIZZLE_THRESHOLD),
                    torch.tensor(0.0, device=p_fusion.device),
                    p_fusion
                )
                
                loss = criterion(p_fusion_gated, exp_b, logits, y_b, mask.to(DEVICE))
                val_loss += loss.item() * x_b.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(router_model.state_dict(), pth_router_path)
            saved_mark = "⭐ 最优模型保存"
        else:
            saved_mark = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {saved_mark}")

    # 全流域推演与导出
    print("\n" + "=" * 75)
    print("🌐 正在执行全网格成对排序 MOE 推理与导出...")
    full_ds = MOEDataset(router_inputs, expert_preds, station_target)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=False)

    router_model.load_state_dict(torch.load(pth_router_path, map_location=DEVICE))
    router_model.eval()

    p_fusion_list = []
    w1_list, w2_list, w3_list = [], [], []

    with torch.no_grad():
        for x_b, exp_b, _ in full_loader:
            x_b, exp_b = x_b.to(DEVICE), exp_b.to(DEVICE)
            weights, _ = router_model(x_b, exp_b)
            p_raw = torch.sum(weights * exp_b, dim=1, keepdim=True)
            
            asset1_p = exp_b[:, 0:1, :, :]
            p_final = torch.where(
                (asset1_p < DRIZZLE_THRESHOLD) | (p_raw < DRIZZLE_THRESHOLD),
                torch.tensor(0.0, device=p_raw.device),
                p_raw
            )
            
            p_fusion_list.append(p_final.cpu().numpy())
            w1_list.append(weights[:, 0:1, :, :].cpu().numpy())
            w2_list.append(weights[:, 1:2, :, :].cpu().numpy())
            w3_list.append(weights[:, 2:3, :, :].cpu().numpy())

    p_fusion_arr = np.concatenate(p_fusion_list, axis=0)[:, 0, :, :]
    w1_arr = np.concatenate(w1_list, axis=0)[:, 0, :, :]
    w2_arr = np.concatenate(w2_list, axis=0)[:, 0, :, :]
    w3_arr = np.concatenate(w3_list, axis=0)[:, 0, :, :]

    # 复盘
    val_indices = np.where(val_mask_idx)[0]
    metrics_list = []
    
    log_lines = []
    log_lines.append("==================================================================================")
    log_lines.append("🏆【成对排序 Top-2 优化 MOE】2020-2024 独立盲测全域复盘")
    log_lines.append("==================================================================================")

    for st_name, (r_idx, c_idx) in station_coords.items():
        obs_st = station_target.numpy()[val_indices, 0, r_idx, c_idx]
        pred_st = p_fusion_arr[val_indices, r_idx, c_idx]
        
        rmse, r2, cc, pod, far, csi = calc_metrics(obs_st, pred_st, threshold=0.1)
        metrics_list.append([rmse, r2, cc, pod, far, csi])

        w1_avg = w1_arr[val_indices, r_idx, c_idx].mean() * 100
        w2_avg = w2_arr[val_indices, r_idx, c_idx].mean() * 100
        w3_avg = w3_arr[val_indices, r_idx, c_idx].mean() * 100

        log_lines.append(f"📍 盲测站点【 {st_name:<4} 】| RMSE: {rmse:5.2f} mm/d | R²: {r2:5.3f} | CC: {cc:5.3f} | POD: {pod:4.2f} | FAR: {far:4.2f} | CSI: {csi:4.2f}")
        log_lines.append(f"   └─► 专家平均权重占比: 资产1(微地形)={w1_avg:4.1f}% | 资产2(时空)={w2_avg:4.1f}% | 资产3(极值)={w3_avg:4.1f}%")

    avg_m = np.mean(metrics_list, axis=0)
    log_lines.append("----------------------------------------------------------------------------------")
    log_lines.append(f"🏆【成对排序版全局 4 站平均】| RMSE: {avg_m[0]:5.2f} mm/d | R²: {avg_m[1]:5.3f} | CC: {avg_m[2]:5.3f} | POD: {avg_m[3]:4.2f} | FAR: {avg_m[4]:4.2f} | CSI: {avg_m[5]:4.2f}")
    log_lines.append("==================================================================================")

    full_log_text = "\n".join(log_lines)
    print("\n" + full_log_text)
    
    with open(txt_result_path, "w", encoding="utf-8") as f:
        f.write(full_log_text + "\n\nBest Val Loss: " + f"{best_val_loss:.4f}\n")

    # 物理裁剪与 NetCDF 导出
    print("\n✂️ 正在将 MOE 降水场与专家权重图裁剪至【北江不规则流域边界】...")
    p_fusion_arr[:, ~basin_mask_2d] = np.nan
    w1_arr[:, ~basin_mask_2d] = np.nan
    w2_arr[:, ~basin_mask_2d] = np.nan
    w3_arr[:, ~basin_mask_2d] = np.nan

    ds_moe = xr.Dataset(
        data_vars={
            'moe_precipitation': (['time', 'lat', 'lon'], p_fusion_arr),
            'weight_asset1_topography': (['time', 'lat', 'lon'], w1_arr),
            'weight_asset2_spatiotemporal': (['time', 'lat', 'lon'], w2_arr),
            'weight_asset3_extreme': (['time', 'lat', 'lon'], w3_arr)
        },
        coords={
            'time': ds_gpm.time,
            'lat': ds_gpm.lat,
            'lon': ds_gpm.lon
        },
        attrs={
            'description': 'Pairwise Top-2 Ranking Optimized MOE Fusion System',
            'spatial_mask': 'Clipped to irregular watershed boundary using DEM mask'
        }
    )
    
    ds_moe.to_netcdf(nc_moe_fusion_out)
    
    print("=" * 75)
    print(f"🎉 🎉 🎉 成对排序 Top-2 优化 MOE 完工！已保存至:\n  📦 {nc_moe_fusion_out}")
    print("=" * 75)

if __name__ == "__main__":
    main()
