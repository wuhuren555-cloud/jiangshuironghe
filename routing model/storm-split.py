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
STORM_RAIN_THRESHOLD = 25.0  # 国家气象标准：日降水量 >= 25 mm/d 划分为暴雨/大暴雨情景

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

pth_router_path     = os.path.join(router_dir, "best_hierarchical_router_model.pth")
txt_result_path     = os.path.join(router_dir, "结果_分层暴雨识别.txt")
nc_moe_fusion_out   = os.path.join(base_dir, "wanggeshuju", "quanliuyu_hierarchical_moe_2012_2024.nc")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ 计算设备: {DEVICE}")

# =========================================================================
# 🏗️ 3. 分层式 MoE 路由器 (Hierarchical Storm-Decoupled Router)
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

class HierarchicalStormDecoupledRouter(nn.Module):
    """【分层暴雨识别解耦 MoE 路由器】：
    1. 共享多尺度卷积与 CBAM 双重注意力主干；
    2. Head 1 (暴雨识别头)：输出暴雨发生概率 p_storm，采用 STE 直通估计器实现离散硬分流；
    3. Head 2 (常态双专家协同头)：在非暴雨情景下专职分配微地形(A1)与时空主干(A2)的权重 (w1+w2=1.0)；
    4. 终极合成：暴雨日全权调用 Asset 3，常态日由 Asset 1 & 2 协同，处处绝对水量守恒。
    """
    def __init__(self, in_channels=33, num_experts=3):
        super().__init__()
        total_in_channels = in_channels + num_experts + 3 # 33 + 3 + 3 = 39

        self.stem = nn.Sequential(
            nn.Conv2d(total_in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

        # 分支 1：权威气象暴雨情景识别头 (Storm Classifier)
        self.head_storm = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # 分支 2：常态微地形与时空主干协同分配头 (Asset 1 vs Asset 2)
        self.head_normal = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 2, kernel_size=1)
        )

        self.gamma_topo = nn.Parameter(torch.tensor(1.2))

    def forward(self, x, exp_b):
        p1 = exp_b[:, 0:1, :, :]
        p2 = exp_b[:, 1:2, :, :]
        p3 = exp_b[:, 2:3, :, :]

        p_base = torch.max(p1, p2)
        diff_12 = F.relu(p1 - p2)
        diff_32 = F.relu(p3 - p2)

        feat_in = torch.cat([x, exp_b, p_base, diff_12, diff_32], dim=1) # (B, 39, H, W)
        feat = self.stem(feat_in)
        feat = self.ca(feat)
        feat = self.sa(feat)

        # 🌟 阶段 1：暴雨情景识别分类
        storm_logits = self.head_storm(feat)
        p_storm = torch.sigmoid(storm_logits)

        # 直通估计器 (Straight-Through Estimator, STE): 前向离散 0/1 切分，反向连续传递梯度
        storm_hard = (p_storm >= 0.5).float()
        g_storm = storm_hard - p_storm.detach() + p_storm

        # 🌟 阶段 2：常态微地形与时空主干双模型分配 (w1 + w2 = 1.0)
        normal_logits = self.head_normal(feat)
        dem_prior = torch.sigmoid(x[:, 0:1, :, :])
        normal_logits[:, 0:1, :, :] = normal_logits[:, 0:1, :, :] + self.gamma_topo * dem_prior
        w_normal = F.softmax(normal_logits, dim=1)
        w1_norm = w_normal[:, 0:1, :, :]
        w2_norm = w_normal[:, 1:2, :, :]

        # 🌟 阶段 3：严格水量守恒的分层解耦权重合成
        w1_eff = (1.0 - g_storm) * w1_norm
        w2_eff = (1.0 - g_storm) * w2_norm
        w3_eff = g_storm

        # 注入无雨物理干湿门控
        wet_mask = (p1 >= DRIZZLE_THRESHOLD).float()
        w1_eff = w1_eff * wet_mask
        w2_eff = w2_eff * wet_mask
        w3_eff = w3_eff * wet_mask

        weights = torch.cat([w1_eff, w2_eff, w3_eff], dim=1)
        p_fusion = w1_eff * p1 + w2_eff * p2 + w3_eff * p3

        return weights, p_fusion, storm_logits, normal_logits

# =========================================================================
# 🌟 4. 联合损失函数：Focal Loss + Huber Loss + 常态协同对齐
# =========================================================================
class HierarchicalMoELoss(nn.Module):
    def __init__(self, delta=3.0, alpha=0.75, gamma=2.0, cls_w=1.5, norm_w=0.8):
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.cls_w = cls_w
        self.norm_w = norm_w

    def forward(self, p_fusion, storm_logits, normal_logits, expert_preds, target, station_mask):
        valid_count = torch.sum(station_mask)
        if valid_count == 0:
            return torch.tensor(0.0, device=p_fusion.device, requires_grad=True)

        w_sample = 1.0 + (target / 10.0)

        # 1. 连续降水 Huber 回归损失
        err = target - p_fusion
        abs_err = torch.abs(err)
        huber_loss = torch.where(
            abs_err <= self.delta,
            0.5 * (err ** 2),
            self.delta * abs_err - 0.5 * (self.delta ** 2)
        )
        loss_huber = torch.sum(huber_loss * w_sample * station_mask) / (torch.sum(station_mask * w_sample) + 1e-8)

        # 2. 权威暴雨情景识别焦点损失 (Class-Balanced Focal Loss for Storm Detection)
        storm_label = (target >= STORM_RAIN_THRESHOLD).float()
        p_storm = torch.sigmoid(storm_logits)
        pt = torch.where(storm_label == 1.0, p_storm, 1.0 - p_storm).clamp(1e-6, 1.0 - 1e-6)
        alpha_t = torch.where(storm_label == 1.0, self.alpha, 1.0 - self.alpha)
        focal_loss = -alpha_t * ((1.0 - pt) ** self.gamma) * torch.log(pt)
        loss_cls = torch.sum(focal_loss * station_mask) / (valid_count + 1e-8)

        # 3. 非暴雨场景常态双专家最优选择交叉熵监督
        e1 = torch.abs(expert_preds[:, 0:1, :, :] - target)
        e2 = torch.abs(expert_preds[:, 1:2, :, :] - target)
        normal_winner = (e2 < e1).long().squeeze(1)
        ce_norm = F.cross_entropy(normal_logits, normal_winner, reduction='none').unsqueeze(1)
        non_storm_mask = station_mask * (1.0 - storm_label)
        loss_norm = torch.sum(ce_norm * w_sample * non_storm_mask) / (torch.sum(non_storm_mask * w_sample) + 1e-8)

        return loss_huber + self.cls_w * loss_cls + self.norm_w * loss_norm

# =========================================================================
# 📊 5. 全套学术评价指标计算函数
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

    alpha = std_p / (std_o + 1e-8)
    beta = mean_p / (mean_o + 1e-8)
    kge = 1.0 - np.sqrt((cc - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

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
    print("=" * 88)
    print("🔥 开始训练【分层暴雨情景识别与极值解耦 MoE 融合系统】...")

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

    # 严格三集划分 (训练集: 2012–2019, 验证集: 2020–2021, 测试集: 2022–2024)
    train_mask_idx = (gpm_dates_str >= '2012-01-01') & (gpm_dates_str <= '2019-12-31')
    val_mask_idx   = (gpm_dates_str >= '2020-01-01') & (gpm_dates_str <= '2021-12-31')
    test_mask_idx  = (gpm_dates_str >= '2022-01-01') & (gpm_dates_str <= '2024-12-31')

    train_ds = MOEDataset(router_inputs[train_mask_idx], expert_preds[train_mask_idx], station_target[train_mask_idx])
    val_ds   = MOEDataset(router_inputs[val_mask_idx],   expert_preds[val_mask_idx],   station_target[val_mask_idx])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)

    router_model = HierarchicalStormDecoupledRouter(in_channels=33, num_experts=3).to(DEVICE)
    optimizer = torch.optim.AdamW(router_model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = HierarchicalMoELoss(delta=3.0, alpha=0.75, gamma=2.0, cls_w=1.5, norm_w=0.8)

    best_val_loss = float('inf')
    epochs = 40

    print("\n⚡ 开始分层暴雨分类与常态解耦协同训练 (40 Epochs)...")
    for epoch in range(1, epochs + 1):
        router_model.train()
        train_loss = 0.0
        for x_b, exp_b, y_b in train_loader:
            x_b, exp_b, y_b = x_b.to(DEVICE), exp_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            
            weights, p_fusion, storm_logits, normal_logits = router_model(x_b, exp_b)
            loss = criterion(p_fusion, storm_logits, normal_logits, exp_b, y_b, mask.to(DEVICE))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_b.size(0)
        train_loss /= len(train_ds)

        router_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, exp_b, y_b in val_loader:
                x_b, exp_b, y_b = x_b.to(DEVICE), exp_b.to(DEVICE), y_b.to(DEVICE)
                weights, p_fusion, storm_logits, normal_logits = router_model(x_b, exp_b)
                loss = criterion(p_fusion, storm_logits, normal_logits, exp_b, y_b, mask.to(DEVICE))
                val_loss += loss.item() * x_b.size(0)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(router_model.state_dict(), pth_router_path)
            saved_mark = "⭐ 最优模型保存"
        else:
            saved_mark = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {saved_mark}")

    # =========================================================================
    # 🌐 7. 全网格推演与导出 (严格绝对水量守恒)
    # =========================================================================
    print("\n" + "=" * 88)
    print("🌐 正在执行全网格绝对水量守恒 Hierarchical MoE 推理与导出...")
    full_ds = MOEDataset(router_inputs, expert_preds, station_target)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=False)

    router_model.load_state_dict(torch.load(pth_router_path, map_location=DEVICE))
    router_model.eval()

    p_fusion_list = []
    w1_list, w2_list, w3_list = [], [], []

    with torch.no_grad():
        for x_b, exp_b, _ in full_loader:
            x_b, exp_b = x_b.to(DEVICE), exp_b.to(DEVICE)
            weights, p_final, _, _ = router_model(x_b, exp_b)
            
            p_fusion_list.append(p_final.cpu().numpy())
            w1_list.append(weights[:, 0:1, :, :].cpu().numpy())
            w2_list.append(weights[:, 1:2, :, :].cpu().numpy())
            w3_list.append(weights[:, 2:3, :, :].cpu().numpy())

    p_fusion_arr = np.concatenate(p_fusion_list, axis=0)[:, 0, :, :]
    w1_arr = np.concatenate(w1_list, axis=0)[:, 0, :, :]
    w2_arr = np.concatenate(w2_list, axis=0)[:, 0, :, :]
    w3_arr = np.concatenate(w3_list, axis=0)[:, 0, :, :]

    # =========================================================================
    # 📊 8. 打印控制台两张标准学术指标表
    # =========================================================================
    train_indices = np.where(train_mask_idx)[0]
    val_indices   = np.where(val_mask_idx)[0]
    test_indices  = np.where(test_mask_idx)[0]

    def extract_split_pairs(split_indices):
        obs_all, pred_all = [], []
        for st_name, (r_idx, c_idx) in station_coords.items():
            obs_st = station_target.numpy()[split_indices, 0, r_idx, c_idx]
            pred_st = p_fusion_arr[split_indices, r_idx, c_idx]
            obs_all.append(obs_st)
            pred_all.append(pred_st)
        return np.concatenate(obs_all), np.concatenate(pred_all)

    train_obs, train_pred = extract_split_pairs(train_indices)
    val_obs,   val_pred   = extract_split_pairs(val_indices)
    test_obs,  test_pred  = extract_split_pairs(test_indices)

    metrics_train = calc_all_metrics(train_obs, train_pred, threshold=0.1)
    metrics_val   = calc_all_metrics(val_obs,   val_pred,   threshold=0.1)
    metrics_test  = calc_all_metrics(test_obs,  test_pred,  threshold=0.1)

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

    # 表 1：三集全景对比表
    table1_lines = [
        "\n" + "=" * 88,
        "评估指标       | 单位         | 训练集 (2012–2019)   | 验证集 (2020–2021)   | 测试集 (2022–2024)   ",
        "-" * 88
    ]
    for m in metric_names:
        disp_m = metric_display_names[m]
        unit   = metric_units[m]
        v_tr   = metrics_train[m]
        v_va   = metrics_val[m]
        v_te   = metrics_test[m]
        table1_lines.append(f"{disp_m:<14} | {unit:<12} | {v_tr:<20.3f} | {v_va:<20.3f} | {v_te:<20.3f}")
    table1_lines.append("=" * 88)
    table1_text = "\n".join(table1_lines)

    # 表 2：站点详情 (测试集 2022–2024)
    table2_lines = [
        "\n" + "=" * 108,
        "📍【站点详情 (测试集: 2022–2024)】",
        "=" * 108,
        f"{'站点':<8} | {'MSE(mm²/d²)':<12} | {'MAE(mm/d)':<10} | {'RMSE(mm/d)':<11} | {'R²':<8} | {'NSE':<8} | {'KGE':<8} | {'POD':<7} | {'FAR':<7} | {'CSI':<7}",
        "-" * 108
    ]
    for st_name, (r_idx, c_idx) in station_coords.items():
        obs_st  = station_target.numpy()[test_indices, 0, r_idx, c_idx]
        pred_st = p_fusion_arr[test_indices, r_idx, c_idx]
        m_st = calc_all_metrics(obs_st, pred_st, threshold=0.1)
        table2_lines.append(
            f"{st_name:<8} | {m_st['MSE']:<12.2f} | {m_st['MAE']:<10.2f} | {m_st['RMSE']:<11.2f} | "
            f"{m_st['R2']:<8.3f} | {m_st['NSE']:<8.3f} | {m_st['KGE']:<8.3f} | "
            f"{m_st['POD']:<7.3f} | {m_st['FAR']:<7.3f} | {m_st['CSI']:<7.3f}"
        )
    table2_lines.append("=" * 108)
    table2_text = "\n".join(table2_lines)

    print(table1_text)
    print(table2_text)

    # 结果写入本地 txt
    full_output_log = table1_text + "\n" + table2_text + f"\n\nBest Val Loss: {best_val_loss:.4f}\n"
    with open(txt_result_path, "w", encoding="utf-8") as f:
        f.write(full_output_log)
    print(f"\n📝 详细评估结果已保存至: {txt_result_path}")

    # =========================================================================
    # 💾 9. 全流域不规则边界掩膜与 NetCDF 导出
    # =========================================================================
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
            'description': 'Hierarchical Storm-Decoupled Strictly Water-Conserved MOE Fusion System',
            'spatial_mask': 'Clipped to irregular watershed boundary using DEM mask'
        }
    )
    ds_moe.to_netcdf(nc_moe_fusion_out)
    print(f"🎉 全流域分层解耦融合降水 NetCDF 导出完成: {nc_moe_fusion_out}")

if __name__ == "__main__":
    main()
