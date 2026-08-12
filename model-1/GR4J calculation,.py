import os
import xarray as xr
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

base_dir = r"C:\Users\26332\OneDrive\Desktop\sun mission\jiangshuironghe-MOE"
flow_csv = os.path.join(base_dir, "shicezhandianshuju", "feilaixia_daily_flow_2017_2021.csv")

asset1_nc = os.path.join(base_dir, "wanggeshuju", "quanliuyu_asset1_fusion_2012_2024.nc")
gpm_nc = os.path.join(base_dir, "wanggeshuju", "juhe_gpm_daily_2012_2024.nc")

# 1. 读取流量与 3D 降水数据
df_flow = pd.read_csv(flow_csv)
df_flow['Date'] = pd.to_datetime(df_flow['Date'])

ds_a1 = xr.open_dataset(asset1_nc).sel(time=slice("2017-03-08", "2021-12-31"))
ds_gpm = xr.open_dataset(gpm_nc, engine="netcdf4").sel(time=slice("2017-03-08", "2021-12-31"))

da_a1 = ds_a1['fusion_precipitation'].transpose('time', 'lat', 'lon')
da_gpm = ds_gpm['precipitation'].transpose('time', 'lat', 'lon')

P_3d_a1 = da_a1.values
P_3d_gpm = da_gpm.values

dates = pd.to_datetime(da_a1.time.values)
months = dates.month.values
lats, lons = da_a1.lat.values, da_a1.lon.values

# 2. 空间水力距离矩阵 (出口: 飞来峡 23.80 N, 113.25 E)
outlet_lat, outlet_lon = 23.80, 113.25
lon_grid, lat_grid = np.meshgrid(lons, lats)
dist_matrix = np.sqrt(((lat_grid - outlet_lat) * 111)**2 + ((lon_grid - outlet_lon) * 110 * np.cos(np.radians(outlet_lat)))**2)

# 3. 月动态潜在蒸散发序列 (根据 Oudin et al., 2005 设定)
EM_monthly_base = np.array([1.5, 1.8, 2.2, 2.8, 3.5, 4.0, 4.5, 4.2, 3.5, 2.8, 2.0, 1.5])
E_series = EM_monthly_base[months - 1]

# 4. 时段划分
calib_mask = (dates >= "2017-06-01") & (dates <= "2019-12-31")
valid_mask = (dates >= "2020-01-01") & (dates <= "2021-12-31")

df_eval = pd.DataFrame({'Date': dates}).merge(df_flow, on='Date', how='left')
Q_obs = df_eval['Q_obs'].values

# =====================================================================
# 5. 【半分布式 GR4J (SD-GR4J) 核心模型】
# =====================================================================
def sd_gr4j_model(P_3d, E_seq, params, Area=34000.0):
    X1, X2, X3, X4 = params
    T, N_lat, N_lon = P_3d.shape
    
    valid_grid = ~np.isnan(P_3d[0]) & (P_3d[0] >= 0)
    num_grids = np.sum(valid_grid)
    if num_grids == 0: return np.zeros(T)
    
    grid_area_km2 = Area / num_grids
    C_flow = grid_area_km2 * 1000.0 / 86400.0 # mm/d -> m³/s
    
    v_speed = 120.0 # km/day
    delay_days = np.round(dist_matrix / v_speed).astype(int)
    
    S = np.full((N_lat, N_lon), X1 * 0.6)
    R = np.full((N_lat, N_lon), X3 * 0.5)
    
    n_uh = int(np.ceil(2 * X4)) + 1
    uh1, uh2 = np.zeros(n_uh), np.zeros(n_uh)
    for i in range(1, n_uh + 1):
        t1, t0 = i, i - 1
        f1 = (t1 / X4)**2.5 if t1 <= X4 else 1.0
        f0 = (t0 / X4)**2.5 if t0 <= X4 else 1.0
        uh1[i-1] = f1 - f0
        
        g1 = 0.5 * (t1 / X4)**2.5 if t1 <= X4 else (1.0 - 0.5 * (2 - t1 / X4)**2.5 if t1 <= 2 * X4 else 1.0)
        g0 = 0.5 * (t0 / X4)**2.5 if t0 <= X4 else (1.0 - 0.5 * (2 - t0 / X4)**2.5 if t0 <= 2 * X4 else 1.0)
        uh2[i-1] = g1 - g0

    q9_buffer = np.zeros((T + n_uh * 2, N_lat, N_lon))
    q1_buffer = np.zeros((T + n_uh * 2, N_lat, N_lon))
    I_outlet = np.zeros(T)

    for t in range(T):
        p_t = np.where(valid_grid, P_3d[t], 0.0)
        e_t = E_seq[t]
        
        pn = np.maximum(0.0, p_t - e_t)
        en = np.maximum(0.0, e_t - p_t)
        
        s_ratio = S / X1
        ps = np.where(p_t >= e_t, X1 * (1 - s_ratio**2) * np.tanh(pn / X1) / (1 + s_ratio * np.tanh(pn / X1)), 0.0)
        es = np.where(p_t < e_t, S * (2 - s_ratio) * np.tanh(en / X1) / (1 + (1 - s_ratio) * np.tanh(en / X1)), 0.0)
        
        S = S - es + ps
        s_ratio_new = S / X1
        perc = S * (1 - (1 + (4/9 * s_ratio_new)**4)**(-0.25))
        S = S - perc
        pr = perc + (pn - ps)
        
        for k in range(n_uh):
            q9_buffer[t + k] += pr * 0.9 * uh1[k]
            q1_buffer[t + k] += pr * 0.1 * uh2[k]
            
        q9, q1 = q9_buffer[t], q1_buffer[t]
        r_ratio = R / X3
        f_exchange = X2 * r_ratio**(3.5)
        
        R_star = np.maximum(0.0, R + q9 + f_exchange)
        qr = R_star * (1 - (1 + (R_star / X3)**4)**(-0.25))
        R = R_star - qr
        qd = np.maximum(0.0, q1 + f_exchange)
        
        q_cell = (qr + qd) * C_flow
        
        for i in range(N_lat):
            for j in range(N_lon):
                if valid_grid[i, j]:
                    tau = delay_days[i, j]
                    if t + tau < T:
                        I_outlet[t + tau] += q_cell[i, j]

    return I_outlet

# =====================================================================
# 6. 【多维评估指标计算函数】
# =====================================================================
def calc_metrics(obs, sim, mask):
    o, s = obs[mask], sim[mask]
    valid = ~np.isnan(o) & ~np.isnan(s)
    o_v, s_v = o[valid], s[valid]
    
    if len(o_v) == 0:
        return -999.0, -999.0, -999.0
        
    # NSE
    nse = 1.0 - np.sum((o_v - s_v)**2) / np.sum((o_v - np.mean(o_v))**2)
    
    # KGE
    r = np.corrcoef(o_v, s_v)[0, 1]
    alpha = np.std(s_v) / np.std(o_v)
    beta = np.mean(s_v) / np.mean(o_v)
    kge = 1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)
    
    # PBIAS (%)
    pbias = np.sum(s_v - o_v) / np.sum(o_v) * 100.0
    
    return nse, kge, pbias

# =====================================================================
# 7. 【全局进化寻优与参数提取】
# =====================================================================
bounds = [
    (100.0, 1500.0), # X1
    (-2.0, 15.0),    # X2
    (10.0, 500.0),   # X3
    (0.5, 4.0)       # X4
]

print("==================================================")
print("🚀 正在率定【资产 1 融合降水】SD-GR4J 水文参数...")
res_a1 = differential_evolution(lambda p: -calc_metrics(Q_obs, sd_gr4j_model(P_3d_a1, E_series, p), calib_mask)[0], bounds, seed=42, maxiter=30, popsize=12)
p_a1 = res_a1.x

print("🚀 正在率定【原始 GPM 降水】SD-GR4J 水文参数...")
res_gpm = differential_evolution(lambda p: -calc_metrics(Q_obs, sd_gr4j_model(P_3d_gpm, E_series, p), calib_mask)[0], bounds, seed=42, maxiter=30, popsize=12)
p_gpm = res_gpm.x

# 演算全序列
Q_sim_a1 = sd_gr4j_model(P_3d_a1, E_series, p_a1)
Q_sim_gpm = sd_gr4j_model(P_3d_gpm, E_series, p_gpm)

# 计算指标
nse_a1_c, kge_a1_c, pbias_a1_c = calc_metrics(Q_obs, Q_sim_a1, calib_mask)
nse_a1_v, kge_a1_v, pbias_a1_v = calc_metrics(Q_obs, Q_sim_a1, valid_mask)

nse_gpm_c, kge_gpm_c, pbias_gpm_c = calc_metrics(Q_obs, Q_sim_gpm, calib_mask)
nse_gpm_v, kge_gpm_v, pbias_gpm_v = calc_metrics(Q_obs, Q_sim_gpm, valid_mask)

# =====================================================================
# 8. 终端控制台直接输出控制表格
# =====================================================================
print("\n" + "=" * 65)
print("📌 表 1：SD-GR4J 最佳率定水文参数对照表 (Calibrated Parameters)")
print("=" * 65)
print(f"{'参数 (Parameter)':<28} | {'原始 GPM 驱动':<12} | {'资产 1 融合驱动':<12}")
print("-" * 65)
print(f"{'X1: 产水容量 (Production, mm)':<28} | {p_gpm[0]:<12.2f} | {p_a1[0]:<12.2f}")
print(f"{'X2: 地下水交换 (Exchange, mm/d)':<28} | {p_gpm[1]:<12.2f} | {p_a1[1]:<12.2f}")
print(f"{'X3: 汇水容量 (Routing, mm)':<28} | {p_gpm[2]:<12.2f} | {p_a1[2]:<12.2f}")
print(f"{'X4: 单位线汇流时间 (Time, d)':<28} | {p_gpm[3]:<12.2f} | {p_a1[3]:<12.2f}")
print("=" * 65)

print("\n" + "=" * 65)
print("📌 表 2：双时段多维水文验证效能评估表 (Hydrological Metrics)")
print("=" * 65)
print(f"{'评估指标 (Metric)':<18} | {'率定期 (2017.06-2019.12)':<20} | {'验证期 (2020.01-2021.12)':<20}")
print(f"{'':<18} | {'GPM':<8} {'资产1':<10} | {'GPM':<8} {'资产1':<10}")
print("-" * 65)
print(f"{'NSE (纳什效率)':<18} | {nse_gpm_c:<8.3f} {nse_a1_c:<10.3f} | {nse_gpm_v:<8.3f} {nse_a1_v:<10.3f}")
print(f"{'KGE (Kling-Gupta)':<18} | {kge_gpm_c:<8.3f} {kge_a1_c:<10.3f} | {kge_gpm_v:<8.3f} {kge_a1_v:<10.3f}")
print(f"{'PBIAS (水量偏差 %)':<18} | {pbias_gpm_c:<+8.1f}% {pbias_a1_c:<+10.1f}% | {pbias_gpm_v:<+8.1f}% {pbias_a1_v:<+10.1f}%")
print("=" * 65 + "\n")

# 9. 绘图
fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
ax.plot(dates, Q_obs, 'k.', label='实测流量', alpha=0.4, markersize=3)
ax.plot(dates, Q_sim_gpm, 'r--', label=f'GPM (率定NSE={nse_gpm_c:.2f}, 验证NSE={nse_gpm_v:.2f})', linewidth=1.2)
ax.plot(dates, Q_sim_a1, 'b-', label=f'资产 1 (率定NSE={nse_a1_c:.2f}, 验证NSE={nse_a1_v:.2f})', linewidth=1.5)

ax.axvline(pd.to_datetime("2017-06-01"), color='gray', linestyle=':', linewidth=1.5)
ax.axvline(pd.to_datetime("2020-01-01"), color='green', linestyle='--', linewidth=1.5)

ax.set_ylabel("流量 ($m^3/s$)")
ax.set_title("北江飞来峡出口半分布式 GR4J 水文驱动对比", fontsize=12, fontweight='bold')
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='upper right')

plt.tight_layout()
plot_path = os.path.join(base_dir, "shicezhandianshuju", "sd_gr4j_final_result.png")
plt.savefig(plot_path)
plt.show()
