import os
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray

# =====================================================================
# 1. 基础路径配置（对应未裁剪 TIF 与全矩形输出）
# =====================================================================
base_dir = r"C:\Users\26332\OneDrive\Desktop\sun mission\jiangshuironghe-MOE"

# 调优后的两阶段模型路径
cls_model_path = os.path.join(base_dir, "wanggeshuju", "dixingchayi-tiaocan_cls_model.pkl")
reg_model_path = os.path.join(base_dir, "wanggeshuju", "dixingchayi-tiaocan_reg_model.pkl")

# GPM 长方形原始网格数据
nc_path = os.path.join(base_dir, "wanggeshuju", "juhe_gpm_daily_2012_2024.nc")

# 🌟【修改 1】替换为过程文件夹下的【未裁剪全矩形】地形 TIF
# 若 DEM 使用 quzheng.tif，请修改文件名（通常 tianwa-quzheng.tif 或 quzheng.tif 均可）
dem_tif = os.path.join(base_dir, "边界文件两幅", "过程文件", "touying.tif")
slope_tif = os.path.join(base_dir, "边界文件两幅", "过程文件", "podu.tif")
aspect_tif = os.path.join(base_dir, "边界文件两幅", "过程文件", "poxiang.tif")

# 终极全矩形融合降水资产 1 导出路径
out_nc = os.path.join(base_dir, "wanggeshuju", "juxing_asset1_2012_2024.nc")

# =====================================================================
# 2. 加载模型资产与 GPM 数据
# =====================================================================
print("正在加载调优后的【资产 1 地形子模型双核权重】...")
cls_model = joblib.load(cls_model_path)
reg_model = joblib.load(reg_model_path)

print("正在载入 GPM 卫星降水网格，截断时间轴至 2012-01-01 ~ 2024-12-31...")
ds_gpm = xr.open_dataset(nc_path, engine="netcdf4")
ds_gpm = ds_gpm.sel(time=slice("2012-01-01", "2024-12-31"))

# 维度规范化为 ('time', 'lat', 'lon')
ds_gpm['precipitation'] = ds_gpm['precipitation'].transpose('time', 'lat', 'lon')

# 创建坐标模板，供全图地形矩阵重采样对齐
match_template = ds_gpm['precipitation'].rename({"lon": "x", "lat": "y"})
match_template = match_template.rio.write_crs("EPSG:4326")

# =====================================================================
# 3. 对齐重采样全矩形地形栅格至 GPM 网格空间分辨率
# =====================================================================
print("正在将【全矩形】微地形栅格 (DEM/坡度/坡向) 重采样对齐至卫星空间网格...")
da_dem = rioxarray.open_rasterio(dem_tif).rio.reproject_match(match_template)
da_slope = rioxarray.open_rasterio(slope_tif).rio.reproject_match(match_template)
da_aspect = rioxarray.open_rasterio(aspect_tif).rio.reproject_match(match_template)

dem_arr = da_dem.values[0]
slope_arr = da_slope.values[0]
aspect_arr = da_aspect.values[0]

# 填充可能存在极少边缘 TIF 缺省值，保证全矩形矩阵计算连续
dem_arr = np.nan_to_num(dem_arr, nan=np.nanmedian(dem_arr))
slope_arr = np.nan_to_num(slope_arr, nan=0.0)
aspect_arr = np.nan_to_num(aspect_arr, nan=0.0)

times = ds_gpm.time.values
lats = ds_gpm.lat.values
lons = ds_gpm.lon.values

# 准备 3D 全矩形预测输出矩阵 (Time x Lat x Lon)
fusion_rain_array = np.full((len(times), len(lats), len(lons)), np.nan, dtype=np.float32)

# 构建经纬度坐标网格 (25, 30)
lon_grid, lat_grid = np.meshgrid(lons, lats)

feature_cols = ['gpm_rain', 'DEM', 'Slope', 'Aspect', 'lat', 'lon', 'Month']

print(f"\n🚀 开始【全矩形框】逐像素时空外推预测 (覆盖 {len(times)} 天 x {len(lats)}x{len(lons)} 全域格点)...")

# =====================================================================
# 4. 逐日推理预测（全矩形推求）
# =====================================================================
for t_idx, t_val in enumerate(times):
    month = pd.to_datetime(t_val).month
    gpm_day = ds_gpm['precipitation'].values[t_idx]
    
    # 🌟【修改 2】放宽掩膜条件：只要 GPM 数据有效，全矩形网格全部参与 AI 推断演算
    valid_mask = ~np.isnan(gpm_day)
    
    if not np.any(valid_mask):
        continue
        
    df_pixel = pd.DataFrame({
        'gpm_rain': gpm_day[valid_mask],
        'DEM': dem_arr[valid_mask],
        'Slope': slope_arr[valid_mask],
        'Aspect': aspect_arr[valid_mask],
        'lat': lat_grid[valid_mask],
        'lon': lon_grid[valid_mask],
        'Month': month
    })
    
    # 两阶段预测：分类 + 回归
    wet_pred = cls_model.predict(df_pixel[feature_cols])
    rain_pred = reg_model.predict(df_pixel[feature_cols])
    
    # 逻辑截断：无雨强行赋 0.0，有雨则为回归雨量
    final_rain = np.where(wet_pred == 1, rain_pred, 0.0)
    final_rain = np.clip(final_rain, 0, None) # 强制非负
    
    # 写回 3D 全矩形矩阵
    fusion_rain_array[t_idx][valid_mask] = final_rain
    
    if (t_idx + 1) % 500 == 0 or (t_idx + 1) == len(times):
        print(f"  [进度] 已完成 {t_idx + 1} / {len(times)} 天 ({((t_idx + 1)/len(times))*100:.1f}%) 的全矩形演算...")

# =====================================================================
# 5. 封装导出为标准 NetCDF 资产 1 (全矩形未裁剪)
# =====================================================================
print("\n正在封装全矩形 NetCDF 数据立方体...")
ds_asset1 = xr.Dataset(
    data_vars=dict(
        fusion_precipitation=(["time", "lat", "lon"], fusion_rain_array)
    ),
    coords=dict(
        time=times,
        lat=lats,
        lon=lons,
    ),
    attrs=dict(
        description="北江中上游基于两阶段全局XGBoost与微地形矫正的融合降水资产1 (全矩形框未裁剪，供 MoE 门控网络训练使用)",
        spatial_resolution="0.1deg x 0.1deg",
        time_coverage="2012-01-01 to 2024-12-31",
        units="mm/d"
    )
)

ds_asset1.to_netcdf(out_nc)

print("=" * 60)
print(f"🎉 🎉 🎉 全矩形资产 1 构建完工！已成型导出！")
print(f"📦 NC 文件路径: {out_nc}")
print("=" * 60)
