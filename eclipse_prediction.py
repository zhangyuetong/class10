"""
日食月食预测模块

此模块提供日食和月食事件的预测功能，基于天体三维位置数据。
"""

import numpy as np
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm

from visualize.solar_plot import plot_eclipse_geometry_3d

"""
日食月食的几何推导

1. 日食原理与精确判定条件

   日食发生的本质是月球遮挡太阳光线射向地球的现象。根据几何光学原理，我们可以进行如下推导：
   ...（省略详细推导）
   
2. 月食原理
   太阳-地球-月球三者接近一条直线，且地球位于太阳和月球之间时，
   月球进入地球的阴影区域，形成月食。
   
   月食判定条件：
   a) 太阳-地球-月球三点夹角接近180度
   b) 月球进入地球的本影或半影区域

3. 计算方法
   - 判断三点是否接近共线：计算向量夹角
   - 计算阴影锥体：基于天体半径和距离
   - 检查是否落入阴影区域：几何关系计算
"""

def body_in_cone(body_pos, body_radius, cone_tip, cone_axis, half_angle):
    """
    判断天体是否在锥体内部
    """
    # 天体到锥顶的向量
    body_to_cone_tip = body_pos - cone_tip
    # 天体到锥轴的垂直投影点
    proj_point = cone_tip + np.dot(body_to_cone_tip, cone_axis) * cone_axis / (np.linalg.norm(cone_axis)**2)
    # 天体到锥轴的垂直距离（对边）
    perpendicular_distance = np.linalg.norm(body_pos - proj_point)
    # 天体到锥轴的水平距离（邻边）
    horizontal_distance = np.linalg.norm(cone_tip - proj_point)

    # 临界距离
    critical_distance = np.tan(half_angle) * horizontal_distance + body_radius / np.cos(half_angle)
    # 判断是否在锥体内
    if perpendicular_distance < critical_distance:
        return True
    return False

def body_totally_in_cone(body_pos, body_radius, cone_tip, cone_axis, half_angle):
    """
    判断天体是否在锥体内部
    """
    # 天体到锥顶的向量
    body_to_cone_tip = body_pos - cone_tip
    # 天体到锥轴的垂直投影点
    proj_point = cone_tip + np.dot(body_to_cone_tip, cone_axis) * cone_axis / (np.linalg.norm(cone_axis)**2)
    # 天体到锥轴的垂直距离（对边）
    perpendicular_distance = np.linalg.norm(body_pos - proj_point)
    # 天体到锥轴的水平距离（邻边）
    horizontal_distance = np.linalg.norm(cone_tip - proj_point)

    # 临界距离
    critical_distance = np.tan(half_angle) * horizontal_distance - body_radius / np.cos(half_angle)
    # 判断是否在锥体内
    if perpendicular_distance > critical_distance:
        return False
    return True  

def predict_eclipses(sol, t0, state_reshaped, sun_idx, earth_idx, moon_idx):
    """
    预测日食和月食事件
    
    参数:
    sol - 数值积分求解结果
    t0 - 初始时间
    state_reshaped - 重塑后的状态数组
    sun_idx - 太阳在状态数组中的索引
    earth_idx - 地球在状态数组中的索引
    moon_idx - 月球在状态数组中的索引
    
    返回:
    eclipse_events - 包含日食和月食事件的字典
    """
    # 初始化日食月食事件字典
    eclipse_events = {
        "solar": [],  # 日食事件列表
        "lunar": []   # 月食事件列表
    }
    
    # 创建进度条
    pbar = tqdm(total=len(sol.t), desc="分析日食月食")
    
    for i, t_sec in enumerate(sol.t):
        # 获取当前历元时间（TT时间系统）
        current_time = t0 + t_sec * u.s  # 从初始时间t0开始累加积分步长
        formatted_time = current_time.iso  # 格式化为ISO标准时间字符串
        
        # 从状态数组中提取天体位置（假设为J2000坐标系）
        # sun_idx, earth_idx, moon_idx 分别为太阳、地球、月球在数组中的索引
        sun_pos = state_reshaped[sun_idx, :3, i]    # 太阳的x,y,z坐标 (m)
        earth_pos = state_reshaped[earth_idx, :3, i] # 地球的坐标 (m)
        moon_pos = state_reshaped[moon_idx, :3, i]   # 月球的坐标 (m)
        
        # 计算相对位置向量（地心视角）
        earth_to_sun = sun_pos - earth_pos  # 从地球指向太阳的向量
        earth_to_moon = moon_pos - earth_pos # 从地球指向月球的向量
        moon_to_sun = sun_pos - moon_pos    # 从月球指向太阳的向量
        
        # 计算天体间欧氏距离（标量）
        earth_sun_dist = np.linalg.norm(earth_to_sun)  # 地日距离 ‖ES‖
        earth_moon_dist = np.linalg.norm(earth_to_moon) # 地月距离 ‖EM‖
        moon_sun_dist = np.linalg.norm(moon_to_sun)    # 月日距离 ‖MS‖
        
        # 归一化方向向量（用于角度计算）
        earth_to_sun_unit = earth_to_sun / earth_sun_dist  # 太阳方向单位向量 EŜ
        earth_to_moon_unit = earth_to_moon / earth_moon_dist # 月球方向单位向量 EM̂
        moon_to_sun_unit = moon_to_sun / moon_sun_dist     # 太阳在月心坐标中的方向 MŜ
        
        # 天体物理常数（单位：米）
        R_sun = 695700e3     # 太阳半径 (IAU 2015)
        R_moon = 1737.4e3    # 月球半径 (LRO激光测距)
        R_earth = 6371e3     # 地球赤道半径 (WGS84)
        
        # ===== 日食初步筛选条件 =====
        # 条件1：月球位于地球与太阳连线的同侧（点积>0）
        # 条件2：地月距离小于地日距离（确保月球在地球和太阳之间）
        if np.dot(earth_to_sun_unit, earth_to_moon_unit) > 0 and (earth_moon_dist < earth_sun_dist):
            # 计算地日-地月向量夹角（精确到1e-12弧度）
            # 使用np.clip防止浮点误差导致反余弦计算异常
            cos_theta = np.clip(np.dot(earth_to_sun_unit, earth_to_moon_unit), -1.0, 1.0)
            solar_eclipse_angle = np.arccos(cos_theta)  # 夹角θ = arccos(EŜ·EM̂)
            solar_eclipse_threshold = 2e-2  # 约1.146度（经验阈值）
            
            # 当夹角小于阈值时值得进入详细计算
            if solar_eclipse_angle < solar_eclipse_threshold:
                # 计算基本参数
                # 影锥轴线
                umbra_dir = -moon_to_sun_unit

                # 本影锥半顶角
                umbra_angle = np.arcsin((R_sun - R_moon) / moon_sun_dist)

                # 本影锥锥顶位置
                umbra_tip = sun_pos + R_sun / np.sin(umbra_angle) * umbra_dir

                # 半影锥半顶角
                penumbra_angle = np.arcsin((R_sun + R_moon) / moon_sun_dist)

                # 半影锥锥顶位置
                penumbra_tip = sun_pos + R_sun / np.sin(penumbra_angle) * umbra_dir

                # 日全食检测：本影锥锥顶在地球内部
                # tqdm.write(f"本影锥顶到地球中心的距离：{np.linalg.norm(umbra_tip - earth_pos)}, 地球半径：{R_earth}")
                if np.linalg.norm(umbra_tip - earth_pos) < R_earth:
                    eclipse_type = "Total"

                # 日偏食检测：本影锥锥顶在地球外部，且地球在半影锥内
                elif body_in_cone(earth_pos, R_earth, penumbra_tip, umbra_dir, penumbra_angle):
                    eclipse_type = "Partial"

                # 日环食检测：本影锥锥顶在地球外部，且地球在本影锥内（此本影锥的方向为+umbra_dir）
                elif body_in_cone(earth_pos, R_earth, umbra_tip, umbra_dir, umbra_angle):
                    eclipse_type = "Annular"

                # 加入日食事件列表
                eclipse_events["solar"].append({
                    "time": formatted_time,
                    "type": eclipse_type,
                })
        
        # ===== 月食初步筛选条件 =====
        # 条件1：月球位于地球与太阳连线的异侧（点积<0）
        # 条件2：地月距离小于地日距离（确保地球在太阳和月球之间）
        if np.dot(earth_to_sun_unit, earth_to_moon_unit) < 0 and (earth_moon_dist < earth_sun_dist):
            # 计算地日-地月向量夹角（精确到1e-12弧度）
            # 使用np.clip防止浮点误差导致反余弦计算异常
            cos_theta = np.clip(np.dot(earth_to_sun_unit, -earth_to_moon_unit), -1.0, 1.0)
            lunar_eclipse_angle = np.arccos(cos_theta)  # 夹角θ = arccos(EŜ·(-EM̂))
            lunar_eclipse_threshold = 2e-2  # 约1.146度（经验阈值）
            
            # 当夹角小于阈值时值得进入详细计算
            if lunar_eclipse_angle < lunar_eclipse_threshold:
                # 计算地球阴影锥基本参数
                # 影锥轴线
                earth_umbra_dir = -earth_to_sun_unit
                
                # 本影锥半顶角
                earth_umbra_angle = np.arcsin((R_sun - R_earth) / earth_sun_dist)
                
                # 本影锥锥顶位置
                earth_umbra_tip = earth_pos + (R_earth / np.tan(earth_umbra_angle)) * earth_umbra_dir
                
                # 半影锥半顶角
                earth_penumbra_angle = np.arcsin((R_sun + R_earth) / earth_sun_dist)
                
                # 半影锥锥顶位置
                earth_penumbra_tip = earth_pos - (R_earth / np.tan(earth_penumbra_angle)) * earth_umbra_dir
                
                # 月全食检测：月球完全进入地球本影锥
                if body_totally_in_cone(moon_pos, R_moon, earth_umbra_tip, earth_umbra_dir, earth_umbra_angle):
                    eclipse_type = "Total"
                    
                # 月偏食检测：月球部分进入地球本影锥
                elif body_in_cone(moon_pos, R_moon, earth_umbra_tip, earth_umbra_dir, earth_umbra_angle):
                    eclipse_type = "Partial"
                    
                # 半影月食检测：月球进入地球半影锥
                elif body_in_cone(moon_pos, R_moon, earth_penumbra_tip, earth_umbra_dir, earth_penumbra_angle):
                    eclipse_type = "Penumbral"
                    
                # 加入月食事件列表
                eclipse_events["lunar"].append({
                    "time": formatted_time,
                    "type": eclipse_type,
                })
        
        # 更新进度条
        pbar.update(1)
    
    # 关闭进度条
    pbar.close()
    
    print(f"分析完成！发现 {len(eclipse_events['solar'])} 次可能的日食和 {len(eclipse_events['lunar'])} 次可能的月食")
    return eclipse_events
