"""
日食月食预测模块

此模块提供日食和月食事件的预测功能，基于天体三维位置数据。
"""

import numpy as np
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm

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

def detect_eclipses(sol, t0, bodies):
    """
    根据模拟结果检测可能的日食和月食事件
    
    参数:
    sol: 积分求解结果
    t0: 初始时间
    bodies: 天体列表
    
    返回:
    日食和月食事件列表
    """
    print("开始分析日食月食事件...")
    
    # 获取天体索引
    sun_idx = bodies.index('sun')
    earth_idx = bodies.index('earth')
    moon_idx = bodies.index('moon')
    
    # 天体半径(米)
    sun_radius = 6.957e8  # 太阳半径
    earth_radius = 6.371e6  # 地球半径
    moon_radius = 1.737e6  # 月球半径
    
    # 预处理数据
    n_bodies = len(bodies)
    state_reshaped = sol.y.reshape(n_bodies, 6, -1)
    
    eclipse_events = {
        "solar": [],  # 日食
        "lunar": []   # 月食
    }
    
    # 使用tqdm创建进度条
    pbar = tqdm(total=len(sol.t), desc="分析进度")
    
    # 遍历所有时间点
    for i, t_sec in enumerate(sol.t):
        # 获取当前时间
        current_time = t0 + t_sec * u.s
        formatted_time = current_time.iso
        
        # 获取天体位置
        sun_pos = state_reshaped[sun_idx, :3, i]
        earth_pos = state_reshaped[earth_idx, :3, i]
        moon_pos = state_reshaped[moon_idx, :3, i]
        
        # 计算向量
        earth_to_sun = sun_pos - earth_pos
        earth_to_moon = moon_pos - earth_pos
        moon_to_sun = sun_pos - moon_pos
        
        # 计算距离
        earth_sun_dist = np.linalg.norm(earth_to_sun)
        earth_moon_dist = np.linalg.norm(earth_to_moon)
        moon_sun_dist = np.linalg.norm(moon_to_sun)
        
        # 归一化向量
        earth_to_sun_unit = earth_to_sun / earth_sun_dist
        earth_to_moon_unit = earth_to_moon / earth_moon_dist
        moon_to_sun_unit = moon_to_sun / moon_sun_dist
        
        # ===== 日食检测 =====
        # 要求月球在地球和太阳之间：即地球–月球与地球–太阳同向，且地月距离 < 地日距离
        if np.dot(earth_to_sun_unit, earth_to_moon_unit) > 0 and (earth_moon_dist < earth_sun_dist):
            # 太阳-月球-地球共线检测
            # 计算向量夹角（弧度）：这里使用 moon_to_sun_unit 和 earth_to_moon_unit 的夹角
            solar_eclipse_angle = np.arccos(np.dot(moon_to_sun_unit, earth_to_moon_unit))
            solar_eclipse_threshold = 1e-2  # 角度阈值(弧度)
            if solar_eclipse_angle < solar_eclipse_threshold:
                # 计算月球本影锥的顶角
                umbra_angle = 2 * np.arctan((sun_radius - moon_radius) / moon_sun_dist)
                
                # 计算月球本影锥长度
                umbra_length = moon_radius / np.tan(umbra_angle / 2)
                
                # 计算月球半影锥的顶角
                penumbra_angle = 2 * np.arctan((sun_radius + moon_radius) / moon_sun_dist)
                
                # 计算月球投影到日地连线上的距离
                projection = np.dot(earth_to_moon, earth_to_sun_unit)
                
                # 计算月球到日地连线的垂直距离（视差距离）
                perp_dist = np.sqrt(earth_moon_dist**2 - projection**2)
                
                # 计算月球本影在地球处的半径
                # 如果本影锥长度大于月球到地球的距离，使用正值；否则使用负值表示"虚本影"
                if umbra_length > earth_moon_dist:
                    # 月球本影可以延伸到地球，可能发生全食
                    umbra_radius_at_earth = moon_radius - (earth_moon_dist * np.tan(umbra_angle / 2))
                else:
                    # 月球本影不够长，不能延伸到地球，可能发生环食
                    umbra_radius_at_earth = -(earth_moon_dist - umbra_length) * np.tan(umbra_angle / 2)
                
                # 计算半影在地球处的半径
                penumbra_radius_at_earth = moon_radius + (earth_moon_dist * np.tan(penumbra_angle / 2))
                
                # 计算日食类型
                if perp_dist < abs(umbra_radius_at_earth):
                    if umbra_radius_at_earth > 0:
                        eclipse_type = "total"  # 全食
                    else:
                        eclipse_type = "annular"  # 环食
                elif perp_dist < penumbra_radius_at_earth:
                    eclipse_type = "partial"  # 偏食
                else:
                    # 虽然满足共线条件，但不符合几何关系，不发生日食
                    continue
                
                # 计算最大食分
                if abs(umbra_radius_at_earth) < penumbra_radius_at_earth:  # 防止除以零
                    eclipse_magnitude = (penumbra_radius_at_earth - perp_dist) / (penumbra_radius_at_earth - abs(umbra_radius_at_earth))
                    eclipse_magnitude = min(1.0, max(0.0, eclipse_magnitude))
                else:
                    eclipse_magnitude = 0.0
                
                # 计算太阳和月球的视半径
                sun_angular_radius = np.arctan(sun_radius / earth_sun_dist)
                moon_angular_radius = np.arctan(moon_radius / earth_moon_dist)
                
                eclipse_events["solar"].append({
                    "time": formatted_time,
                    "type": eclipse_type,
                    "angle": float(solar_eclipse_angle),
                    "perpendicular_distance": float(perp_dist),
                    "umbra_radius": float(umbra_radius_at_earth),
                    "penumbra_radius": float(penumbra_radius_at_earth),
                    "magnitude": float(eclipse_magnitude),
                    "sun_angular_radius": float(sun_angular_radius),
                    "moon_angular_radius": float(moon_angular_radius)
                })
                
                # 在进度条下方输出检测结果
                tqdm.write(f"检测到可能的日食事件: {formatted_time}, 类型: {eclipse_type}, 食分: {eclipse_magnitude:.2f}")
        
        # ===== 月食检测 =====
        # 要求地球位于太阳和月球之间，即地球–太阳与地球–月球反向（点积 < 0）
        if np.dot(earth_to_sun_unit, earth_to_moon_unit) < 0:
            # 计算太阳-地球-月球夹角（弧度），应接近π
            lunar_eclipse_angle = np.arccos(np.dot(earth_to_sun_unit, earth_to_moon_unit))
            lunar_eclipse_threshold = 1e-2  # 角度阈值(弧度)
            if abs(lunar_eclipse_angle - np.pi) < lunar_eclipse_threshold:
                # 计算地球本影锥顶角
                earth_umbra_angle = 2 * np.arctan((sun_radius - earth_radius) / earth_sun_dist)
                
                # 计算地球本影锥长度
                earth_umbra_length = earth_radius / np.tan(earth_umbra_angle / 2)
                
                # 计算地球半影锥顶角
                earth_penumbra_angle = 2 * np.arctan((sun_radius + earth_radius) / earth_sun_dist)
                
                # 若月球在地球影子范围内（本影或半影）
                if earth_moon_dist < earth_umbra_length or True:  # 这里不再做严格距离限制
                    # 计算月球到日地连线的垂直距离
                    projection = np.dot(earth_to_moon, earth_to_sun_unit)
                    perp_dist = np.sqrt(earth_moon_dist**2 - projection**2)
                    
                    # 计算地球本影在月球距离处的半径
                    umbra_radius_at_moon = earth_radius - (earth_moon_dist * np.tan(earth_umbra_angle / 2))
                    
                    # 计算地球半影在月球距离处的半径
                    penumbra_radius_at_moon = earth_radius + (earth_moon_dist * np.tan(earth_penumbra_angle / 2))
                    
                    # 判断月食类型
                    if perp_dist < umbra_radius_at_moon - moon_radius:
                        eclipse_type = "total"  # 全食
                    elif perp_dist < umbra_radius_at_moon + moon_radius:
                        eclipse_type = "partial"  # 偏食
                    elif perp_dist < penumbra_radius_at_moon + moon_radius:
                        eclipse_type = "penumbral"  # 半影食
                    else:
                        continue
                    
                    # 计算月食食分
                    if eclipse_type in ["total", "partial"]:
                        magnitude = (umbra_radius_at_moon + moon_radius - perp_dist) / (2 * moon_radius)
                    else:
                        magnitude = (penumbra_radius_at_moon + moon_radius - perp_dist) / (2 * moon_radius)
                    
                    magnitude = min(1.0, max(0.0, magnitude))
                    
                    eclipse_events["lunar"].append({
                        "time": formatted_time,
                        "type": eclipse_type,
                        "angle": float(lunar_eclipse_angle),
                        "perpendicular_distance": float(perp_dist),
                        "umbra_radius": float(umbra_radius_at_moon),
                        "penumbra_radius": float(penumbra_radius_at_moon),
                        "magnitude": float(magnitude)
                    })
                    
                    # 在进度条下方输出检测结果
                    tqdm.write(f"检测到可能的月食事件: {formatted_time}, 类型: {eclipse_type}, 食分: {magnitude:.2f}")
        
        # 更新进度条
        pbar.update(1)
    
    # 关闭进度条
    pbar.close()
    
    print(f"分析完成！发现 {len(eclipse_events['solar'])} 次可能的日食和 {len(eclipse_events['lunar'])} 次可能的月食")
    return eclipse_events
