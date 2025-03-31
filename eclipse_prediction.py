"""
日食月食预测模块

此模块提供日食和月食事件的预测功能，基于天体三维位置数据。
"""

import numpy as np
from astropy.time import Time
import astropy.units as u

"""
日食月食的几何推导

1. 日食原理与精确判定条件

   日食发生的本质是月球遮挡太阳光线射向地球的现象。根据几何光学原理，我们可以进行如下推导：

   a) 锥体阴影模型:
      设太阳半径为Rs，月球半径为Rm，地球半径为Re
      太阳到月球距离为Dsm，月球到地球距离为Dme
      
      - 本影锥(Umbra)：完全遮挡太阳的区域
        * 锥顶角 α = 2·arctan((Rs-Rm)/Dsm)
        * 本影锥长度 Lu = Rm / tan(α/2)
        * 月球本影在地球处的半径 Ru = Rm - Dme·tan(α/2)
        
      - 半影锥(Penumbra)：部分遮挡太阳的区域
        * 锥顶角 β = 2·arctan((Rs+Rm)/Dsm)
        * 半影在地球处的半径 Rp = Rm + Dme·tan(β/2)
   
   b) 日食类型判定：
      - 全食：当Lu > Dme且观测点在本影区域内，即月球本影覆盖地球表面
        条件：Ru > 0 且观测点到月球-地球连线的距离 < Ru
      
      - 环食：当Lu < Dme，月球本影锥不够长，不能延伸到地球表面
        条件：Ru < 0 且|Ru| < Re，即虽然本影不够长，但理论上的"负半径本影"仍在地球范围内
      
      - 偏食：观测点在半影区域内但不在本影区域内
        条件：观测点到月球-地球连线的距离 > Ru 且 < Rp

   c) 视直径比较：
      太阳视半径 = arctan(Rs/Dse)，月球视半径 = arctan(Rm/Dme)
      - 全食：月球视半径 > 太阳视半径
      - 环食：月球视半径 < 太阳视半径
      - 最大食分 = (太阳视直径 + 月球视直径 - 视距离)/太阳视直径
      
   d) 精确轨道考量：
      - 考虑轨道偏心率导致的距离变化
      - 考虑月球和地球轨道的倾角
      - 地球表面观测点的纬度和经度影响

   e) 共线条件的精确表达：
      如果定义向量：
      地球-太阳向量: ⃗r_ES = ⃗r_S - ⃗r_E
      地球-月球向量: ⃗r_EM = ⃗r_M - ⃗r_E
      则共线条件为向量夹角 θ = arccos(⃗r_ES·⃗r_EM/(|⃗r_ES|·|⃗r_EM|)) 接近 0或π

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
    
    # 遍历所有时间点
    for i, t_sec in enumerate(sol.t):
        # 每50个点显示一次进度，避免输出过多
        if i % 50 == 0:
            print(f"正在分析时间点 {i}/{len(sol.t)}")
            
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
        
        # 日食判断：太阳-月球-地球共线检测
        # 计算向量夹角（弧度）
        solar_eclipse_angle = np.arccos(np.dot(-moon_to_sun_unit, earth_to_moon_unit))
        
        # 月食判断：太阳-地球-月球夹角
        lunar_eclipse_angle = np.arccos(np.dot(earth_to_sun_unit, earth_to_moon_unit))
        
        # ===== 精确的日食判断 =====
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
            # 近似计算：(半影半径 - 垂直距离) / (半影半径 - |本影半径|)
            if abs(umbra_radius_at_earth) < penumbra_radius_at_earth:  # 防止除以零
                eclipse_magnitude = (penumbra_radius_at_earth - perp_dist) / (penumbra_radius_at_earth - abs(umbra_radius_at_earth))
                eclipse_magnitude = min(1.0, max(0.0, eclipse_magnitude))  # 限制在0-1之间
            else:
                eclipse_magnitude = 0.0
            
            # 计算太阳和月球的视半径
            sun_angular_radius = np.arctan(sun_radius / earth_sun_dist)
            moon_angular_radius = np.arctan(moon_radius / earth_moon_dist)
            
            # 记录日食事件
            eclipse_events["solar"].append({
                "time": formatted_time,
                "type": eclipse_type,
                "angle": float(solar_eclipse_angle),  # 转换为Python标准浮点数
                "perpendicular_distance": float(perp_dist),
                "umbra_radius": float(umbra_radius_at_earth),
                "penumbra_radius": float(penumbra_radius_at_earth),
                "magnitude": float(eclipse_magnitude),
                "sun_angular_radius": float(sun_angular_radius),
                "moon_angular_radius": float(moon_angular_radius)
            })
            
            print(f"检测到可能的日食事件: {formatted_time}, 类型: {eclipse_type}, 食分: {eclipse_magnitude:.2f}")
        
        # 月食检测：太阳-地球-月球接近共线，且地球在太阳和月球之间
        lunar_eclipse_threshold = 1e-2  # 角度阈值(弧度)
        if lunar_eclipse_angle < lunar_eclipse_threshold:
            # 计算地球本影锥顶角
            earth_umbra_angle = 2 * np.arctan((sun_radius - earth_radius) / earth_sun_dist)
            
            # 计算地球本影锥长度
            earth_umbra_length = earth_radius / np.tan(earth_umbra_angle / 2)
            
            # 计算地球半影锥顶角
            earth_penumbra_angle = 2 * np.arctan((sun_radius + earth_radius) / earth_sun_dist)
            
            # 如果月球在地球本影或半影区域内
            if earth_moon_dist < earth_umbra_length:
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
                    # 虽然满足共线条件，但几何关系不满足，不发生月食
                    continue
                
                # 计算月食食分
                if eclipse_type == "total" or eclipse_type == "partial":
                    # 本影食的食分
                    magnitude = (umbra_radius_at_moon + moon_radius - perp_dist) / (2 * moon_radius)
                else:
                    # 半影食的食分
                    magnitude = (penumbra_radius_at_moon + moon_radius - perp_dist) / (2 * moon_radius)
                
                magnitude = min(1.0, max(0.0, magnitude))  # 限制在0-1之间
                
                eclipse_events["lunar"].append({
                    "time": formatted_time,
                    "type": eclipse_type,
                    "angle": float(lunar_eclipse_angle),
                    "perpendicular_distance": float(perp_dist),
                    "umbra_radius": float(umbra_radius_at_moon),
                    "penumbra_radius": float(penumbra_radius_at_moon),
                    "magnitude": float(magnitude)
                })
                
                print(f"检测到可能的月食事件: {formatted_time}, 类型: {eclipse_type}, 食分: {magnitude:.2f}")
    
    print(f"分析完成！发现 {len(eclipse_events['solar'])} 次可能的日食和 {len(eclipse_events['lunar'])} 次可能的月食")
    return eclipse_events 