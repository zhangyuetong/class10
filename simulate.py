import numpy as np
from scipy.integrate import solve_ivp
import json
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u
from astropy.constants import G
import time

# 导入配置文件
from config import (
    BODIES, MASSES, START_DATE, TIME_SCALE, 
    SIMULATION_YEARS, SECONDS_PER_YEAR, OUTPUT_INTERVAL,
    SOLVER_METHOD, SOLVER_RTOL, SOLVER_ATOL,
    OUTPUT_FILENAME, PROGRESS_INTERVAL,
    INCLUDE_VELOCITY, COMPUTE_ORBITAL_PROPERTIES, POSITION_UNIT
)

# 万有引力常数（SI 单位 m^3 / (kg s^2)）
G_val = G.value

# 设置初始时刻
t0 = Time(START_DATE, scale=TIME_SCALE)
print(f"模拟开始时间: {t0.iso}")
print(f"模拟天体: {', '.join(BODIES)}")
print(f"模拟时长: {SIMULATION_YEARS} 年 (约 {SIMULATION_YEARS*365.25:.1f} 天)")

# 获取天体的初始位置和速度（相对于太阳系质心）
print("正在获取天体初始状态...")
init_state = []
for body in BODIES:
    pos, vel = get_body_barycentric_posvel(body, t0)
    # 转换为 SI 单位：位置单位转换为 m，速度转换为 m/s
    pos = pos.get_xyz().to(u.m).value  # 三维位置数组
    vel = vel.xyz.to(u.m/u.s).value  # 三维速度数组
    init_state.append(pos)
    init_state.append(vel)
# 将状态向量展平
y0 = np.hstack(init_state)

# 将字典转换为数组，保持顺序与BODIES一致
masses = np.array([MASSES[body] for body in BODIES])

def calculate_accelerations_vectorized(positions, masses):
    """
    使用向量化操作计算所有天体的加速度
    
    参数:
    positions: 形状为(n_bodies, 3)的数组，包含所有天体的位置
    masses: 形状为(n_bodies,)的数组，包含所有天体的质量
    
    返回:
    加速度: 形状为(n_bodies, 3)的数组
    """
    n_bodies = len(masses)
    accelerations = np.zeros((n_bodies, 3))
    
    # 计算所有天体对之间的位置差和距离
    # 使用广播机制: r_ij[i,j] = positions[i] - positions[j]
    r_ij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    
    # 计算距离的三次方
    # 添加小量epsilon避免自身作用时的除零错误
    epsilon = 1e-10
    dist_squared = np.sum(r_ij**2, axis=2)
    dist_cubed = np.power(dist_squared + epsilon, 1.5)
    
    # 计算加速度
    for i in range(n_bodies):
        # 对每个天体i，计算所有其他天体j对它的引力
        for j in range(n_bodies):
            if i != j:
                accelerations[i] += -G_val * masses[j] * r_ij[i, j] / dist_cubed[i, j]
    
    return accelerations

def derivatives(t, y):
    """
    计算状态向量的导数
    y: 状态向量，每个天体有6个元素 [x, y, z, vx, vy, vz]
    返回 dy/dt
    """
    n_bodies = len(BODIES)
    # 重构状态向量为 (n_bodies, 6) 形状
    state = y.reshape((n_bodies, 6))
    positions = state[:, :3]
    velocities = state[:, 3:]
    
    # 初始化导数向量
    dydt = np.zeros_like(state)
    
    # 位置的导数是速度
    dydt[:, :3] = velocities
    
    # 使用向量化函数计算加速度
    accelerations = calculate_accelerations_vectorized(positions, masses)
    
    # 速度的导数是加速度
    dydt[:, 3:] = accelerations
    
    return dydt.flatten()

# 计算系统的总能量和角动量
def compute_orbital_properties(state, masses):
    n_bodies = len(masses)
    positions = state[:, :3]
    velocities = state[:, 3:]
    
    # 计算总动能
    kinetic_energy = 0
    for i in range(n_bodies):
        kinetic_energy += 0.5 * masses[i] * np.sum(velocities[i]**2)
    
    # 计算总势能
    potential_energy = 0
    for i in range(n_bodies):
        for j in range(i+1, n_bodies):
            r_ij = positions[i] - positions[j]
            distance = np.linalg.norm(r_ij)
            potential_energy -= G_val * masses[i] * masses[j] / distance
    
    # 计算总角动量
    angular_momentum = np.zeros(3)
    for i in range(n_bodies):
        angular_momentum += masses[i] * np.cross(positions[i], velocities[i])
    
    return {
        "kinetic_energy": kinetic_energy,
        "potential_energy": potential_energy,
        "total_energy": kinetic_energy + potential_energy,
        "angular_momentum": angular_momentum.tolist()
    }

def process_output_data(sol, t0, save_file=True):
    """
    处理积分结果并输出数据
    
    参数:
    sol: 积分求解结果
    t0: 初始时间
    save_file: 是否保存到文件，默认为True
    
    返回:
    处理后的数据
    """
    print("开始处理输出数据...")
    output = []
    # 预处理数据以提高速度
    n_bodies = len(BODIES)
    state_reshaped = sol.y.reshape(n_bodies, 6, -1)  # 重塑为 (天体数, 状态维度, 时间点数)

    # 单位转换系数
    position_unit_factor = 1.0
    if POSITION_UNIT.lower() == 'au':
        position_unit_factor = 1.0 / 1.496e11  # 从米转换为AU
        print(f"位置将以天文单位(AU)输出")
    else:
        print(f"位置将以米(m)输出")

    # 确定是否输出速度
    if INCLUDE_VELOCITY:
        print("将包含速度数据")
    # 确定是否计算轨道特性
    if COMPUTE_ORBITAL_PROPERTIES:
        print("将计算轨道特性")

    # 按照配置的间隔输出进度
    total_points = len(sol.t)
    data_process_start = time.time()
    print(f"开始处理 {total_points} 个时间点的数据...")

    for i, t_sec in enumerate(sol.t):
        if i % PROGRESS_INTERVAL == 0:
            progress = i/total_points*100
            elapsed = time.time() - data_process_start
            est_total = elapsed / (i+1) * total_points if i > 0 else 0
            est_remaining = est_total - elapsed if est_total > 0 else 0
            print(f"已处理 {i}/{total_points} 点 ({progress:.1f}%) - 剩余时间约 {est_remaining:.1f} 秒")
        
        # 计算当前时刻
        current_time = t0 + t_sec * u.s
        
        # 确保时间格式化精确到整秒
        # 首先获取完整时间戳，然后格式化输出，忽略微小的数值误差
        exact_seconds = round(t_sec)
        formatted_time = (t0 + exact_seconds * u.s).iso
        
        # 构建当前时刻的数据记录
        record = {"time": formatted_time}
        
        # 提取当前时刻的状态
        current_state = np.zeros((n_bodies, 6))
        for j in range(n_bodies):
            current_state[j, :3] = state_reshaped[j, :3, i] * position_unit_factor
            current_state[j, 3:] = state_reshaped[j, 3:, i]
        
        # 添加位置数据和可选的速度数据
        for j, body in enumerate(BODIES):
            if INCLUDE_VELOCITY:
                record[body] = {
                    "position": current_state[j, :3].tolist(),
                    "velocity": current_state[j, 3:].tolist()
                }
            else:
                record[body] = current_state[j, :3].tolist()
        
        # 如果需要计算轨道特性
        if COMPUTE_ORBITAL_PROPERTIES and i % 10 == 0:  # 每10个点计算一次以减少计算量
            record["orbital_properties"] = compute_orbital_properties(current_state, masses)
        
        output.append(record)

    # 输出到 JSON 文件
    if save_file:
        print("正在保存数据到JSON文件...")
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"结果已保存至 {OUTPUT_FILENAME}")
        print(f"文件包含 {len(output)} 个时间点的数据，时间范围从 {output[0]['time']} 到 {output[-1]['time']}")
    
    return output

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

def detect_eclipses(sol, t0):
    """
    根据模拟结果检测可能的日食和月食事件
    
    参数:
    sol: 积分求解结果
    t0: 初始时间
    
    返回:
    日食和月食事件列表
    """
    print("开始分析日食月食事件...")
    
    # 获取天体索引
    sun_idx = BODIES.index('sun')
    earth_idx = BODIES.index('earth')
    moon_idx = BODIES.index('moon')
    
    # 天体半径(米)
    sun_radius = 6.957e8  # 太阳半径
    earth_radius = 6.371e6  # 地球半径
    moon_radius = 1.737e6  # 月球半径
    
    # 预处理数据
    n_bodies = len(BODIES)
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

# 设置积分总时间
total_time = SIMULATION_YEARS * SECONDS_PER_YEAR

# 设置积分输出时间点
# 使用arange保证时间点精确为整数倍的OUTPUT_INTERVAL
n_points = int(total_time/OUTPUT_INTERVAL) + 1
t_eval = np.arange(0, total_time + 0.1, OUTPUT_INTERVAL)  # 添加0.1秒的容差确保包含结束时间
print(f"将生成 {len(t_eval)} 个数据点，时间间隔为 {OUTPUT_INTERVAL} 秒 (每 {OUTPUT_INTERVAL/3600:.1f} 小时)")

# 使用指定方法求解微分方程
print(f"\n开始积分计算，使用 {SOLVER_METHOD} 积分器...")
start_time = time.time()

# 求解微分方程
sol = solve_ivp(
    derivatives, 
    [0, total_time], 
    y0, 
    method=SOLVER_METHOD, 
    t_eval=t_eval, 
    rtol=SOLVER_RTOL, 
    atol=SOLVER_ATOL
)

elapsed = time.time() - start_time
print(f"积分计算完成！用时 {elapsed:.2f} 秒\n")

# 使用积分结果计算是否发生日食月食
eclipse_events = detect_eclipses(sol, t0)

# 将日食月食事件保存到JSON文件
with open("eclipse_events.json", "w", encoding="utf-8") as f:
    json.dump(eclipse_events, f, indent=2, ensure_ascii=False)
print(f"日食月食事件已保存至 eclipse_events.json")

# 默认不调用输出数据处理函数
# 如果需要输出轨道数据，取消下面一行的注释
# output_data = process_output_data(sol, t0)

total_elapsed = time.time() - start_time
print(f"模拟完成！总用时 {total_elapsed:.2f} 秒")
