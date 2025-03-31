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

1. 日食原理
   太阳-月球-地球三者接近一条直线，且月球位于太阳和地球之间时，
   月球遮挡了部分或全部射向地球的阳光，形成日食。

   日食判定条件：
   a) 太阳-月球-地球三点夹角接近180度
   b) 月球的视直径足够大，能够遮挡太阳
   c) 月球的阴影覆盖了地球表面的观测点

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
        
        # 计算夹角（弧度）
        # 日食判断：太阳-月球-地球夹角
        solar_eclipse_angle = np.arccos(np.dot(-moon_to_sun_unit, earth_to_moon_unit))
        # 月食判断：太阳-地球-月球夹角
        lunar_eclipse_angle = np.arccos(np.dot(earth_to_sun_unit, earth_to_moon_unit))
        
        # 太阳在月球上的半影锥角度
        sun_penumbra_angle = np.arctan((sun_radius + moon_radius) / moon_sun_dist)
        # 太阳在地球上的半影锥角度
        sun_earth_penumbra_angle = np.arctan((sun_radius + earth_radius) / earth_sun_dist)
        
        # 日食检测：太阳-月球-地球接近共线，且月球在太阳和地球之间
        solar_eclipse_threshold = 1e-2  # 角度阈值(弧度)
        if solar_eclipse_angle < solar_eclipse_threshold:
            # 计算月球投影到日地连线上的距离
            projection = np.dot(earth_to_moon, earth_to_sun_unit)
            
            # 计算月球到日地连线的垂直距离
            perp_dist = np.sqrt(earth_moon_dist**2 - projection**2)
            
            # 计算太阳阴影锥在地球处的半径
            shadow_radius_at_earth = (sun_radius - moon_radius) * (earth_sun_dist / moon_sun_dist) - moon_radius
            
            if perp_dist < shadow_radius_at_earth:
                eclipse_type = "total" if perp_dist < (moon_radius - shadow_radius_at_earth) else "partial"
                eclipse_events["solar"].append({
                    "time": formatted_time,
                    "type": eclipse_type,
                    "angle": solar_eclipse_angle,
                    "distance": perp_dist
                })
                print(f"检测到可能的日食事件: {formatted_time}, 类型: {eclipse_type}")
        
        # 月食检测：太阳-地球-月球接近共线，且地球在太阳和月球之间
        lunar_eclipse_threshold = 1e-2  # 角度阈值(弧度)
        if lunar_eclipse_angle < lunar_eclipse_threshold:
            # 计算地球本影锥长度
            earth_umbra_length = earth_radius * moon_sun_dist / (sun_radius - earth_radius)
            
            # 如果月球在地球本影内
            if earth_moon_dist < earth_umbra_length:
                # 计算月球到日地连线的垂直距离
                projection = np.dot(earth_to_moon, earth_to_sun_unit)
                perp_dist = np.sqrt(earth_moon_dist**2 - projection**2)
                
                # 计算地球阴影锥在月球距离处的半径
                shadow_radius_at_moon = earth_radius * (1 - earth_moon_dist / earth_umbra_length)
                
                if perp_dist < shadow_radius_at_moon:
                    eclipse_type = "total" if perp_dist < (shadow_radius_at_moon - moon_radius) else "partial"
                    eclipse_events["lunar"].append({
                        "time": formatted_time,
                        "type": eclipse_type,
                        "angle": lunar_eclipse_angle,
                        "distance": perp_dist
                    })
                    print(f"检测到可能的月食事件: {formatted_time}, 类型: {eclipse_type}")
    
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
