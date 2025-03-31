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

# 导入日食月食预测模块
from eclipse_prediction import detect_eclipses

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
eclipse_events = detect_eclipses(sol, t0, BODIES)

# 将日食月食事件保存到JSON文件
with open("eclipse_events.json", "w", encoding="utf-8") as f:
    json.dump(eclipse_events, f, indent=2, ensure_ascii=False)
print(f"日食月食事件已保存至 eclipse_events.json")

# 默认不调用输出数据处理函数
# 如果需要输出轨道数据，取消下面一行的注释
# output_data = process_output_data(sol, t0)

total_elapsed = time.time() - start_time
print(f"模拟完成！总用时 {total_elapsed:.2f} 秒")
