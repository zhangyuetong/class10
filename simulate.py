import numpy as np
from scipy.integrate import solve_ivp
import json
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u
from astropy.constants import G

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

# 获取天体的初始位置和速度（相对于太阳系质心）
init_state = []
for body in BODIES:
    pos, vel = get_body_barycentric_posvel(body, t0)
    # 转换为 SI 单位：位置单位转换为 m，速度转换为 m/s
    pos = pos.get_xyz().to(u.m).value  # 三维位置数组
    vel = vel.xyz.to(u.m/u.s).value  # 三维速度数组
    init_state.append(pos)
    init_state.append(vel)
# 将状态向量展平，形状为(18,)
y0 = np.hstack(init_state)

# 将字典转换为数组，保持顺序与BODIES一致
masses = np.array([MASSES[body] for body in BODIES])

def derivatives(t, y):
    """
    y: 状态向量，共 18 个元素：
       [r_sun (3), v_sun (3), r_earth (3), v_earth (3), r_moon (3), v_moon (3)]
    返回 dy/dt，同样为 18 维向量。
    """
    dydt = np.zeros_like(y)
    # 为简化，我们可以先把状态向量重构为 (n_bodies, 6)
    n_bodies = len(BODIES)
    state = y.reshape((n_bodies, 6))
    pos = state[:, :3]
    vel = state[:, 3:]
    
    # 对每个天体，其位置的导数即为当前速度
    dydt[0:n_bodies*6:6] = vel[:, 0]
    dydt[1:n_bodies*6:6] = vel[:, 1]
    dydt[2:n_bodies*6:6] = vel[:, 2]
    
    # 计算加速度：对每个天体 i，a_i = sum_{j≠i} -G*m_j*(r_i - r_j)/|r_i-r_j|^3
    acc = np.zeros((n_bodies, 3))
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i == j:
                continue
            r_ij = pos[i] - pos[j]
            distance = np.linalg.norm(r_ij)
            # 避免除零：在真实问题中天体不会重合
            acc[i] += -G_val * masses[j] * r_ij / distance**3
    # 将加速度存入导数向量
    dydt = y.copy().reshape((n_bodies,6))
    dydt[:, :3] = vel  # 位置的导数
    dydt[:, 3:] = acc  # 速度的导数
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

# 设置积分总时间
total_time = SIMULATION_YEARS * SECONDS_PER_YEAR

# 设置积分输出时间点
t_eval = np.linspace(0, total_time, int(total_time/OUTPUT_INTERVAL) + 1)

# 使用指定方法求解微分方程
print("开始积分计算，请耐心等待...")
sol = solve_ivp(derivatives, [0, total_time], y0, method=SOLVER_METHOD, 
                t_eval=t_eval, rtol=SOLVER_RTOL, atol=SOLVER_ATOL)
print("积分计算完成.")

# 整理输出数据，构造一个列表，每个元素为一个时刻的记录
print("开始处理输出数据...")
output = []
# 预处理数据以提高速度
state_reshaped = sol.y.reshape(len(BODIES), 6, -1)  # 重塑为 (天体数, 状态维度, 时间点数)

# 单位转换系数
position_unit_factor = 1.0
if POSITION_UNIT.lower() == 'au':
    position_unit_factor = 1.0 / 1.496e11  # 从米转换为AU

# 按照配置的间隔输出进度
total_points = len(sol.t)
for i, t_sec in enumerate(sol.t):
    if i % PROGRESS_INTERVAL == 0:
        print(f"已处理 {i}/{total_points} 点 ({i/total_points*100:.1f}%)")
    
    # 计算当前时刻
    current_time = t0 + t_sec * u.s
    
    # 构建当前时刻的数据记录
    record = {"time": current_time.iso}
    
    # 提取当前时刻的状态
    current_state = np.zeros((len(BODIES), 6))
    for j in range(len(BODIES)):
        current_state[j, :3] = state_reshaped[j, :3, i] * position_unit_factor
        current_state[j, 3:] = state_reshaped[j, 3:, i]
    
    # 添加位置数据
    for j, body in enumerate(BODIES):
        if INCLUDE_VELOCITY:
            record[body] = {
                "position": current_state[j, :3].tolist(),
                "velocity": current_state[j, 3:].tolist()
            }
        else:
            record[body] = current_state[j, :3].tolist()
    
    # 如果需要计算轨道特性
    if COMPUTE_ORBITAL_PROPERTIES:
        record["orbital_properties"] = compute_orbital_properties(current_state, masses)
    
    output.append(record)

# 输出到 JSON 文件
print("正在保存数据到JSON文件...")
with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"结果已保存至 {OUTPUT_FILENAME}")
