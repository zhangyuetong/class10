import numpy as np 
from scipy.integrate import solve_ivp
import json
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u
import time
from astropy.constants import G

# 导入配置文件
from config import (
    BODIES, MASSES, START_DATE, TIME_SCALE, 
    SIMULATION_YEARS, SECONDS_PER_YEAR, OUTPUT_INTERVAL,
    SOLVER_METHOD, SOLVER_RTOL, SOLVER_ATOL,
)

# 导入日食月食预测模块
from eclipse_prediction import predict_eclipses

# 导入误差分析模块
from error_analysis import run_error_analysis

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
    vel = vel.xyz.to(u.m/u.s).value    # 三维速度数组
    init_state.append(pos)
    init_state.append(vel)
# 将状态向量展平
y0 = np.hstack(init_state)

# 将字典转换为数组，保持顺序与BODIES一致
masses = np.array([MASSES[body] for body in BODIES])

def calculate_accelerations_vectorized(positions, masses):
    """
    使用向量化操作计算所有天体的加速度
    """
    n_bodies = len(masses)
    accelerations = np.zeros((n_bodies, 3))
    
    # 计算所有天体对之间的位置差和距离
    r_ij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    epsilon = 1e-10
    dist_squared = np.sum(r_ij**2, axis=2)
    dist_cubed = np.power(dist_squared + epsilon, 1.5)
    
    # 计算加速度
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                accelerations[i] += -G_val * masses[j] * r_ij[i, j] / dist_cubed[i, j]
    
    return accelerations

def derivatives(t, y):
    """
    计算状态向量的导数
    """
    n_bodies = len(BODIES)
    state = y.reshape((n_bodies, 6))
    positions = state[:, :3]
    velocities = state[:, 3:]
    
    dydt = np.zeros_like(state)
    # 位置的导数是速度
    dydt[:, :3] = velocities
    # 速度的导数是加速度
    accelerations = calculate_accelerations_vectorized(positions, masses)
    dydt[:, 3:] = accelerations
    
    return dydt.flatten()

# 设置积分总时间
total_time = SIMULATION_YEARS * SECONDS_PER_YEAR

# 设置积分输出时间点
n_points = int(total_time / OUTPUT_INTERVAL) + 1
t_eval = np.arange(0, total_time + 0.1, OUTPUT_INTERVAL)
print(f"将生成 {len(t_eval)} 个数据点，时间间隔为 {OUTPUT_INTERVAL} 秒 (每 {OUTPUT_INTERVAL/60:.2f} 分钟)")

print(f"\n开始积分计算，使用 {SOLVER_METHOD} 积分器...")
start_time = time.time()

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

# 重塑状态数组 (天体数, 6, 时间点数)
n_bodies = len(BODIES)
state_reshaped = sol.y.reshape(n_bodies, 6, -1)

# 日食月食预测
eclipse_events = predict_eclipses(sol, t0, state_reshaped, BODIES.index('sun'), BODIES.index('earth'), BODIES.index('moon'))
with open("eclipse_events.json", "w", encoding="utf-8") as f:
    json.dump(eclipse_events, f, indent=2, ensure_ascii=False)
print(f"日食月食事件已保存至 eclipse_events.json")

# 运行误差分析
run_error_analysis(sol, t0, state_reshaped, BODIES, t_eval)

total_elapsed = time.time() - start_time
print(f"模拟完成！总用时 {total_elapsed:.2f} 秒")
