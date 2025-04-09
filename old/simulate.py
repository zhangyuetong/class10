import numpy as np
from scipy.integrate import solve_ivp
import json
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel

# 全局设置星表
from astropy.coordinates import solar_system_ephemeris
solar_system_ephemeris.set('de440')

import astropy.units as u
import time

# 导入配置文件
from config import (
    BODIES,
    GM_DICT,
    START_DATE,
    TIME_SCALE,
    SIMULATION_YEARS,
    SECONDS_PER_YEAR,
    OUTPUT_INTERVAL,
    SOLVER_METHOD,
    SOLVER_RTOL,
    SOLVER_ATOL,
)

# 导入日食月食预测模块（此处已是向量化优化版）
from eclipse_prediction import predict_eclipses

# 如果有误差分析模块，可在此导入
try:
    from error_analysis import run_error_analysis
    ERROR_ANALYSIS_AVAILABLE = True
except ImportError:
    ERROR_ANALYSIS_AVAILABLE = False
    print("[警告] 未找到 error_analysis.py，跳过误差分析。")


# 设置初始时刻
t0 = Time(START_DATE, scale=TIME_SCALE)
print(f"模拟开始时间: {t0.iso}")
print(f"模拟天体: {', '.join(BODIES)}")
print(f"模拟时长: {SIMULATION_YEARS} 年 (约 {SIMULATION_YEARS*365.24:.1f} 天)")

# 获取天体的初始位置和速度（相对于太阳系质心）
print("正在获取天体初始状态...")
init_state = []
for body in BODIES:
    pos, vel = get_body_barycentric_posvel(body, t0)
    # 转换为 SI 单位：位置（米），速度（米/秒）
    pos_m = pos.get_xyz().to(u.m).value  
    vel_m_s = vel.xyz.to(u.m/u.s).value  
    init_state.append(pos_m)
    init_state.append(vel_m_s)

# 初始状态向量 y0
y0 = np.hstack(init_state)

# 将 GM_DICT (各天体 GM) 转为数组，与 BODIES 顺序对应
gm_values = np.array([GM_DICT[body] for body in BODIES])

# ------------------------------
# 计算加速度的辅助函数（向量化）
# ------------------------------
def calculate_accelerations_vectorized(positions, gm_values):
    """
    使用向量化操作计算所有天体的加速度 (3D)。
    positions : shape = (n_bodies, 3)
    gm_values : shape = (n_bodies,)  # 各天体 GM
    """
    n_bodies = len(gm_values)
    accelerations = np.zeros((n_bodies, 3))

    # 所有位置差向量: r_ij = r_i - r_j
    r_ij = positions[:, None, :] - positions[None, :, :]  # shape=(n_bodies, n_bodies, 3)
    dist_sq = np.sum(r_ij**2, axis=2)                    # shape=(n_bodies, n_bodies)
    dist_cubed = dist_sq**1.5                            # shape=(n_bodies, n_bodies)

    # 计算引力加速度
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                accelerations[i] -= gm_values[j] * r_ij[i, j] / dist_cubed[i, j]

    return accelerations

# ------------------------------
# ODE 求解器右端函数
# ------------------------------
def derivatives(t, y):
    """
    y 包含 n_bodies 个天体的 [x,y,z, vx,vy,vz]，
    返回同样形状的导数 dydt = [vx,vy,vz, ax,ay,az].
    """
    n_bodies = len(BODIES)
    state = y.reshape((n_bodies, 6))
    positions = state[:, :3]
    velocities = state[:, 3:]

    # 位置导数 = 速度
    dydt = np.zeros_like(state)
    dydt[:, :3] = velocities

    # 速度导数 = 引力加速度
    accelerations = calculate_accelerations_vectorized(positions, gm_values)
    dydt[:, 3:] = accelerations

    return dydt.flatten()

# ------------------------------
# 设置积分时间和采样
# ------------------------------
total_time = SIMULATION_YEARS * SECONDS_PER_YEAR
t_eval = np.arange(0, total_time + 0.1, OUTPUT_INTERVAL)  # 每隔 OUTPUT_INTERVAL 秒输出

print(f"将生成 {len(t_eval)} 个数据点，时间间隔 = {OUTPUT_INTERVAL} 秒")
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

# 整理积分结果
n_bodies = len(BODIES)
state_reshaped = sol.y.reshape(n_bodies, 6, -1)

# ------------------------------
# 日食月食预测
# ------------------------------
sun_idx   = BODIES.index('sun')
earth_idx = BODIES.index('earth')
moon_idx  = BODIES.index('moon')

eclipse_events = predict_eclipses(
    sol, t0, state_reshaped,
    sun_idx, earth_idx, moon_idx
)

with open("eclipse_events.json", "w", encoding="utf-8") as f:
    json.dump(eclipse_events, f, indent=2, ensure_ascii=False)
print(f"日食月食事件已保存至 eclipse_events.json")

# ------------------------------
# 如果需要，可执行误差分析
# ------------------------------
if ERROR_ANALYSIS_AVAILABLE:
    run_error_analysis(sol, t0, state_reshaped, BODIES, t_eval)

total_elapsed = time.time() - start_time
print(f"模拟完成！总用时 {total_elapsed:.2f} 秒")
