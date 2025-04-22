import numpy as np
from scipy.integrate import solve_ivp
import json
import time
from datetime import timedelta, datetime

from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
import astropy.units as u

import matplotlib.pyplot as plt  # 用于最终的误差分析绘图

# 配置星历表
solar_system_ephemeris.set('de440')

# 导入配置(示例)
from config import (
    BODIES, GM_DICT, 
    START_DATE, TIME_SCALE, SIMULATION_YEARS, SECONDS_PER_YEAR,
    SOLVER_METHOD, SOLVER_RTOL, SOLVER_ATOL, MAX_STEP_TIME, 
    RELATIVITY_SWITCH, C_LIGHT,
    J2_SWITCH, J2_DICT, RADIUS_DICT,
    MAX_SIN_ANGLE, 
    ERROR_EVAL_INTERVAL, ERROR_ANALYSIS_BODIES
)

# ------------------- 几何判断函数 -------------------
def body_in_cone(body_pos, body_radius, cone_tip, cone_axis, half_angle):
    """
    判断天体是否进入某个圆锥区域（半影/本影等）
    若返回值 < 0 则说明“在锥内”
    """
    cone_tip_to_body = body_pos - cone_tip
    proj_point = cone_tip + np.dot(cone_tip_to_body, cone_axis) * cone_axis / np.linalg.norm(cone_axis)**2
    perpendicular_distance = np.linalg.norm(body_pos - proj_point)
    horizontal_distance = np.linalg.norm(cone_tip - proj_point)
    critical_distance = np.tan(half_angle) * horizontal_distance + body_radius / np.cos(half_angle)
    return perpendicular_distance - critical_distance


def body_totally_in_cone(body_pos, body_radius, cone_tip, cone_axis, half_angle):
    """
    判断天体是否完全处于圆锥区域（比如月全食中的月球完全进入地球本影）
    若返回值 < 0 则说明“完全在锥内”
    """
    cone_tip_to_body = body_pos - cone_tip
    proj_point = cone_tip + np.dot(cone_tip_to_body, cone_axis) * cone_axis / np.linalg.norm(cone_axis)**2
    perpendicular_distance = np.linalg.norm(body_pos - proj_point)
    horizontal_distance = np.linalg.norm(cone_tip - proj_point)
    critical_distance = np.tan(half_angle) * horizontal_distance - body_radius / np.cos(half_angle)
    return perpendicular_distance - critical_distance

def near_alignment_cross(es_vec, em_vec):
    """
    用于判断日月地是否几乎成一直线
    cross_prod ~ 0 时说明向量近似平行
    threshold 是经验阈值，用于限制误差或近似判断
    """
    cross_prod = np.cross(es_vec, em_vec)
    cross_mag = np.linalg.norm(cross_prod)
    threshold = 1.49e11 * 3.84e8 * MAX_SIN_ANGLE
    return cross_mag < threshold

# ------------------- 初始化 -------------------
t0 = Time(START_DATE, scale=TIME_SCALE)
print(f"模拟开始时间: {t0.iso}")
print(f"模拟天体: {', '.join(BODIES)}")
print(f"模拟时长: {SIMULATION_YEARS} 年")

# 准备初始状态向量
init_state = []
for body in BODIES:
    pos, vel = get_body_barycentric_posvel(body, t0)
    init_state.extend([pos.get_xyz().to(u.m).value, vel.xyz.to(u.m/u.s).value])

y0 = np.hstack(init_state)
gm_values = np.array([GM_DICT[body] for body in BODIES])
j2_array = np.array([J2_DICT[body] for body in BODIES])
r_array = np.array([RADIUS_DICT[body] for body in BODIES])

# 索引
sun_idx = BODIES.index("sun")
earth_idx = BODIES.index("earth")
moon_idx = BODIES.index("moon")

# 天体半径（米）
R_sun = 6.9634e8
R_earth = 6.371e6
R_moon = 1.7374e6

# ------------------- Event 函数工厂 -------------------
def eclipse_event_factory(event_type):
    def eclipse_event(t, y):
        state = y.reshape(len(BODIES), 6)
        sp = state[sun_idx, :3]
        ep = state[earth_idx, :3]
        mp = state[moon_idx, :3]

        es_vec = sp - ep
        em_vec = mp - ep

        # 简单判断地球-太阳-月球是否在同一侧
        if np.dot(es_vec, em_vec) < 0 and 'solar' in event_type:
            return 1
        if np.dot(es_vec, em_vec) > 0 and 'lunar' in event_type:
            return 1

        # 若三者不近似成一线，也不触发事件
        if not near_alignment_cross(es_vec, em_vec):
            return 1

        # 根据 event_type 判断进入/离开圆锥区域情况
        if 'solar' in event_type:
            # 日食，以月球为“遮挡者”
            ms_vec = sp - mp
            dist_ms = np.linalg.norm(ms_vec)
            umbra_dir = -ms_vec / dist_ms
            umbra_angle = np.arcsin((R_sun - R_moon) / dist_ms)
            penumbra_angle = np.arcsin((R_sun + R_moon) / dist_ms)
            umbra_tip = sp + R_sun / np.sin(umbra_angle) * umbra_dir
            penumbra_tip = sp + R_sun / np.sin(penumbra_angle) * umbra_dir

            if event_type == 'solar_total':
                return np.linalg.norm(umbra_tip - ep) - R_earth
            elif event_type == 'solar_partial':
                return body_in_cone(ep, R_earth, penumbra_tip, umbra_dir, penumbra_angle)
            elif event_type == 'solar_annular':
                # 假设 umbra_tip 到达不到地球时可能发生环食
                if np.linalg.norm(umbra_tip - ep) < R_earth:
                    return 1
                return body_in_cone(ep, R_earth, umbra_tip, umbra_dir, umbra_angle)

        elif 'lunar' in event_type:
            # 月食，以地球为“遮挡者”
            dist_es = np.linalg.norm(es_vec)
            earth_umbra_dir = -es_vec / dist_es
            earth_umbra_angle = np.arcsin((R_sun - R_earth) / dist_es)
            earth_penumbra_angle = np.arcsin((R_sun + R_earth) / dist_es)
            earth_umbra_tip = ep + R_earth / np.tan(earth_umbra_angle) * earth_umbra_dir
            earth_penumbra_tip = ep - R_earth / np.tan(earth_penumbra_angle) * earth_umbra_dir

            if event_type == 'lunar_total':
                return body_totally_in_cone(mp, R_moon, earth_umbra_tip, earth_umbra_dir, earth_umbra_angle)
            elif event_type == 'lunar_partial':
                return body_in_cone(mp, R_moon, earth_umbra_tip, earth_umbra_dir, earth_umbra_angle)
            elif event_type == 'lunar_penumbral':
                return body_in_cone(mp, R_moon, earth_penumbra_tip, earth_umbra_dir, earth_penumbra_angle)

        return 1

    eclipse_event.terminal = False
    return eclipse_event

# 所有事件函数
events = [
    eclipse_event_factory("solar_partial"),
    eclipse_event_factory("solar_annular"),
    eclipse_event_factory("solar_total"),
    eclipse_event_factory("lunar_penumbral"),
    eclipse_event_factory("lunar_partial"),
    eclipse_event_factory("lunar_total"),
]

# ------------------- 加速度计算（向量化） -------------------
def calculate_accelerations_vectorized(positions, velocities, gm_values):
    # ------------------- 1) 牛顿引力 -------------------
    r_ij = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
    
    dist_sq = np.sum(r_ij**2, axis=2) + 1e-10             # (n, n)
    np.fill_diagonal(dist_sq, np.inf)                     # 避免除零，自身不参与
    
    dist = np.sqrt(dist_sq)                               # r
    dist_cubed = dist_sq * dist                           # r^3

    factors_newton = gm_values[None, :, None] / dist_cubed[:, :, None]  # (n, n, 1)
    acc_newton = -np.sum(r_ij * factors_newton, axis=1)                 # (n, 3)
    acc_total = acc_newton

    # ------------------- 2) 简化 1PN 修正 -------------------
    if RELATIVITY_SWITCH:
        v_ij = velocities[:, None, :] - velocities[None, :, :]       # (n, n, 3)
        v2_ij = np.sum(v_ij**2, axis=2)                               # (n, n)
        r_dot_v_ij = np.sum(r_ij * v_ij, axis=2)                      # (n, n)

        c2 = C_LIGHT**2
        factor_1pn = gm_values[None, :] / (c2 * dist**3)              # (n, n)
        factor_1pn = factor_1pn[:, :, None]                           # (n, n, 1)

        GMj_over_r = gm_values[None, :] / dist                        # (n, n)
        tmp = 4.0 * GMj_over_r - v2_ij                                # (n, n)
        tmp = tmp[:, :, None]                                         # (n, n, 1)

        term1 = tmp * r_ij
        term1 += (4.0 * r_dot_v_ij[:, :, None]) * v_ij

        a_1pn_ij = factor_1pn * term1
        acc_1pn = np.sum(a_1pn_ij, axis=1)                            # (n, 3)
        acc_total += acc_1pn

    # ------------------- 3) J2 项 -------------------
    if J2_SWITCH:
        z_ij = r_ij[..., 2]                                           # (n, n)
        bracket_ij = 5.0 * (z_ij**2 / dist_sq) - 1.0                  # (n, n)

        J2_j = j2_array[None, :]                                      # (1, n)
        GM_j = gm_values[None, :]                                     # (1, n)
        R_j2 = r_array[None, :]**2                                    # (1, n)

        factor_j2_ij = (1.5 * J2_j * GM_j * R_j2) / (dist**5)         # (n, n)
        factor_j2_ij *= bracket_ij                                    # (n, n)

        a_j2_ij = factor_j2_ij[..., None] * r_ij                      # (n, n, 3)
        acc_j2 = np.sum(a_j2_ij, axis=1)                              # (n, 3)
        acc_total += acc_j2

    return acc_total

# ------------------- 微分方程 -------------------
def derivatives(t, y):
    n = len(BODIES)
    state = y.reshape((n, 6))
    pos, vel = state[:, :3], state[:, 3:]

    # 注意，这里 gm_values 不变
    
    acc = calculate_accelerations_vectorized(pos, vel, gm_values)

    dydt = np.zeros_like(state)
    dydt[:, :3] = vel
    dydt[:, 3:] = acc
    return dydt.flatten()

# ------------------- 开始积分 -------------------
total_time = SIMULATION_YEARS * SECONDS_PER_YEAR
print(f"开始积分计算，使用 {SOLVER_METHOD} 方法...")

start_time = time.time()

# 关键：将 dense_output 设为 True，用于后续插值获取解
sol = solve_ivp(
    derivatives,
    [0, total_time],
    y0,
    method=SOLVER_METHOD,
    rtol=SOLVER_RTOL,
    atol=SOLVER_ATOL,
    events=events,
    dense_output=True,
    max_step=MAX_STEP_TIME,
)

elapsed = time.time() - start_time
print(f"积分完成，用时 {elapsed:.2f} 秒")

# ------------------- 输出日食月食事件结果 -------------------
dt_start = datetime.strptime(t0.utc.iso, "%Y-%m-%d %H:%M:%S.%f")
event_names = [
    "0° Solar Partial",
    "1° Solar Annular",
    "2° Solar Total",
    "3° Lunar Penumbral",
    "4° Lunar Partial",
    "5° Lunar Total",
]
eclipse_events = {}

for idx, (name, times) in enumerate(zip(event_names, sol.t_events)):
    # 每两次事件为一对，代表开始 ~ 结束
    pairs = list(zip(times[::2], times[1::2]))
    print(f"\n{name} ({len(pairs)} 次):")
    eclipse_events[name] = []
    for start, end in pairs:
        t1 = (dt_start + timedelta(seconds=start)).replace(microsecond=0)
        t2 = (dt_start + timedelta(seconds=end)).replace(microsecond=0)
        duration = str(t2 - t1)
        midpoint = (t1 + (t2 - t1)/2).replace(microsecond=0)  # 去除微秒
        print(f"{t1} ~ {t2} | dur: {duration} | mid: {midpoint}")
        eclipse_events[name].append({
            "start": str(t1), 
            "end": str(t2), 
            "duration": duration,
            "midpoint": str(midpoint)  # 将中点也存入JSON
        })

# 保存为 JSON（若不需要可去掉）
with open("eclipse_events.json", "w", encoding="utf-8") as f:
    json.dump(eclipse_events, f, indent=2, ensure_ascii=False)
print("日食月食事件已保存至 eclipse_events.json")

print(f"模拟完成，总用时 {time.time() - start_time:.2f} 秒")

import math
print("\n开始误差分析（多天体分量误差图）...")

evaluation_times = np.arange(0, total_time, ERROR_EVAL_INTERVAL)
n_bodies = len(BODIES)
num_targets = len(ERROR_ANALYSIS_BODIES)

# 准备画布尺寸（每行最多3个）
cols = 3
rows = math.ceil(num_targets / cols)
fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
axes = axes.flatten()

for i, body_name in enumerate(ERROR_ANALYSIS_BODIES):
    idx = BODIES.index(body_name)
    print(f"分析 {body_name.capitalize()} 的位置误差...")

    # 数值模拟位置
    sim_pos = np.zeros((len(evaluation_times), 3))
    for j, t_eval in enumerate(evaluation_times):
        full_state = sol.sol(t_eval)
        sim_pos[j, :] = full_state[idx * 6: idx * 6 + 3]

    # 星历位置
    std_pos = np.zeros_like(sim_pos)
    for j, t_eval in enumerate(evaluation_times):
        current_time = t0 + t_eval * u.s
        pos_std, _ = get_body_barycentric_posvel(body_name, current_time)
        std_pos[j, :] = pos_std.xyz.to(u.m).value

    # 误差计算（单位：km）
    diff = (sim_pos - std_pos) / 1000.0
    abs_error = np.linalg.norm(diff, axis=1)  # 绝对误差
    error_x = diff[:, 0]
    error_y = diff[:, 1]
    error_z = diff[:, 2]
    time_years = evaluation_times / SECONDS_PER_YEAR

    # 绘图
    ax = axes[i]
    ax.plot(time_years, abs_error, color='black', label='Absolute Error', linewidth='0.5')
    ax.plot(time_years, error_x, color='red', label='X Error', linewidth='0.5')
    ax.plot(time_years, error_y, color='green', label='Y Error', linewidth='0.5')
    ax.plot(time_years, error_z, color='blue', label='Z Error', linewidth='0.5')

    ax.set_title(f"{body_name.capitalize()}")
    ax.set_xlabel("Years")
    ax.set_ylabel("Error (km)")
    ax.grid(True)

# 清理多余子图
for j in range(num_targets, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Position Error Components vs DE440", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("all_bodies_error_components.png")
plt.show()

print("误差图已保存为 all_bodies_error_components.png")