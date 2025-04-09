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
    BODIES, GM_DICT, START_DATE, TIME_SCALE,
    SIMULATION_YEARS, SECONDS_PER_YEAR,
    SOLVER_METHOD, SOLVER_RTOL, SOLVER_ATOL, MAX_STEP_TIME, ERROR_EVAL_INTERVAL
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
    threshold = 1.49e11 * 3.84e8 * 0.02
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
def calculate_accelerations_vectorized(positions, gm_values):
    """
    利用 NumPy 广播一次性计算所有天体之间的万有引力加速度。
    positions: shape (n, 3)
    gm_values: shape (n,)
    返回结果: shape (n, 3)
    """
    # r_ij[i,j] = positions[i] - positions[j]
    r_ij = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
    dist_sq = np.sum(r_ij**2, axis=2) + 1e-10  # 防止除0
    np.fill_diagonal(dist_sq, np.inf)  # 自身不计算
    dist_cubed = dist_sq * np.sqrt(dist_sq)
    
    factors = gm_values[None, :, None] / dist_cubed[:, :, None]  # (n, n, 1)
    acc = -np.sum(r_ij * factors, axis=1)
    return acc

# ------------------- 微分方程 -------------------
def derivatives(t, y):
    n = len(BODIES)
    state = y.reshape((n, 6))
    pos, vel = state[:, :3], state[:, 3:]
    acc = calculate_accelerations_vectorized(pos, gm_values)
    dydt = np.zeros_like(state)
    dydt[:, :3], dydt[:, 3:] = vel, acc
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
        print(f"{t1} ~ {t2} | {duration}")
        eclipse_events[name].append({"start": str(t1), "end": str(t2), "duration": duration})

# 保存为 JSON（若不需要可去掉）
with open("eclipse_events.json", "w", encoding="utf-8") as f:
    json.dump(eclipse_events, f, indent=2, ensure_ascii=False)
print("日食月食事件已保存至 eclipse_events.json")

print(f"模拟完成，总用时 {time.time() - start_time:.2f} 秒")


# ------------------- 误差分析（地球） -------------------
print("\n开始误差分析（仅地球）...")

# 1. 定期生成需要检查的时间点（秒）
evaluation_times = np.arange(0, total_time, ERROR_EVAL_INTERVAL)

# 2. 在这些时间点上，通过 sol.sol(t_eval) 获得数值解中地球的位置
#   sol.sol(...) 返回形状为 (6*body_count, len(t_eval)) 的数组
n_bodies = len(BODIES)
earth_pos_sim = np.zeros((len(evaluation_times), 3))

for i, t_eval in enumerate(evaluation_times):
    # 整个系统状态(6*n_bodies,) -> 每个天体 6 个分量
    full_state = sol.sol(t_eval)  # shape = (6*n_bodies,)
    # 地球的索引范围
    start_idx = earth_idx * 6
    end_idx = start_idx + 3
    earth_pos_sim[i, :] = full_state[start_idx:end_idx]

# 3. 获取标准星历下的地球位置（de440），并比较两者差距
earth_pos_std = np.zeros((len(evaluation_times), 3))

for i, t_eval in enumerate(evaluation_times):
    # 当前的绝对时刻（Astropy Time）
    current_time = t0 + t_eval * u.s
    pos_std, _ = get_body_barycentric_posvel('earth', current_time)
    earth_pos_std[i, :] = pos_std.xyz.to(u.m).value

# 4. 计算误差（欧几里得距离）
diff = earth_pos_sim - earth_pos_std
errors = np.linalg.norm(diff, axis=1)  # 每个时刻的三维差值的模
errors_km = errors / 1000.0

# 5. 绘图：横轴是年，纵轴是误差（km）
time_years = evaluation_times / SECONDS_PER_YEAR

plt.figure()
plt.plot(time_years, errors_km)
plt.xlabel("Time (years)")
plt.ylabel("Position Error (km)")
plt.title("Earth Position Error vs. DE440")
plt.grid(True)
plt.show()

print("误差分析结束！")
