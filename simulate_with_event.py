#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
simulate_with_event.py

行星-月食/日食与误差分析综合模拟
J2 扁率项已按真实自转轴方向修正
"""

# ------------------- 基础依赖 -------------------
import math
import time
import json
from datetime import timedelta, datetime

import numpy as np
from scipy.integrate import solve_ivp

from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
import astropy.units as u

import matplotlib.pyplot as plt          # 仅用于误差分析绘图
from numba import njit, float64, int64    # Numba 加速

# ------------------- 配置与全局常量 -------------------
solar_system_ephemeris.set('de440')       # JPL DE440 星历

from config import (
    BODIES, GM_DICT,
    START_DATE, TIME_SCALE, SIMULATION_YEARS, SECONDS_PER_YEAR,
    SOLVER_METHOD, SOLVER_RTOL, SOLVER_ATOL, MAX_STEP_TIME,
    RELATIVITY_SWITCH, C_LIGHT,
    J2_SWITCH, J2_DICT, RADIUS_DICT,
    SPIN_AXIS_DICT,                         # ★ 新增：行星自转轴方向
    MAX_SIN_ANGLE,
    ERROR_EVAL_INTERVAL, ERROR_ANALYSIS_BODIES
)

# ------------------- 初始化天体初值 -------------------
t0 = Time(START_DATE, scale=TIME_SCALE)
print(f"模拟开始时间: {t0.iso}")
print(f"模拟天体    : {', '.join(BODIES)}")
print(f"模拟时长    : {SIMULATION_YEARS} 年")

init_state = []
for body in BODIES:
    pos, vel = get_body_barycentric_posvel(body, t0)
    init_state.extend([pos.get_xyz().to(u.m).value, vel.xyz.to(u.m/u.s).value])

y0 = np.hstack(init_state).astype(np.float64)             # 初始状态向量
gm_values     = np.array([GM_DICT[body]     for body in BODIES], dtype=np.float64)
j2_array      = np.array([J2_DICT[body]     for body in BODIES], dtype=np.float64)
radius_arr    = np.array([RADIUS_DICT[body] for body in BODIES], dtype=np.float64)
spin_axis_arr = np.array([SPIN_AXIS_DICT[body] for body in BODIES], dtype=np.float64)

# 重要索引
sun_idx   = BODIES.index("sun")
earth_idx = BODIES.index("earth")
moon_idx  = BODIES.index("moon")

# 天体均质半径(米)——用于食相判定
R_sun   = 6.9634e8
R_earth = 6.371e6
R_moon  = 1.7374e6

# ------------------- ❶ 预编译版加速度核心 -------------------
@njit(cache=True, fastmath=True)
def calc_acc_numba(pos, vel,
                   gm_values, j2_array, radius_arr, spin_axis_arr,
                   relativity, c_light, use_j2):
    """
    Numba-JIT 加速度计算（O(n²) 双循环，适合天体数量 ≤ 数十）
    采用真实自转轴方向计算 J2 扁率项
    参数均为 *裸 numpy.ndarray*，确保完全可编译
    """
    n = pos.shape[0]
    acc = np.zeros((n, 3), dtype=np.float64)

    c2 = c_light * c_light

    for i in range(n):
        for j in range(n):
            if i == j:
                continue          # 自身引力忽略

            # -------- Newton 引力 --------
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]

            dist_sq   = dx*dx + dy*dy + dz*dz + 1e-10     # 防零溢出
            dist      = math.sqrt(dist_sq)
            dist_cube = dist_sq * dist

            nx = dx / dist
            ny = dy / dist
            nz = dz / dist

            factor = gm_values[j] / dist_cube
            acc[i, 0] -= factor * dx
            acc[i, 1] -= factor * dy
            acc[i, 2] -= factor * dz

            # -------- 1 PN 相对论修正 --------
            if relativity:
                dvx = vel[i, 0] - vel[j, 0]
                dvy = vel[i, 1] - vel[j, 1]
                dvz = vel[i, 2] - vel[j, 2]

                vi2 = vel[i, 0]**2 + vel[i, 1]**2 + vel[i, 2]**2
                vj2 = vel[j, 0]**2 + vel[j, 1]**2 + vel[j, 2]**2
                vi_dot_vj = vel[i, 0]*vel[j, 0] + vel[i, 1]*vel[j, 1] + vel[i, 2]*vel[j, 2]
                n_dot_vj = nx*vel[j, 0] + ny*vel[j, 1] + nz*vel[j, 2]
                n_dot_vi = nx*vel[i, 0] + ny*vel[i, 1] + nz*vel[i, 2]

                GMj_over_r = gm_values[j] / dist

                prefac = gm_values[j] / (c2 * dist_cube)

                dir_term = (4 * GMj_over_r - vi2 - 2 * vj2 + 4 * vi_dot_vj + 1.5 * n_dot_vj * n_dot_vj)
                vel_term = (4 * n_dot_vi - 3 * n_dot_vj)

                acc[i, 0] += prefac * (dir_term * dx + vel_term * dvx * dist)
                acc[i, 1] += prefac * (dir_term * dy + vel_term * dvy * dist)
                acc[i, 2] += prefac * (dir_term * dz + vel_term * dvz * dist)

            # -------- J2 扁率项（按真实自转轴） --------
            if use_j2 and j2_array[j] != 0.0:
                axj = spin_axis_arr[j, 0]
                ayj = spin_axis_arr[j, 1]
                azj = spin_axis_arr[j, 2]

                rk = dx*axj + dy*ayj + dz*azj      # r·k
                fac_j2 = 1.5 * j2_array[j] * gm_values[j] * (radius_arr[j]**2) / (dist**5)
                common = 5.0 * (rk*rk) / dist_sq - 1.0

                ax = fac_j2 * (common * dx - 2.0 * rk * axj)
                ay = fac_j2 * (common * dy - 2.0 * rk * ayj)
                az = fac_j2 * (common * dz - 2.0 * rk * azj)

                acc[i, 0] += ax
                acc[i, 1] += ay
                acc[i, 2] += az
                
                mass_ratio = gm_values[i] / gm_values[j]
                acc[j, 0] -= mass_ratio * ax
                acc[j, 1] -= mass_ratio * ay
                acc[j, 2] -= mass_ratio * az

    return acc


# ------------------- ❷ SciPy ODE 回调 -------------------
def derivatives(t, y):
    """
    SciPy ODE 接口：拆分状态→调用已预编译的 calc_acc_numba→合并导数
    """
    n = len(BODIES)
    state = y.reshape((n, 6))
    pos   = state[:, :3]
    vel   = state[:, 3:]

    acc = calc_acc_numba(
        pos, vel,
        gm_values, j2_array, radius_arr, spin_axis_arr,
        RELATIVITY_SWITCH, C_LIGHT, J2_SWITCH
    )

    dydt = np.empty_like(state)
    dydt[:, :3] = vel
    dydt[:, 3:] = acc
    return dydt.flatten()

# ------------------- 几何辅助函数（保持原逻辑） -------------------
def body_in_cone(body_pos, body_radius, cone_tip, cone_axis, half_angle):
    """判断天体是否部分落入圆锥（负值⇒在锥内）"""
    cone_tip_to_body = body_pos - cone_tip
    proj_point = cone_tip + np.dot(cone_tip_to_body, cone_axis) * cone_axis / np.linalg.norm(cone_axis)**2
    perpendicular_distance = np.linalg.norm(body_pos - proj_point)
    horizontal_distance = np.linalg.norm(cone_tip - proj_point)
    critical_distance = np.tan(half_angle) * horizontal_distance + body_radius / np.cos(half_angle)
    return perpendicular_distance - critical_distance

def body_totally_in_cone(body_pos, body_radius, cone_tip, cone_axis, half_angle):
    """判断天体是否完全落入圆锥（负值⇒完全在锥内）"""
    cone_tip_to_body = body_pos - cone_tip
    proj_point = cone_tip + np.dot(cone_tip_to_body, cone_axis) * cone_axis / np.linalg.norm(cone_axis)**2
    perpendicular_distance = np.linalg.norm(body_pos - proj_point)
    horizontal_distance = np.linalg.norm(cone_tip - proj_point)
    critical_distance = np.tan(half_angle) * horizontal_distance - body_radius / np.cos(half_angle)
    return perpendicular_distance - critical_distance

def near_alignment_cross(es_vec, em_vec):
    """|ES×EM| 足够小 ⇒ 日-地-月近似共线"""
    cross_mag = np.linalg.norm(np.cross(es_vec, em_vec))
    threshold = 1.49e11 * 3.84e8 * MAX_SIN_ANGLE  # 经验阈值
    return cross_mag < threshold

# ------------------- ❸ 日月食事件工厂（保持原逻辑） -------------------
def eclipse_event_factory(event_type):
    """根据类别生成 SciPy 事件函数，判定进入/离开食区"""
    def eclipse_event(t, y):
        state = y.reshape(len(BODIES), 6)
        sp = state[sun_idx, :3]
        ep = state[earth_idx, :3]
        mp = state[moon_idx, :3]

        es_vec = sp - ep
        em_vec = mp - ep

        # 简易同侧判定
        if np.dot(es_vec, em_vec) < 0 and 'solar' in event_type:
            return 1
        if np.dot(es_vec, em_vec) > 0 and 'lunar' in event_type:
            return 1

        # 若不近似共线，事件不触发
        if not near_alignment_cross(es_vec, em_vec):
            return 1

        # ---------- 日食 ----------
        if 'solar' in event_type:
            ms_vec = sp - mp
            dist_ms = np.linalg.norm(ms_vec)
            umbra_dir    = -ms_vec / dist_ms
            umbra_angle  = math.asin((R_sun - R_moon) / dist_ms)
            penum_angle  = math.asin((R_sun + R_moon) / dist_ms)
            umbra_tip    = sp + R_sun / math.sin(umbra_angle) * umbra_dir
            penum_tip    = sp + R_sun / math.sin(penum_angle) * umbra_dir

            if event_type == 'solar_total':
                return np.linalg.norm(umbra_tip - ep) - R_earth
            elif event_type == 'solar_partial':
                return body_in_cone(ep, R_earth, penum_tip, umbra_dir, penum_angle)
            elif event_type == 'solar_annular':
                if np.linalg.norm(umbra_tip - ep) < R_earth:
                    return 1
                return body_in_cone(ep, R_earth, umbra_tip, umbra_dir, umbra_angle)

        # ---------- 月食 ----------
        if 'lunar' in event_type:
            dist_es = np.linalg.norm(es_vec)
            umb_dir = -es_vec / dist_es
            umb_ang = math.asin((R_sun - R_earth) / dist_es)
            pen_ang = math.asin((R_sun + R_earth) / dist_es)
            umb_tip = ep + R_earth / math.tan(umb_ang) * umb_dir
            pen_tip = ep - R_earth / math.tan(pen_ang) * umb_dir

            if event_type == 'lunar_total':
                return body_totally_in_cone(mp, R_moon, umb_tip, umb_dir, umb_ang)
            elif event_type == 'lunar_partial':
                return body_in_cone(mp, R_moon, umb_tip, umb_dir, umb_ang)
            elif event_type == 'lunar_penumbral':
                return body_in_cone(mp, R_moon, pen_tip, umb_dir, pen_ang)

        return 1  # 默认非触发

    eclipse_event.terminal = False
    return eclipse_event

events = [
    eclipse_event_factory("solar_partial"),
    eclipse_event_factory("solar_annular"),
    eclipse_event_factory("solar_total"),
    eclipse_event_factory("lunar_penumbral"),
    eclipse_event_factory("lunar_partial"),
    eclipse_event_factory("lunar_total"),
]

# ------------------- ❹ 数值积分 -------------------
total_time = SIMULATION_YEARS * SECONDS_PER_YEAR
print(f"\n开始积分计算（{SOLVER_METHOD}），目标区间: 0 – {total_time:.3e} s")

tic = time.time()
sol = solve_ivp(
    derivatives,                      # 已封装的导数函数
    [0.0, total_time],
    y0,
    method=SOLVER_METHOD,
    rtol=SOLVER_RTOL,
    atol=SOLVER_ATOL,
    events=events,
    dense_output=True,
    max_step=MAX_STEP_TIME,
)
toc = time.time()
print(f"积分完成，用时 {toc - tic:.2f} 秒")

# ------------------- ❺ 食相事件输出 -------------------
dt_start = datetime.strptime(t0.utc.iso, "%Y-%m-%d %H:%M:%S.%f")
event_titles = [
    "0° Solar Partial",
    "1° Solar Annular",
    "2° Solar Total",
    "3° Lunar Penumbral",
    "4° Lunar Partial",
    "5° Lunar Total",
]
eclipse_events = {}

for title, times in zip(event_titles, sol.t_events):
    pairs = list(zip(times[::2], times[1::2]))  # 每两次为一对：开始→结束
    print(f"\n{title} ({len(pairs)} 次):")
    eclipse_events[title] = []
    for t_start, t_end in pairs:
        dt1 = (dt_start + timedelta(seconds=float(t_start))).replace(microsecond=0)
        dt2 = (dt_start + timedelta(seconds=float(t_end))).replace(microsecond=0)
        duration  = dt2 - dt1
        midpoint  = (dt1 + duration / 2).replace(microsecond=0)

        print(f"{dt1}  ~  {dt2} | dur: {duration} | mid: {midpoint}")

        eclipse_events[title].append({
            "start": str(dt1),
            "end": str(dt2),
            "duration": str(duration),
            "midpoint": str(midpoint)
        })

with open("./report/eclipse_events.json", "w", encoding="utf-8") as f_json:
    json.dump(eclipse_events, f_json, indent=2, ensure_ascii=False)
print("\n✅ 日月食事件已保存到 report/eclipse_events.json")

# ------------------- ❻ 误差分析 -------------------
from error_analysis import run_error_analysis

run_error_analysis(
    sol=sol,
    BODIES=BODIES,
    GM_DICT=GM_DICT,
    ERROR_ANALYSIS_BODIES=ERROR_ANALYSIS_BODIES,
    SECONDS_PER_YEAR=SECONDS_PER_YEAR,
    ERROR_EVAL_INTERVAL=ERROR_EVAL_INTERVAL,
    t0_str=START_DATE,
    t0_scale=TIME_SCALE
)

print(f"\n模拟整体完成，总耗时 {time.time() - tic:.2f} 秒（含误差分析）")
