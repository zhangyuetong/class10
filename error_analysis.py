import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u
import math
from datetime import datetime

def run_error_analysis(
    sol, BODIES, GM_DICT, ERROR_ANALYSIS_BODIES,
    SECONDS_PER_YEAR, ERROR_EVAL_INTERVAL,
    t0_str, t0_scale
):
    print("\n开始误差分析（多天体分量误差图，包括质心）...")

    t0 = Time(t0_str, scale=t0_scale)
    total_time = sol.t[-1]
    evaluation_times = np.arange(0, total_time, ERROR_EVAL_INTERVAL)
    time_years = evaluation_times / SECONDS_PER_YEAR

    n_bodies = len(BODIES)
    num_targets = len(ERROR_ANALYSIS_BODIES) + 1

    cols = 3
    rows = math.ceil(num_targets / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    diff_dict = {}

    for i, body_name in enumerate(ERROR_ANALYSIS_BODIES):
        idx = BODIES.index(body_name)
        print(f"分析 {body_name.capitalize()} 的位置误差...")

        sim_pos = np.array([sol.sol(t)[idx * 6: idx * 6 + 3] for t in evaluation_times])
        std_pos = np.array([
            get_body_barycentric_posvel(body_name, t0 + t * u.s)[0].xyz.to(u.m).value
            for t in evaluation_times
        ])

        diff = (sim_pos - std_pos) / 1000.0
        diff_dict[body_name] = diff

        abs_error = np.linalg.norm(diff, axis=1)
        error_x, error_y, error_z = diff.T

        ax = axes[i]
        ax.plot(time_years, abs_error, label='Abs', linewidth=0.5, color='black')
        ax.plot(time_years, error_x,  label='X', linewidth=0.5, color='red')
        ax.plot(time_years, error_y,  label='Y', linewidth=0.5, color='green')
        ax.plot(time_years, error_z,  label='Z', linewidth=0.5, color='blue')
        ax.set_title(f"{body_name.capitalize()}")
        ax.set_xlabel("Years")
        ax.set_ylabel("Error (km)")
        ax.grid(True)

    print("分析系统质心的位置误差...")

    mass_arr = np.array([GM_DICT[body] for body in BODIES]) / 6.67430e-11
    sim_cm_pos = []
    std_cm_pos = []

    for t in evaluation_times:
        state = sol.sol(t).reshape((n_bodies, 6))
        sim_cm = np.average(state[:, :3], axis=0, weights=mass_arr)
        sim_cm_pos.append(sim_cm)

        std_positions = [
            get_body_barycentric_posvel(body, t0 + t * u.s)[0].xyz.to(u.m).value
            for body in BODIES
        ]
        std_cm = np.average(std_positions, axis=0, weights=mass_arr)
        std_cm_pos.append(std_cm)

    sim_cm_pos = np.array(sim_cm_pos)
    std_cm_pos = np.array(std_cm_pos)
    diff_cm = (sim_cm_pos - std_cm_pos) / 1000.0
    diff_dict["barycenter"] = diff_cm

    abs_error_cm = np.linalg.norm(diff_cm, axis=1)
    error_x_cm, error_y_cm, error_z_cm = diff_cm.T

    ax = axes[len(ERROR_ANALYSIS_BODIES)]
    ax.plot(time_years, abs_error_cm, label='Abs', linewidth=0.5, color='black')
    ax.plot(time_years, error_x_cm,   label='X',   linewidth=0.5, color='red')
    ax.plot(time_years, error_y_cm,   label='Y',   linewidth=0.5, color='green')
    ax.plot(time_years, error_z_cm,   label='Z',   linewidth=0.5, color='blue')
    ax.set_title("Barycenter")
    ax.set_xlabel("Years")
    ax.set_ylabel("Error (km)")
    ax.grid(True)

    for j in range(num_targets, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Position Error Components vs DE440", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("./report/all_bodies_error_components.png")
    #plt.show()
    print("总误差图已保存为 report/all_bodies_error_components.png")

    print("\n开始绘制『减去质心误差』的新图...")
    fig2, axes2 = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes2 = axes2.flatten()

    for i, body_name in enumerate(ERROR_ANALYSIS_BODIES):
        diff_rel = diff_dict[body_name] - diff_cm
        abs_error_rel = np.linalg.norm(diff_rel, axis=1)
        err_x_rel, err_y_rel, err_z_rel = diff_rel.T

        ax2 = axes2[i]
        ax2.plot(time_years, abs_error_rel, linewidth=0.5, label='Abs', color='black')
        ax2.plot(time_years, err_x_rel, linewidth=0.5, label='X', color='red')
        ax2.plot(time_years, err_y_rel, linewidth=0.5, label='Y', color='green')
        ax2.plot(time_years, err_z_rel, linewidth=0.5, label='Z', color='blue')
        ax2.set_title(f"{body_name.capitalize()} – Δr − Δr_CM")
        ax2.set_xlabel("Years")
        ax2.set_ylabel("Residual Error (km)")
        ax2.grid(True)

    for j in range(len(ERROR_ANALYSIS_BODIES), len(axes2)):
        fig2.delaxes(axes2[j])

    fig2.suptitle("Position Error minus Barycenter Error", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("./report/error_minus_barycenter.png")
    #plt.show()
    print("新图已保存为 report/error_minus_barycenter.png")

import matplotlib.animation as animation

def plot_sun_earth_moon_three_views_animation(sol, BODIES, t0_str, t0_scale, SECONDS_PER_YEAR):
    print("\n开始绘制日-地-月三体三视图动图...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    limits = 2e11  # 设置坐标范围 ±2e11 米
    view_labels = ['X-Y Plane', 'X-Z Plane', 'Y-Z Plane']
    coord_pairs = [(0, 1), (0, 2), (1, 2)]

    sun_dots = []
    earth_dots = []
    moon_dots = []

    for ax, label in zip(axes, view_labels):
        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.set_aspect('equal')
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Position (m)')
        ax.set_title(label)
        sun_dot, = ax.plot([], [], 'yo', markersize=12, label='Sun')
        earth_dot, = ax.plot([], [], 'bo', markersize=6, label='Earth')
        moon_dot, = ax.plot([], [], 'ko', markersize=3, label='Moon')
        ax.legend()
        sun_dots.append(sun_dot)
        earth_dots.append(earth_dot)
        moon_dots.append(moon_dot)

    t0 = Time(t0_str, scale=t0_scale)
    total_time = sol.t[-1]
    frames = np.linspace(0, total_time, 300)  # 取300帧

    sun_idx = BODIES.index("sun")
    earth_idx = BODIES.index("earth")
    moon_idx = BODIES.index("moon")

    def init():
        for sun_dot, earth_dot, moon_dot in zip(sun_dots, earth_dots, moon_dots):
            sun_dot.set_data([], [])
            earth_dot.set_data([], [])
            moon_dot.set_data([], [])
        return sun_dots + earth_dots + moon_dots

    def update(frame):
        state = sol.sol(frame)
        sun_pos = state[sun_idx*6 : sun_idx*6+3]
        earth_pos = state[earth_idx*6 : earth_idx*6+3]
        moon_pos = state[moon_idx*6 : moon_idx*6+3]

        for (i, (x_idx, y_idx)) in enumerate(coord_pairs):
            sun_dots[i].set_data(sun_pos[x_idx], sun_pos[y_idx])
            earth_dots[i].set_data(earth_pos[x_idx], earth_pos[y_idx])
            moon_dots[i].set_data(moon_pos[x_idx], moon_pos[y_idx])

        return sun_dots + earth_dots + moon_dots

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)

    ani.save("./report/sun_earth_moon_three_views.gif", writer='pillow')
    print("三视图动图已保存为 report/sun_earth_moon_three_views.gif")
    # plt.show()
