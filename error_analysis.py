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
    plt.savefig("all_bodies_error_components.png")
    #plt.show()
    print("总误差图已保存为 all_bodies_error_components.png")

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
    plt.savefig("error_minus_barycenter.png")
    #plt.show()
    print("新图已保存为 error_minus_barycenter.png")
