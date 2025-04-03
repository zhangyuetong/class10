import numpy as np 
from scipy.integrate import solve_ivp
import json
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.constants import G
import time

# 导入配置文件
from config import (
    BODIES, MASSES, START_DATE, TIME_SCALE, 
    SIMULATION_YEARS, SECONDS_PER_YEAR, OUTPUT_INTERVAL,
    SOLVER_METHOD, SOLVER_RTOL, SOLVER_ATOL,
)

# 导入日食月食预测模块
from eclipse_prediction import predict_eclipses

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

total_elapsed = time.time() - start_time
print(f"模拟完成！总用时 {total_elapsed:.2f} 秒")

# ---------------------------------------------------------
# 以下代码用于对比数值积分结果与 Astropy 内置星历 (JPL) 的位置/速度差异
# ---------------------------------------------------------

# 1. 选取需要对比的采样时刻的下标（这里演示每隔 500 个点取一个，也可以根据需要调整）
compare_indices = np.arange(0, len(t_eval), 500)

# 准备存储结果的容器，结构随意，这里示例用字典套字典
error_stats = {body: [] for body in BODIES}

print("\n开始对比数值积分与内置星历的差异...")

for idx in compare_indices:
    # 当前时刻（相对 t0 的秒数）
    t_offset = t_eval[idx] * u.s
    # 计算此时的绝对时间
    current_time = t0 + t_offset
    
    # 针对每个天体，分别比较位置和速度
    for i, body in enumerate(BODIES):
        # 数值积分结果 (m, m/s)
        x_sim = state_reshaped[i, 0, idx]  # x
        y_sim = state_reshaped[i, 1, idx]  # y
        z_sim = state_reshaped[i, 2, idx]  # z
        vx_sim = state_reshaped[i, 3, idx] # vx
        vy_sim = state_reshaped[i, 4, idx] # vy
        vz_sim = state_reshaped[i, 5, idx] # vz
        
        # 调用 astropy 获取同一时刻的位置、速度 (默认质心 Frame)
        pos_astropy, vel_astropy = get_body_barycentric_posvel(body, current_time)
        
        # 转换为数值积分一致的单位：米和米/秒
        pos_astropy = pos_astropy.get_xyz().to(u.m).value   # [x, y, z]
        vel_astropy = vel_astropy.xyz.to(u.m/u.s).value     # [vx, vy, vz]
        
        # 计算差值 (sim - astropy)
        diff_pos = np.array([x_sim, y_sim, z_sim]) - pos_astropy
        diff_vel = np.array([vx_sim, vy_sim, vz_sim]) - vel_astropy
        
        # 位置和速度的欧几里得范数（误差绝对值）
        pos_error = np.linalg.norm(diff_pos)
        vel_error = np.linalg.norm(diff_vel)
        
        # 保存到 error_stats 中
        error_stats[body].append({
            "time": current_time.iso,          # 当前时刻的 ISO 字符串
            "t_offset_day": t_offset.to(u.day).value,  # 相对 t0 的天数
            "pos_error_m": pos_error,
            "vel_error_m_s": vel_error
        })

# 打印一些结果示例（也可以改为写入文件等）
print("\n误差对比结果(示例):")
for body in BODIES:
    # 只演示打印该 body 前 3 条记录
    print(f"\n天体: {body}")
    for rec in error_stats[body][:3]:
        print(
            f"  时间 = {rec['time']}, "
            f"  相对t0天数 = {rec['t_offset_day']:.2f}, "
            f"  位置误差 = {rec['pos_error_m']:.2e} m, "
            f"  速度误差 = {rec['vel_error_m_s']:.2e} m/s"
        )

# 你也可以对 error_stats 做更进一步的统计分析，如最大/最小/平均误差等。
# 下面仅示例如何计算每个天体在所有采样点的最大位置误差和最大速度误差。
print("\n=== 误差统计(最大值) ===")
for body in BODIES:
    pos_errs = [item["pos_error_m"] for item in error_stats[body]]
    vel_errs = [item["vel_error_m_s"] for item in error_stats[body]]
    print(f"{body}: 最大位置误差 = {max(pos_errs):.3e} m, 最大速度误差 = {max(vel_errs):.3e} m/s")

# ==========================================
# 绘制误差随时间变化图
# ==========================================
print("\n开始绘制误差图...")

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
fig.suptitle('数值积分与星历表对比误差随时间变化', fontsize=16, fontproperties='SimHei')

# 颜色和标记循环，用于区分不同天体
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

# 为每个天体绘制误差曲线
for i, body in enumerate(BODIES):
    # 提取数据
    times = [Time(item["time"]).datetime for item in error_stats[body]]
    pos_errors = [item["pos_error_m"] for item in error_stats[body]]
    vel_errors = [item["vel_error_m_s"] for item in error_stats[body]]
    
    # 在第一个子图中绘制位置误差
    color_idx = i % len(colors)
    marker_idx = i % len(markers)
    ax1.semilogy(times, pos_errors, 
                 label=body, 
                 color=colors[color_idx],
                 marker=markers[marker_idx], 
                 markersize=8,
                 linestyle='-',
                 linewidth=1.5,
                 markevery=max(1, len(times)//10))  # 只显示部分标记点，避免过于拥挤
    
    # 在第二个子图中绘制速度误差
    ax2.semilogy(times, vel_errors, 
                 label=body, 
                 color=colors[color_idx],
                 marker=markers[marker_idx], 
                 markersize=8,
                 linestyle='-',
                 linewidth=1.5,
                 markevery=max(1, len(times)//10))

# 设置第一个子图(位置误差)的属性
ax1.set_ylabel('位置误差 (米)', fontsize=14, fontproperties='SimHei')
ax1.set_title('位置误差随时间变化', fontsize=14, fontproperties='SimHei')
ax1.grid(True, which='both', linestyle='--', alpha=0.7)
ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10, prop={'family':'SimHei'})
ax1.set_ylim(bottom=1e-5)  # 设置Y轴下限，根据实际误差调整

# 设置第二个子图(速度误差)的属性
ax2.set_ylabel('速度误差 (米/秒)', fontsize=14, fontproperties='SimHei')
ax2.set_xlabel('时间', fontsize=14, fontproperties='SimHei')
ax2.set_title('速度误差随时间变化', fontsize=14, fontproperties='SimHei')
ax2.grid(True, which='both', linestyle='--', alpha=0.7)
ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10, prop={'family':'SimHei'})
ax2.set_ylim(bottom=1e-9)  # 设置Y轴下限，根据实际误差调整

# 格式化x轴日期
date_fmt = mdates.DateFormatter('%Y-%m-%d')
ax2.xaxis.set_major_formatter(date_fmt)
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 每6个月显示一个刻度
fig.autofmt_xdate()  # 自动格式化日期标签，避免重叠

# 调整布局
plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # 留出右侧空间放图例

# 保存图像
plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
print("误差图已保存为 error_analysis.png")

# 显示图形
plt.show()

# 保存误差数据到JSON文件，方便后续分析
error_data = {body: [
    {
        "time": item["time"],
        "t_offset_day": item["t_offset_day"],
        "pos_error_m": item["pos_error_m"],
        "vel_error_m_s": item["vel_error_m_s"]
    } for item in error_stats[body]
] for body in BODIES}

with open("error_stats.json", "w", encoding="utf-8") as f:
    json.dump(error_data, f, indent=2, ensure_ascii=False)
print("误差统计数据已保存至 error_stats.json")
