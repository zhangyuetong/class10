import numpy as np
import json
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def analyze_errors(sol, t0, state_reshaped, BODIES, t_eval, sample_interval=500):
    """
    对比数值积分结果与Astropy内置星历(JPL)的位置/速度差异
    
    参数:
    sol - solve_ivp 求解结果
    t0 - 初始时间 (Time对象)
    state_reshaped - 重塑后的状态数组 (天体数, 6, 时间点数)
    BODIES - 天体列表
    t_eval - 评估时间点
    sample_interval - 采样间隔，默认每500个点取一个
    
    返回:
    error_stats - 误差统计字典
    """
    # 选取需要对比的采样时刻的下标
    compare_indices = np.arange(0, len(t_eval), sample_interval)

    # 准备存储结果的容器
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
    
    return error_stats

def print_error_summary(error_stats, BODIES):
    """打印误差统计摘要"""
    # 打印一些结果示例
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

    # 打印每个天体在所有采样点的最大位置误差和最大速度误差
    print("\n=== 误差统计(最大值) ===")
    for body in BODIES:
        pos_errs = [item["pos_error_m"] for item in error_stats[body]]
        vel_errs = [item["vel_error_m_s"] for item in error_stats[body]]
        print(f"{body}: 最大位置误差 = {max(pos_errs):.3e} m, 最大速度误差 = {max(vel_errs):.3e} m/s")

def plot_errors(error_stats, BODIES):
    """绘制误差随时间变化图"""
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

def save_error_data(error_stats, BODIES):
    """保存误差数据到JSON文件"""
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

def run_error_analysis(sol, t0, state_reshaped, BODIES, t_eval, sample_interval=500):
    """
    执行完整的误差分析流程
    """
    # 分析误差
    error_stats = analyze_errors(sol, t0, state_reshaped, BODIES, t_eval, sample_interval)
    
    # 打印误差摘要
    print_error_summary(error_stats, BODIES)
    
    # 绘制误差图
    plot_errors(error_stats, BODIES)
    
    # 保存误差数据
    save_error_data(error_stats, BODIES)
    
    return error_stats
