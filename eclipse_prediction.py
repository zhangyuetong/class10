"""
日食月食预测模块

此模块提供日食和月食事件的预测功能，基于天体三维位置数据。
"""

import numpy as np
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from visualize.solar_plot import plot_eclipse_geometry_3d

"""
日食月食的几何推导

1. 日食原理与精确判定条件

   日食发生的本质是月球遮挡太阳光线射向地球的现象。根据几何光学原理，我们可以进行如下推导：
   ...（省略详细推导）
   
2. 月食原理
   太阳-地球-月球三者接近一条直线，且地球位于太阳和月球之间时，
   月球进入地球的阴影区域，形成月食。
   
   月食判定条件：
   a) 太阳-地球-月球三点夹角接近180度
   b) 月球进入地球的本影或半影区域

3. 计算方法
   - 判断三点是否接近共线：计算向量夹角
   - 计算阴影锥体：基于天体半径和距离
   - 检查是否落入阴影区域：几何关系计算
"""




def predict_eclipses(sol, t0, state_reshaped, sun_idx, earth_idx, moon_idx):
    """
    预测日食和月食事件
    
    参数:
    sol - 数值积分求解结果
    t0 - 初始时间
    state_reshaped - 重塑后的状态数组
    sun_idx - 太阳在状态数组中的索引
    earth_idx - 地球在状态数组中的索引
    moon_idx - 月球在状态数组中的索引
    
    返回:
    eclipse_events - 包含日食和月食事件的字典
    """
    # 初始化日食月食事件字典
    eclipse_events = {
        "solar": [],  # 日食事件列表
        "lunar": []   # 月食事件列表
    }
    
    # 创建进度条
    pbar = tqdm(total=len(sol.t), desc="分析日食月食")
    
    for i, t_sec in enumerate(sol.t):
        # 获取当前历元时间（TT时间系统）
        current_time = t0 + t_sec * u.s  # 从初始时间t0开始累加积分步长
        formatted_time = current_time.iso  # 格式化为ISO标准时间字符串
        
        # 从状态数组中提取天体位置（假设为J2000坐标系）
        # sun_idx, earth_idx, moon_idx 分别为太阳、地球、月球在数组中的索引
        sun_pos = state_reshaped[sun_idx, :3, i]    # 太阳的x,y,z坐标 (m)
        earth_pos = state_reshaped[earth_idx, :3, i] # 地球的坐标 (m)
        moon_pos = state_reshaped[moon_idx, :3, i]   # 月球的坐标 (m)
        
        # 计算相对位置向量（地心视角）
        earth_to_sun = sun_pos - earth_pos  # 从地球指向太阳的向量
        earth_to_moon = moon_pos - earth_pos # 从地球指向月球的向量
        moon_to_sun = sun_pos - moon_pos    # 从月球指向太阳的向量
        
        # 计算天体间欧氏距离（标量）
        earth_sun_dist = np.linalg.norm(earth_to_sun)  # 地日距离 ‖ES‖
        earth_moon_dist = np.linalg.norm(earth_to_moon) # 地月距离 ‖EM‖
        moon_sun_dist = np.linalg.norm(moon_to_sun)    # 月日距离 ‖MS‖
        
        # 归一化方向向量（用于角度计算）
        earth_to_sun_unit = earth_to_sun / earth_sun_dist  # 太阳方向单位向量 EŜ
        earth_to_moon_unit = earth_to_moon / earth_moon_dist # 月球方向单位向量 EM̂
        moon_to_sun_unit = moon_to_sun / moon_sun_dist     # 太阳在月心坐标中的方向 MŜ
        
        # 天体物理常数（单位：米）
        R_sun = 695700e3     # 太阳半径 (IAU 2015)
        R_moon = 1737.4e3    # 月球半径 (LRO激光测距)
        R_earth = 6371e3     # 地球赤道半径 (WGS84)
        
        # ===== 日食初步筛选条件 =====
        # 条件1：月球位于地球与太阳连线的同侧（点积>0）
        # 条件2：地月距离小于地日距离（确保月球在地球和太阳之间）
        if np.dot(earth_to_sun_unit, earth_to_moon_unit) > 0 and (earth_moon_dist < earth_sun_dist):
            # 计算地日-地月向量夹角（精确到1e-12弧度）
            # 使用np.clip防止浮点误差导致反余弦计算异常
            cos_theta = np.clip(np.dot(earth_to_sun_unit, earth_to_moon_unit), -1.0, 1.0)
            solar_eclipse_angle = np.arccos(cos_theta)  # 夹角θ = arccos(EŜ·EM̂)
            solar_eclipse_threshold = 2e-2  # 约1.146度（经验阈值）
            
            # 当夹角小于阈值时值得进入详细计算
            if solar_eclipse_angle < solar_eclipse_threshold:
                # 绘制日食几何关系图
                plot_eclipse_geometry_3d(earth_pos, moon_pos, sun_pos, R_earth, R_moon, formatted_time)
                
                # 调试输出：显示角度信息（转换为度数）
                tqdm.write(f"[DEBUG] ===== 时间: {formatted_time} | 地日-地月夹角: {np.degrees(solar_eclipse_angle):.6f}° =====")
                # ===== 本影锥参数计算 =====
                # 本影锥长度公式推导：相似三角形 (R_sun - R_moon)/D_sm = R_moon/L_umbra
                D_sm = moon_sun_dist  # 月日距离 ‖MS‖
                tqdm.write(f"[DEBUG] 月日距离: {D_sm/1e3:.1f}千公里")
                L_umbra = (R_moon * D_sm) / (R_sun - R_moon)  # 本影锥理论长度
                tqdm.write(f"[DEBUG] 本影锥理论长度: {L_umbra/1e3:.1f}千公里")

                # 本影锥方向向量：从太阳指向月球的方向（与光照方向相反）
                dir_umbra = (moon_pos - sun_pos) / D_sm  # 单位向量 MŜ
                tqdm.write(f"[DEBUG] 本影锥方向向量: {dir_umbra}")

                # 本影锥顶点计算：顶点位于月球背阳侧延长线上
                # V_umbra = M + L_umbra * MŜ (沿月日连线延伸)
                V_umbra = moon_pos + dir_umbra * L_umbra
                tqdm.write(f"[DEBUG] 本影锥顶点: {V_umbra}")
                # 计算地球到本影锥顶点的向量
                VE = earth_pos - V_umbra  # 向量VE = E - V_umbra
                tqdm.write(f"[DEBUG] 地球到本影锥顶点向量: {VE}")
                # 地球沿本影轴线方向的投影距离（标量投影）
                # t_umbra = VE · dir_umbra （带符号的投影长度）
                t_umbra = np.dot(VE, dir_umbra)
                tqdm.write(f"[DEBUG] 地球沿本影轴线方向的投影距离: {t_umbra}")

                # 地球到本影轴线的垂直距离（向量分解）
                # 垂直分量 = VE - (t_umbra * dir_umbra)
                d_umbra = np.linalg.norm(VE - t_umbra * dir_umbra)
                tqdm.write(f"[DEBUG] 地球到本影轴线的垂直距离: {d_umbra}")

                # 本影锥半顶角计算（锥体张开角度）
                alpha_umbra = np.arctan((R_sun - R_moon) / D_sm)  # tanα = (R_sun-R_moon)/D_sm
                tqdm.write(f"[DEBUG] 本影锥半顶角: {alpha_umbra}")
                # 本影锥在地球位置的截面半径（随时间变化的锥体宽度）
                # 锥体半径公式：r = R_earth + t * tanα （考虑地球自身半径）
                max_r_umbra = R_earth + t_umbra * np.tan(alpha_umbra)
                tqdm.write(f"[DEBUG] 本影锥在地球位置的截面半径: {max_r_umbra}")
                # 调试输出本影参数（转换为千米）
                tqdm.write(f"[DEBUG] 本影长度: {L_umbra/1e3:.1f}千公里 | 顶点距地: {np.linalg.norm(VE)/1e3:.1f}千公里")
                tqdm.write(f"[DEBUG] 轴向投影: t={t_umbra/1e3:.1f}千公里 | 垂直距离: d={d_umbra/1e3:.1f}千公里")
                tqdm.write(f"[DEBUG] 本影锥半径: {max_r_umbra/1e3:.1f}千公里 | 地球半径: {R_earth/1e3:.1f}千公里")
                
                # 初始化事件参数
                eclipse_type = None
                eclipse_magnitude = 0.0
                
                # ===== 日全食条件判断 =====
                tqdm.write(f"[DEBUG] === 日全食判断 地月距离: {earth_moon_dist/1e3:.1f}千公里 | 本影长度: {L_umbra/1e3:.1f}千公里 ===")
                # 条件1：地月距离 < 本影长度 → 本影可达地球轨道附近
                # 条件2：t_umbra ≥ 0 → 地球在本影锥的延长线上（非后方）
                # 条件3：d_umbra ≤ max_r_umbra → 地球进入本影锥截面
                if (earth_moon_dist < L_umbra) and (t_umbra >= 0) and (d_umbra <= max_r_umbra):
                    eclipse_type = "Total"
                    eclipse_magnitude = 1.0  # 全食时食分强制为1.0
                    tqdm.write(f"[DEBUG] 日全食条件满足 | D_me/L_umbra={earth_moon_dist/L_umbra:.3f}")
                
                # ===== 日环食条件判断 =====
                else:
                    tqdm.write(f"[DEBUG] === 日环食判断 地月距离: {earth_moon_dist/1e3:.1f}千公里 | 本影长度: {L_umbra/1e3:.1f}千公里 ===")
                    # 伪本影（antumbra）参数计算
                    beta_antumbra = np.arctan((R_sun + R_moon) / D_sm)  # 伪本影半顶角
                    max_r_antumbra = R_earth + t_umbra * np.tan(beta_antumbra)  # 伪本影截面半径
                    
                    tqdm.write(f"[DEBUG] 伪本影半径: {max_r_antumbra/1e3:.1f}千公里 | 地球半径: {R_earth/1e3:.1f}千公里")
                    
                    # 条件1：地月距离 > 本影长度 → 本影无法到达地球
                    # 条件2：t_umbra ≥ 0 → 地球在伪本影延长线上
                    # 条件3：d_umbra ≤ max_r_antumbra → 地球进入伪本影
                    if (earth_moon_dist > L_umbra) and (t_umbra >= 0) and (d_umbra <= max_r_antumbra):
                        eclipse_type = "Annular"
                        eclipse_magnitude = 1.0  # 环食时食分同样为1.0
                        tqdm.write(f"[DEBUG] 日环食条件满足 | D_me/L_umbra={earth_moon_dist/L_umbra:.3f}")
                
                    # ===== 日偏食条件判断 =====
                    else:
                        tqdm.write(f"[DEBUG] === 日偏食判断 地月距离: {earth_moon_dist/1e3:.1f}千公里 | 本影长度: {L_umbra/1e3:.1f}千公里 ===")
                        # 半影锥参数计算（与伪本影方向相同）
                        L_penumbra = (R_moon * D_sm) / (R_sun + R_moon)  # 半影锥长度
                        V_penumbra = moon_pos + dir_umbra * L_penumbra   # 半影锥顶点
                        VE_pen = earth_pos - V_penumbra                 # 地球到半影顶点向量
                        
                        # 投影参数计算（类似本影计算）
                        t_pen = np.dot(VE_pen, dir_umbra)  # 沿半影轴线的投影距离
                        d_pen = np.linalg.norm(VE_pen - t_pen * dir_umbra)  # 到轴线的垂直距离
                        alpha_penumbra = np.arctan((R_sun + R_moon) / D_sm) # 半影半顶角
                        max_r_penumbra = R_earth + t_pen * np.tan(alpha_penumbra) # 半影截面半径
                        
                        tqdm.write(f"[DEBUG] 半影长度: {L_penumbra/1e3:.1f}千公里 | 顶点距地: {np.linalg.norm(VE_pen)/1e3:.1f}千公里")
                        tqdm.write(f"[DEBUG] 半影投影: t={t_pen/1e3:.1f}千公里 | d={d_pen/1e3:.1f}千公里")
                        tqdm.write(f"[DEBUG] 半影半径: {max_r_penumbra/1e3:.1f}千公里 vs 地球半径: {R_earth/1e3:.1f}千公里")
                        
                        # 判断半影是否覆盖地球
                        if t_pen >= 0 and d_pen <= max_r_penumbra:
                            eclipse_type = "Partial"
                            # 食分计算：基于视圆面重叠比例
                            sun_ang_rad = np.arcsin(R_sun / earth_sun_dist)  # 太阳视半径 ≈0.266°
                            moon_ang_rad = np.arcsin(R_moon / earth_moon_dist) # 月球视半径 ≈0.259°
                            ang_separation = solar_eclipse_angle  # 地心视角距
                            # 食分公式：(月球视半径 + 太阳视半径 - 角距) / (2×太阳视半径)
                            eclipse_magnitude = (moon_ang_rad + sun_ang_rad - ang_separation) / (2 * sun_ang_rad)
                            eclipse_magnitude = np.clip(eclipse_magnitude, 0, 1)  # 约束在[0,1]范围
                            tqdm.write(f"[DEBUG] 视半径 | 太阳: {np.degrees(sun_ang_rad)*3600:.1f}″ 月球: {np.degrees(moon_ang_rad)*3600:.1f}″")
                
                # ===== 事件记录 =====
                if eclipse_type:
                    eclipse_events["solar"].append({
                        "time": formatted_time,
                        "type": eclipse_type,
                        "magnitude": round(eclipse_magnitude, 3)  # 保留3位小数
                    })
                    # 格式化输出检测结果
                    tqdm.write(f"[检测] 时间: {formatted_time} | 类型: {eclipse_type:7} | 食分: {eclipse_magnitude:.3f}")
        
        # ===== 月食检测 =====
        # 要求地球位于太阳和月球之间，即地球–太阳与地球–月球反向（点积 < 0）
        if np.dot(earth_to_sun_unit, earth_to_moon_unit) < 0:
            # 计算太阳-地球-月球夹角（弧度），应接近π
            lunar_eclipse_angle = np.arccos(np.clip(np.dot(-earth_to_sun_unit, earth_to_moon_unit), -1.0, 1.0))
            lunar_eclipse_threshold = 1e-2  # 角度阈值(弧度)
            
            if abs(lunar_eclipse_angle) < lunar_eclipse_threshold:
                # 计算地球本影锥顶角
                earth_umbra_angle = 2 * np.arctan((R_sun - R_earth) / earth_sun_dist)
                
                # 计算地球本影锥长度
                earth_umbra_length = R_earth / np.tan(earth_umbra_angle / 2)
                
                # 计算地球半影锥顶角
                earth_penumbra_angle = 2 * np.arctan((R_sun + R_earth) / earth_sun_dist)
                
                # 若月球在地球影子范围内（本影或半影）
                if earth_moon_dist < earth_umbra_length or True:  # 这里不再做严格距离限制
                    # 计算月球到日地连线的垂直距离
                    projection = np.dot(earth_to_moon, earth_to_sun_unit)
                    perp_dist = np.sqrt(earth_moon_dist**2 - projection**2)
                    
                    # 计算地球本影在月球距离处的半径
                    umbra_radius_at_moon = R_earth - (earth_moon_dist * np.tan(earth_umbra_angle / 2))
                    
                    # 计算地球半影在月球距离处的半径
                    penumbra_radius_at_moon = R_earth + (earth_moon_dist * np.tan(earth_penumbra_angle / 2))
                    
                    # 判断月食类型
                    lunar_eclipse_type = None
                    if perp_dist < umbra_radius_at_moon - R_moon:
                        lunar_eclipse_type = "Total"  # 全食
                    elif perp_dist < umbra_radius_at_moon + R_moon:
                        lunar_eclipse_type = "Partial"  # 偏食
                    elif perp_dist < penumbra_radius_at_moon + R_moon:
                        lunar_eclipse_type = "Penumbral"  # 半影食
                    else:
                        continue
                    
                    # 计算月食食分
                    if lunar_eclipse_type in ["Total", "Partial"]:
                        magnitude = (umbra_radius_at_moon + R_moon - perp_dist) / (2 * R_moon)
                    else:
                        magnitude = (penumbra_radius_at_moon + R_moon - perp_dist) / (2 * R_moon)
                    
                    magnitude = min(1.0, max(0.0, magnitude))
                    
                    eclipse_events["lunar"].append({
                        "time": formatted_time,
                        "type": lunar_eclipse_type,
                        "angle": float(lunar_eclipse_angle),
                        "perpendicular_distance": float(perp_dist),
                        "umbra_radius": float(umbra_radius_at_moon),
                        "penumbra_radius": float(penumbra_radius_at_moon),
                        "magnitude": float(magnitude)
                    })
                    
                    # 在进度条下方输出检测结果
                    tqdm.write(f"检测到可能的月食事件: {formatted_time}, 类型: {lunar_eclipse_type}, 食分: {magnitude:.2f}")
        
        # 更新进度条
        pbar.update(1)
    
    # 关闭进度条
    pbar.close()
    
    print(f"分析完成！发现 {len(eclipse_events['solar'])} 次可能的日食和 {len(eclipse_events['lunar'])} 次可能的月食")
    return eclipse_events
