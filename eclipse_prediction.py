"""
日食月食预测模块
"""
import numpy as np
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm

# ------------------------------------------------------------------------------------
# 不修改原始“锥体判断”逻辑
# ------------------------------------------------------------------------------------
def body_in_cone(body_pos, body_radius, cone_tip, cone_axis, half_angle):
    """
    判断天体是否在锥体内部（部分进入），保持原逻辑不变。
    """
    body_to_cone_tip = body_pos - cone_tip
    proj_point = cone_tip + np.dot(body_to_cone_tip, cone_axis) * cone_axis / (np.linalg.norm(cone_axis)**2)
    perpendicular_distance = np.linalg.norm(body_pos - proj_point)
    horizontal_distance = np.linalg.norm(cone_tip - proj_point)

    critical_distance = np.tan(half_angle) * horizontal_distance + body_radius / np.cos(half_angle)
    if perpendicular_distance < critical_distance:
        return True
    return False


def body_totally_in_cone(body_pos, body_radius, cone_tip, cone_axis, half_angle):
    """
    判断天体是否完全进入锥体（本影/半影），保持原逻辑不变。
    """
    body_to_cone_tip = body_pos - cone_tip
    proj_point = cone_tip + np.dot(body_to_cone_tip, cone_axis) * cone_axis / (np.linalg.norm(cone_axis)**2)
    perpendicular_distance = np.linalg.norm(body_pos - proj_point)
    horizontal_distance = np.linalg.norm(cone_tip - proj_point)

    critical_distance = np.tan(half_angle) * horizontal_distance - body_radius / np.cos(half_angle)
    if perpendicular_distance > critical_distance:
        return False
    return True


# ------------------------------------------------------------------------------------
# 向量化+两级筛选后的主函数
# ------------------------------------------------------------------------------------
def predict_eclipses(sol, t0, state_reshaped, sun_idx, earth_idx, moon_idx):
    """
    预测日食和月食事件 (两级筛选版本)。
    
    第一级筛选：向量化计算(地->日,地->月)夹角、距离比，
              快速排除绝大多数时刻；
    第二级筛选：对剩余时刻做原始的日食/月食几何锥体计算。

    参数:
    ----------
    sol           : solve_ivp 的结果对象 (包含时间 sol.t)
    t0            : astropy Time，积分起始时间
    state_reshaped: (n_bodies, 6, n_times) 状态向量
    sun_idx       : 太阳索引
    earth_idx     : 地球索引
    moon_idx      : 月球索引

    返回:
    ----------
    eclipse_events = {"solar": [...], "lunar": [...]}
    """
    # 初始化结果
    eclipse_events = {
        "solar": [],
        "lunar": []
    }

    # 位置数组 (shape = (3, n_times))
    sun_pos   = state_reshaped[sun_idx, :3, :]
    earth_pos = state_reshaped[earth_idx, :3, :]
    moon_pos  = state_reshaped[moon_idx, :3, :]

    # 相对向量 & 距离
    earth_to_sun  = sun_pos   - earth_pos  # shape=(3,n_times)
    earth_to_moon = moon_pos  - earth_pos
    moon_to_sun   = sun_pos   - moon_pos

    dist_es = np.linalg.norm(earth_to_sun,  axis=0)  # 地日距离
    dist_em = np.linalg.norm(earth_to_moon, axis=0)  # 地月距离
    dist_ms = np.linalg.norm(moon_to_sun,   axis=0)  # 月日距离

    # 单位向量 (shape = (3,n_times))
    # 避免除零风险，可确保 dist>0
    earth_to_sun_unit  = earth_to_sun  / dist_es
    earth_to_moon_unit = earth_to_moon / dist_em
    moon_to_sun_unit   = moon_to_sun   / dist_ms

    # -----------------------
    # 日食初筛条件 (向量化)
    # dot_es_em = E->S · E->M
    # 1) dot_es_em > 0 (同侧)
    # 2) dist_em < dist_es
    # 3) 夹角小于某阈值 => arccos(dot_es_em) < threshold
    # -----------------------
    dot_es_em = np.einsum('ij,ij->j', earth_to_sun_unit, earth_to_moon_unit)
    dot_es_em_clamped = np.clip(dot_es_em, -1.0, 1.0)
    angle_solar = np.arccos(dot_es_em_clamped)  # shape=(n_times,)

    # 此阈值是经验值，可自行微调
    solar_threshold = 2e-2  # 约1.146度
    solar_mask = (
        (dot_es_em > 0) &
        (dist_em < dist_es) &
        (angle_solar < solar_threshold)
    )

    # -----------------------
    # 月食初筛条件 (向量化)
    # 要求月球在地球后面 => dot(E->S, E->M) < 0
    # 并且 dist_em < dist_es
    # 以及夹角 < threshold
    # -----------------------
    # lunar 用 dot(E->S, -E->M) 计算夹角
    dot_es_neg_em = np.einsum('ij,ij->j', earth_to_sun_unit, -earth_to_moon_unit)
    dot_es_neg_em_clamped = np.clip(dot_es_neg_em, -1.0, 1.0)
    angle_lunar = np.arccos(dot_es_neg_em_clamped)

    lunar_threshold = 2e-2
    lunar_mask = (
        (dot_es_em < 0) &
        (dist_em < dist_es) &
        (angle_lunar < lunar_threshold)
    )

    # 太阳 & 地球 & 月球物理半径 (仅示例值)
    R_sun   = 695700e3
    R_earth = 6371e3
    R_moon  = 1737.4e3

    # --------------------------------------------------------------------------------
    # 第二级：对初筛通过的“潜在日食”时刻，做精细锥体几何计算
    # --------------------------------------------------------------------------------
    idx_solar_candidates = np.where(solar_mask)[0]  # 时刻下标
    pbar_solar = tqdm(idx_solar_candidates, desc="分析日食", leave=False)

    for i in pbar_solar:
        t_sec = sol.t[i]
        current_time = t0 + t_sec * u.s
        formatted_time = current_time.iso

        # 单时刻位置向量
        sp = sun_pos[:, i]
        ep = earth_pos[:, i]
        mp = moon_pos[:, i]

        earth_to_sun_ = sp - ep
        earth_to_moon_ = mp - ep
        moon_to_sun_ = sp - mp

        dist_es_ = np.linalg.norm(earth_to_sun_)
        dist_em_ = np.linalg.norm(earth_to_moon_)
        dist_ms_ = np.linalg.norm(moon_to_sun_)

        # 单位向量
        earth_to_sun_unit_  = earth_to_sun_  / dist_es_
        earth_to_moon_unit_ = earth_to_moon_ / dist_em_
        moon_to_sun_unit_   = moon_to_sun_   / dist_ms_

        # 保持原逻辑：再判断一次同侧 + 距离
        if (np.dot(earth_to_sun_unit_, earth_to_moon_unit_) > 0) and (dist_em_ < dist_es_):
            # 计算夹角
            cos_theta = np.clip(np.dot(earth_to_sun_unit_, earth_to_moon_unit_), -1.0, 1.0)
            solar_eclipse_angle = np.arccos(cos_theta)
            solar_eclipse_threshold = 2e-2

            if solar_eclipse_angle < solar_eclipse_threshold:
                # 这里保持原“本影/半影”计算逻辑
                umbra_dir = -moon_to_sun_unit_

                # 本影锥半顶角
                umbra_angle = np.arcsin((R_sun - R_moon) / dist_ms_)
                # 本影锥锥顶位置
                umbra_tip = sp + R_sun / np.sin(umbra_angle) * umbra_dir

                # 半影锥半顶角
                penumbra_angle = np.arcsin((R_sun + R_moon) / dist_ms_)
                # 半影锥锥顶位置
                penumbra_tip = sp + R_sun / np.sin(penumbra_angle) * umbra_dir

                # 判断地球与这些锥体的位置关系
                dist_umbra_tip_to_earth = np.linalg.norm(umbra_tip - ep)
                eclipse_type = None

                # 日全食
                if dist_umbra_tip_to_earth < R_earth:
                    eclipse_type = "Total"
                # 日偏食
                elif body_in_cone(ep, R_earth, penumbra_tip, umbra_dir, penumbra_angle):
                    eclipse_type = "Partial"
                # 日环食
                elif body_in_cone(ep, R_earth, umbra_tip, umbra_dir, umbra_angle):
                    eclipse_type = "Annular"

                if eclipse_type is not None:
                    eclipse_events["solar"].append({
                        "time": formatted_time,
                        "type": eclipse_type
                    })

    # --------------------------------------------------------------------------------
    # 第二级：对初筛通过的“潜在月食”时刻，做精细锥体几何计算
    # --------------------------------------------------------------------------------
    idx_lunar_candidates = np.where(lunar_mask)[0]
    pbar_lunar = tqdm(idx_lunar_candidates, desc="分析月食", leave=False)

    for i in pbar_lunar:
        t_sec = sol.t[i]
        current_time = t0 + t_sec * u.s
        formatted_time = current_time.iso

        sp = sun_pos[:, i]
        ep = earth_pos[:, i]
        mp = moon_pos[:, i]

        earth_to_sun_ = sp - ep
        earth_to_moon_ = mp - ep
        dist_es_ = np.linalg.norm(earth_to_sun_)
        dist_em_ = np.linalg.norm(earth_to_moon_)

        earth_to_sun_unit_  = earth_to_sun_  / dist_es_
        earth_to_moon_unit_ = earth_to_moon_ / dist_em_

        # 保持原逻辑：再判断一次异侧 + 距离
        if (np.dot(earth_to_sun_unit_, earth_to_moon_unit_) < 0) and (dist_em_ < dist_es_):
            cos_theta = np.clip(np.dot(earth_to_sun_unit_, -earth_to_moon_unit_), -1.0, 1.0)
            lunar_eclipse_angle = np.arccos(cos_theta)
            lunar_eclipse_threshold = 2e-2

            if lunar_eclipse_angle < lunar_eclipse_threshold:
                # 继续原逻辑：地球本影/半影锥
                earth_umbra_dir = -earth_to_sun_unit_

                # 本影锥半顶角
                earth_umbra_angle = np.arcsin((R_sun - R_earth) / dist_es_)
                earth_umbra_tip = ep + (R_earth / np.tan(earth_umbra_angle)) * earth_umbra_dir

                # 半影锥半顶角
                earth_penumbra_angle = np.arcsin((R_sun + R_earth) / dist_es_)
                earth_penumbra_tip = ep - (R_earth / np.tan(earth_penumbra_angle)) * earth_umbra_dir

                # 判断月球是否进入这些锥体
                dist_ms_ = np.linalg.norm(sp - mp)  # 月日距离 (可不一定用上)
                eclipse_type = None

                if body_totally_in_cone(mp, R_moon, earth_umbra_tip, earth_umbra_dir, earth_umbra_angle):
                    eclipse_type = "Total"
                elif body_in_cone(mp, R_moon, earth_umbra_tip, earth_umbra_dir, earth_umbra_angle):
                    eclipse_type = "Partial"
                elif body_in_cone(mp, R_moon, earth_penumbra_tip, earth_umbra_dir, earth_penumbra_angle):
                    eclipse_type = "Penumbral"

                if eclipse_type is not None:
                    eclipse_events["lunar"].append({
                        "time": formatted_time,
                        "type": eclipse_type
                    })

    # --------------------------------------------------------------------------------
    # 打印最终统计
    # --------------------------------------------------------------------------------
    print(f"分析完成！发现 {len(eclipse_events['solar'])} 次日食，"
          f"{len(eclipse_events['lunar'])} 次月食。")

    return eclipse_events
