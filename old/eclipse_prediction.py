"""
日食月食预测模块
"""
import numpy as np
from astropy.time import Time
import astropy.units as u
from tqdm import tqdm

# ------------------------------------------------------------------------------
# 原始“锥体”逻辑保留，不改动
# ------------------------------------------------------------------------------
def body_in_cone(body_pos, body_radius, cone_tip, cone_axis, half_angle):
    """
    判断天体是否在锥体内部（部分进入）。
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
    判断天体是否完全进入锥体（本影/半影）。
    """
    body_to_cone_tip = body_pos - cone_tip
    proj_point = cone_tip + np.dot(body_to_cone_tip, cone_axis) * cone_axis / (np.linalg.norm(cone_axis)**2)
    perpendicular_distance = np.linalg.norm(body_pos - proj_point)
    horizontal_distance = np.linalg.norm(cone_tip - proj_point)

    critical_distance = np.tan(half_angle) * horizontal_distance - body_radius / np.cos(half_angle)
    if perpendicular_distance > critical_distance:
        return False
    return True


# ------------------------------------------------------------------------------
# 第一层筛选：用更简单的近似判定，只要“角度够小”即可，不用 arccos
# ------------------------------------------------------------------------------
def near_alignment_cross(es_vec, em_vec, max_sin_angle):
    """
    判断两个向量 es_vec (地->日), em_vec (地->月) 是否近似共线，
    这里用叉积衡量 sin(夹角) = |es x em| / (|es| * |em|)。
    
    如果 sin(夹角) < max_sin_angle，则认为“足够接近共线”。
    
    注意：本函数不做任何距离精确判断，只是快速估计。
    """
    cross_prod = np.cross(es_vec, em_vec)
    cross_mag = np.linalg.norm(cross_prod)
    denom = np.linalg.norm(es_vec) * np.linalg.norm(em_vec)
    # sinθ = cross_mag / denom
    return (cross_mag < max_sin_angle * denom)


# ------------------------------------------------------------------------------
# 主函数：两级筛选
# ------------------------------------------------------------------------------
def predict_eclipses(sol, t0, state_reshaped, sun_idx, earth_idx, moon_idx):
    """
    预测日食和月食事件 (改写版：第一层简化筛选，第二层仍用锥体几何)。
    """
    eclipse_events = {"solar": [], "lunar": []}

    # 位置数组 (shape = (3, n_times))
    sun_pos   = state_reshaped[sun_idx, :3, :]
    earth_pos = state_reshaped[earth_idx, :3, :]
    moon_pos  = state_reshaped[moon_idx, :3, :]

    n_times = sun_pos.shape[1]

    # 预定义一些物理半径
    R_sun   = 695700e3
    R_earth = 6371e3
    R_moon  = 1737.4e3

    # 设定一个“最大允许正弦值”来判定向量近似共线
    # sin(theta) < 0.02 对应角度 ~1.15度
    max_sin_angle = 0.02

    # 为了显示进度，用 tqdm
    pbar = tqdm(range(n_times), desc="预判食相")

    # ----------------------
    # 第一层简化判定
    # ----------------------
    # solar_mask, lunar_mask 分别记录潜在日食/月食时刻
    solar_candidates = []
    lunar_candidates = []

    for i in pbar:
        es_vec = sun_pos[:, i] - earth_pos[:, i]   # 地->日
        em_vec = moon_pos[:, i] - earth_pos[:, i]  # 地->月

        dot_val = np.dot(es_vec, em_vec)

        # 1) 看向量点积的正负 => 同侧/异侧
        # 2) 看是否近似共线 => cross(es, em) 小
        # 我们此处不严格比较 |EM| < |ES|, 也可在此加入简化的距离判断

        # --- potential solar eclipse ---
        if dot_val > 0:  # 同侧
            # 检查近似共线
            if near_alignment_cross(es_vec, em_vec, max_sin_angle):
                solar_candidates.append(i)

        # --- potential lunar eclipse ---
        if dot_val < 0:  # 异侧
            # 同理
            if near_alignment_cross(es_vec, em_vec, max_sin_angle):
                lunar_candidates.append(i)

    # ----------------------
    # 第二层：对候选时刻做精细的锥体判断
    # ----------------------
    pbar_solar = tqdm(solar_candidates, desc="分析日食", leave=False)
    for i in pbar_solar:
        t_sec = sol.t[i]
        current_time = t0 + t_sec * u.s
        formatted_time = current_time.iso

        sp = sun_pos[:, i]
        ep = earth_pos[:, i]
        mp = moon_pos[:, i]

        # 复用您原本的严格几何判定
        earth_to_sun_ = sp - ep
        earth_to_moon_ = mp - ep
        dist_es_ = np.linalg.norm(earth_to_sun_)
        dist_em_ = np.linalg.norm(earth_to_moon_)

        # 保持原逻辑：同侧 + dist(EM)<dist(ES)
        if (np.dot(earth_to_sun_, earth_to_moon_) > 0) and (dist_em_ < dist_es_):
            # 原本要算夹角或阈值等，这里不改
            # =========== 影锥计算 ===========
            moon_to_sun_ = sp - mp
            dist_ms_ = np.linalg.norm(moon_to_sun_)
            umbra_dir = - (moon_to_sun_ / dist_ms_)

            umbra_angle = np.arcsin((R_sun - R_moon) / dist_ms_)
            umbra_tip = sp + R_sun / np.sin(umbra_angle) * umbra_dir

            penumbra_angle = np.arcsin((R_sun + R_moon) / dist_ms_)
            penumbra_tip = sp + R_sun / np.sin(penumbra_angle) * umbra_dir

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

    pbar_lunar = tqdm(lunar_candidates, desc="分析月食", leave=False)
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

        # 保持原逻辑：异侧 + dist(EM)<dist(ES)
        if (np.dot(earth_to_sun_, earth_to_moon_) < 0) and (dist_em_ < dist_es_):
            # =========== 影锥计算 ===========
            earth_umbra_dir = - earth_to_sun_ / dist_es_

            earth_umbra_angle = np.arcsin((R_sun - R_earth) / dist_es_)
            earth_umbra_tip = ep + (R_earth / np.tan(earth_umbra_angle)) * earth_umbra_dir

            earth_penumbra_angle = np.arcsin((R_sun + R_earth) / dist_es_)
            earth_penumbra_tip = ep - (R_earth / np.tan(earth_penumbra_angle)) * earth_umbra_dir

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

    # 统计结果
    print(f"分析完成！发现 {len(eclipse_events['solar'])} 次日食，"
          f"{len(eclipse_events['lunar'])} 次月食。")

    return eclipse_events
