import numpy as np
from scipy.integrate import solve_ivp
import json
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u
from astropy.constants import G, M_sun, M_earth

# 手动定义月球质量（kg）
M_moon = 7.34767309e22

# 设置初始时刻
t0 = Time("2025-01-01T00:00:00", scale='utc')

# 获取太阳、地球、月亮的初始位置和速度（相对于太阳系质心）
# 注意：对于太阳，由于天文历书中通常以太阳系质心为参考系，
# 这里直接使用 "sun" 得到的是太阳相对于质心的运动
bodies = ['sun', 'earth', 'moon']
init_state = []
for body in bodies:
    pos, vel = get_body_barycentric_posvel(body, t0)
    # 转换为 SI 单位：位置单位转换为 m，速度转换为 m/s
    pos = pos.get_xyz().to(u.m).value  # 三维位置数组
    vel = vel.xyz.to(u.m/u.s).value  # 三维速度数组
    init_state.append(pos)
    init_state.append(vel)
# 将状态向量展平，形状为(18,)
y0 = np.hstack(init_state)

# 定义各天体的质量（单位：kg）
masses = np.array([M_sun.value, M_earth.value, M_moon])

# 万有引力常数（SI 单位 m^3 / (kg s^2)）
G_val = G.value

def derivatives(t, y):
    """
    y: 状态向量，共 18 个元素：
       [r_sun (3), v_sun (3), r_earth (3), v_earth (3), r_moon (3), v_moon (3)]
    返回 dy/dt，同样为 18 维向量。
    """
    dydt = np.zeros_like(y)
    # 解析出位置和速度，reshape 成 (3,3) 数组，每一行代表一个天体
    positions = y[0::6].reshape(-1, 3)  # 注意：此处提取方式需调整为正确的索引方式
    # 为简化，我们可以先把状态向量重构为 (n_bodies, 6)
    n_bodies = 3
    state = y.reshape((n_bodies, 6))
    pos = state[:, :3]
    vel = state[:, 3:]
    
    # 对每个天体，其位置的导数即为当前速度
    dydt[0:n_bodies*6:6] = vel[:, 0]
    dydt[1:n_bodies*6:6] = vel[:, 1]
    dydt[2:n_bodies*6:6] = vel[:, 2]
    
    # 计算加速度：对每个天体 i，a_i = sum_{j≠i} -G*m_j*(r_i - r_j)/|r_i-r_j|^3
    acc = np.zeros((n_bodies, 3))
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i == j:
                continue
            r_ij = pos[i] - pos[j]
            distance = np.linalg.norm(r_ij)
            # 避免除零：在真实问题中天体不会重合
            acc[i] += -G_val * masses[j] * r_ij / distance**3
    # 将加速度存入导数向量
    dydt = y.copy().reshape((n_bodies,6))
    dydt[:, :3] = vel  # 位置的导数
    dydt[:, 3:] = acc  # 速度的导数
    return dydt.flatten()

# 设置积分总时间：10 年（秒）
seconds_per_year = 365.25 * 86400
total_time = 1 * seconds_per_year

# 设置积分输出时间点：每一分钟采样一次
dt = 60  # 秒
t_eval = np.arange(0, total_time + dt, dt)

# 使用 DOP853 方法求解微分方程
print("开始积分计算，请耐心等待...")
sol = solve_ivp(derivatives, [0, total_time], y0, method='DOP853', t_eval=t_eval, rtol=1e-9, atol=1e-12)
print("积分计算完成.")

# 整理输出数据，构造一个列表，每个元素为一个时刻的记录
# 记录格式：{"time": ISO格式字符串, "sun": [x,y,z], "earth": [x,y,z], "moon": [x,y,z]}
print("开始处理输出数据...")
output = []
# 预处理数据以提高速度
state_reshaped = sol.y.reshape(3, 6, -1)  # 重塑为 (天体数, 状态维度, 时间点数)

# 每1000个数据点输出一次进度
total_points = len(sol.t)
for i, t_sec in enumerate(sol.t):
    if i % 1000 == 0:
        print(f"已处理 {i}/{total_points} 点 ({i/total_points*100:.1f}%)")
    
    # 计算当前时刻
    current_time = t0 + t_sec * u.s
    
    # 直接从重塑后的数组获取位置数据
    output.append({
        "time": current_time.iso,
        "sun": state_reshaped[0, :3, i].tolist(),
        "earth": state_reshaped[1, :3, i].tolist(),
        "moon": state_reshaped[2, :3, i].tolist()
    })

# 输出到 JSON 文件
print("正在保存数据到JSON文件...")
output_filename = "three_body_positions.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"结果已保存至 {output_filename}")
