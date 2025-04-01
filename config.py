"""
模拟配置文件，包含太阳系行星模拟所需的所有参数
"""

import astropy.units as u
from astropy.constants import G, M_sun, M_earth

# 时间配置
START_DATE = "2025-01-01T00:00:00"  # 模拟开始时间
TIME_SCALE = 'utc'  # 时间尺度 (可选: 'utc', 'tt', 'tdb')

# 手动定义天体质量 (kg)
# 参考来源: NASA 和 IAU 数据
M_mercury = 3.3011e23  # 水星质量
M_venus = 4.8675e24    # 金星质量
M_moon = 7.34767309e22 # 月球质量
M_mars = 6.4171e23     # 火星质量
M_jupiter = 1.8982e27  # 木星质量
M_saturn = 5.6834e26   # 土星质量
M_uranus = 8.6810e25   # 天王星质量
M_neptune = 1.02413e26 # 海王星质量

# 天体配置 - 包含太阳和所有行星
BODIES = [
    'sun',
    'mercury', 'venus', 'earth', 'mars',
    'jupiter', 'saturn', 'uranus', 'neptune',
    'moon'
]  # 模拟天体列表

# 天体质量字典 (kg)
MASSES = {
    'sun': M_sun.value,                  # 太阳质量
    'mercury': M_mercury,                # 水星质量
    'venus': M_venus,                    # 金星质量
    'earth': M_earth.value,              # 地球质量
    'moon': M_moon,                      # 月球质量
    'mars': M_mars,                      # 火星质量
    'jupiter': M_jupiter,                # 木星质量
    'saturn': M_saturn,                  # 土星质量
    'uranus': M_uranus,                  # 天王星质量
    'neptune': M_neptune,                # 海王星质量
}

# 模拟参数
SIMULATION_YEARS = 1                  # 模拟时长（年）- 增加时长以便观察更明显的变化
SECONDS_PER_YEAR = 365.25 * 86400        # 每年的秒数
OUTPUT_INTERVAL = 3600                   # 输出时间间隔（秒）- 增加间隔减少数据量

# 求解器参数
# 求解器方法选项：'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
SOLVER_METHOD = 'DOP853'                 # 积分方法，DOP853为高精度方法
SOLVER_RTOL = 1e-9                       # 相对误差容限
SOLVER_ATOL = 1e-12                      # 绝对误差容限

# 输出配置
OUTPUT_FILENAME = "solar_system_positions.json"  # 输出文件名
PROGRESS_INTERVAL = 100                  # 进度显示间隔（数据点数）

# 以下是一些可选配置，可根据需要启用

# 是否包含速度数据在输出中
INCLUDE_VELOCITY = True

# 是否自动计算轨道特性（如角动量、能量）
COMPUTE_ORBITAL_PROPERTIES = True

# 输出单位配置
# 可选：'m'（米）或'au'（天文单位）
POSITION_UNIT = 'au'  # 使用天文单位更适合太阳系尺度 