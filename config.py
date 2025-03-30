"""
模拟配置文件，包含三体问题模拟所需的所有参数
"""

import astropy.units as u
from astropy.constants import G, M_sun, M_earth

# 时间配置
START_DATE = "2025-01-01T00:00:00"  # 模拟开始时间
TIME_SCALE = 'utc'  # 时间尺度 (可选: 'utc', 'tt', 'tdb')

# 天体配置
BODIES = ['sun', 'earth', 'moon']  # 模拟天体
# 天体质量字典 (kg)
MASSES = {
    'sun': M_sun.value,           # 太阳质量
    'earth': M_earth.value,       # 地球质量
    'moon': 7.34767309e22,        # 月球质量
    # 可根据需要添加其他天体
    # 'mercury': 3.3011e23,       # 水星质量
    # 'venus': 4.8675e24,         # 金星质量
    # 'mars': 6.4171e23,          # 火星质量
    # 'jupiter': 1.8982e27,       # 木星质量
}

# 模拟参数
SIMULATION_YEARS = 0.01           # 模拟时长（年）
SECONDS_PER_YEAR = 365.25 * 86400  # 每年的秒数
OUTPUT_INTERVAL = 60              # 输出时间间隔（秒）

# 求解器参数
# 求解器方法选项：'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
SOLVER_METHOD = 'DOP853'          # 积分方法，DOP853为高精度方法
SOLVER_RTOL = 1e-9                # 相对误差容限
SOLVER_ATOL = 1e-12               # 绝对误差容限

# 输出配置
OUTPUT_FILENAME = "three_body_positions.json"  # 输出文件名
PROGRESS_INTERVAL = 1000          # 进度显示间隔（数据点数）

# 以下是一些可选配置，可根据需要启用

# 是否包含速度数据在输出中
INCLUDE_VELOCITY = False

# 是否自动计算轨道特性（如角动量、能量）
COMPUTE_ORBITAL_PROPERTIES = False

# 输出单位配置
# 可选：'m'（米）或'au'（天文单位）
POSITION_UNIT = 'm' 