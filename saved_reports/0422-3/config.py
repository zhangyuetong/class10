"""
模拟配置文件，包含太阳系行星模拟所需的所有参数
"""

# 时间配置
START_DATE = "2025-01-01T00:00:00"  # 模拟开始时间
TIME_SCALE = 'utc'  # 时间尺度 (可选: 'utc', 'tt', 'tdb')
SIMULATION_YEARS = 50                 # 模拟时长（年）- 增加时长以便观察更明显的变化
SECONDS_PER_YEAR = 365.24 * 86400        # 每年的秒数

# 天体配置 - 包含太阳和所有行星
BODIES = [
    'sun',
    'mercury', 'venus', 'earth', 'mars',
    'jupiter', 'saturn', 'uranus', 'neptune',
    'moon', 'pluto',
]  # 模拟天体列表

GM_DICT = {
    'sun':    1.32712440041279419e20,  # [m^3/s^2]
    'mercury':  2.2031868551e13,
    'venus':    3.24858592000e14,
    'earth':   3.98600435507e14,
    'moon':     4.902800118e12,
    'mars':    4.2828375816e13, # system
    'jupiter': 1.26712764100000e17, # system
    'saturn':  3.7940584841800e16, # system
    'uranus':   5.794556400000e15, # system
    'neptune': 6.836527100580e15, # system
    'pluto':  9.75500000e11, # system
}

C_LIGHT = 299792458.0  # 光速（m/s）

# 求解器参数
# 求解器方法选项：'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
SOLVER_METHOD = 'DOP853'                 # 积分方法，DOP853为高精度方法
SOLVER_RTOL = 1e-13                       # 相对误差容限
SOLVER_ATOL = 1e-16                      # 绝对误差容限
MAX_STEP_TIME = 86400                    # simulate_with_event里面的检测周期（秒）

RELATIVITY_SWITCH = True # 相对论修正开关

MAX_SIN_ANGLE = 0.05

# 误差分析间隔（秒）
ERROR_EVAL_INTERVAL = 86400 * 30
# 误差分析天体列表
ERROR_ANALYSIS_BODIES = [
    'sun',
    'mercury', 'venus', 'earth', 'mars',
    'jupiter', 'saturn', 'uranus', 'neptune',
    'moon'
]

J2_SWITCH = True
J2_DICT = {
    'sun':      2.0e-7,       # 太阳
    'mercury':  0.0,          # 水星（可忽略）
    'venus':    0.0,          # 金星（可忽略）
    'earth':    1.08263e-3,   # 地球
    'moon':     2.034e-4,     # 月球
    'mars':     1.96045e-3,   # 火星
    'jupiter':  1.4736e-2,    # 木星
    'saturn':   1.6298e-2,    # 土星
    'uranus':   3.34343e-3,   # 天王星
    'neptune':  3.411e-3,     # 海王星
    'pluto':    0.0,          # 冥王星（可忽略）
}

RADIUS_DICT = {
    'sun':      6.9634e8,     # 太阳赤道半径 (m)
    'mercury':  2.4397e6,     # 水星
    'venus':    6.0518e6,     # 金星
    'earth':    6.3781363e6,  # 地球
    'moon':     1.7371e6,     # 月球
    'mars':     3.3895e6,     # 火星
    'jupiter':  7.1492e7,     # 木星赤道半径
    'saturn':   6.0268e7,     # 土星赤道半径
    'uranus':   2.5559e7,     # 天王星赤道半径
    'neptune':  2.4764e7,     # 海王星赤道半径
    'pluto':    1.1883e6,     # 冥王星
}