"""
模拟配置文件，包含太阳系行星模拟所需的所有参数
"""

# 时间配置
START_DATE = "2025-01-01T00:00:00"  # 模拟开始时间
TIME_SCALE = 'utc'  # 时间尺度 (可选: 'utc', 'tt', 'tdb')
SIMULATION_YEARS = 5                 # 模拟时长（年）- 增加时长以便观察更明显的变化
SECONDS_PER_YEAR = 365.24 * 86400        # 每年的秒数

# 天体配置 - 包含太阳和所有行星
BODIES = [
    'sun',
    'mercury', 'venus', 'earth', 'mars',
    'jupiter', 'saturn', 'uranus', 'neptune',
    'moon'
]  # 模拟天体列表

GM_DICT = {
    'sun':    1.327124400189e20,  # [m^3/s^2]
    'mercury': 2.2031868551e13,
    'venus':   3.24858592000e14,
    'earth':   3.98600435507e14,
    'moon':    4.902800118e12,
    'mars':    4.2828375816e13,
    'jupiter': 1.267127641e17,
    'saturn':  3.79405848418e16,
    'uranus':  5.794556400000e15,
    'neptune': 6.836527100580e15,
    # 视需求还可以包含 pluto, ceres 等
}

C_LIGHT = 299792458.0  # 光速（m/s）

# 求解器参数
# 求解器方法选项：'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
SOLVER_METHOD = 'DOP853'                 # 积分方法，DOP853为高精度方法
SOLVER_RTOL = 1e-13                       # 相对误差容限
SOLVER_ATOL = 1e-16                      # 绝对误差容限
MAX_STEP_TIME = 3600                    # simulate_with_event里面的检测周期（秒）

RELATIVITY = True # 相对论修正开关

# 误差分析间隔（秒）
ERROR_EVAL_INTERVAL = 86400 * 30
# 误差分析天体列表
ERROR_ANALYSIS_BODIES = [
    'sun',
    'mercury', 'venus', 'earth', 'mars',
    'jupiter', 'saturn', 'uranus', 'neptune',
    'moon'
]