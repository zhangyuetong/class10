# 太阳系行星模拟器

这是一个使用Python模拟太阳系行星运动的程序，考虑了所有天体之间的万有引力作用。

## 功能特点

- 基于天文精确数据进行太阳系行星运动模拟
- 考虑所有天体之间的引力相互作用
- 使用天文学库获取准确的天体初始位置和速度
- 采用高精度数值积分器求解微分方程
- 可配置的模拟参数
- 将模拟结果保存为JSON格式
- 可选择输出速度数据和轨道特性
- 支持多种输出单位

## 依赖库

- NumPy
- SciPy
- Astropy
- JSON
- time

## 使用方法

1. 在`config.py`中配置模拟参数
2. 运行`simulate.py`进行模拟

```bash
python simulate.py
```

## 配置参数说明

`config.py`中包含以下可配置参数：

- **时间配置**
  - `START_DATE`: 模拟开始时间
  - `TIME_SCALE`: 时间尺度（'utc', 'tt', 'tdb'）

- **天体配置**
  - `BODIES`: 模拟的天体列表，包含太阳和所有行星（及月球）
  - `MASSES`: 天体质量字典（kg）

- **模拟参数**
  - `SIMULATION_YEARS`: 模拟总时长（年）
  - `SECONDS_PER_YEAR`: 每年的秒数
  - `OUTPUT_INTERVAL`: 输出数据间隔（秒）

- **求解器参数**
  - `SOLVER_METHOD`: 数值积分方法（'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'）
  - `SOLVER_RTOL`: 相对误差容限
  - `SOLVER_ATOL`: 绝对误差容限

- **输出配置**
  - `OUTPUT_FILENAME`: 输出文件名
  - `PROGRESS_INTERVAL`: 进度显示间隔
  - `INCLUDE_VELOCITY`: 是否包含速度数据
  - `COMPUTE_ORBITAL_PROPERTIES`: 是否计算轨道特性
  - `POSITION_UNIT`: 位置单位（'m'或'au'）

## 包含的天体

默认情况下，模拟包含以下天体：

- 太阳 (sun)
- 水星 (mercury)
- 金星 (venus)
- 地球 (earth)
- 月球 (moon)
- 火星 (mars)
- 木星 (jupiter)
- 土星 (saturn)
- 天王星 (uranus)
- 海王星 (neptune)

## 模拟原理

模拟基于以下物理原理：

1. 牛顿万有引力定律：任意两个天体之间存在引力，大小与质量乘积成正比，与距离平方成反比
2. 牛顿第二定律：F = ma，力等于质量乘以加速度
3. 数值积分：使用高精度ODE求解器计算天体在不同时刻的位置和速度

模拟考虑了所有天体之间的引力相互作用，每个天体受到其他所有天体的引力影响。

## 输出数据格式

根据配置，输出的JSON文件包含每个时间点的天体位置数据，格式如下：

### 基本格式（仅位置数据）

```json
[
  {
    "time": "2025-01-01 00:00:00.000",
    "sun": [x, y, z],
    "mercury": [x, y, z],
    "venus": [x, y, z],
    "earth": [x, y, z],
    "moon": [x, y, z],
    "mars": [x, y, z],
    "jupiter": [x, y, z],
    "saturn": [x, y, z],
    "uranus": [x, y, z],
    "neptune": [x, y, z]
  },
  ...
]
```

### 包含速度数据时

当`INCLUDE_VELOCITY=True`时，每个天体的数据包含位置和速度：

```json
[
  {
    "time": "2025-01-01 00:00:00.000",
    "sun": {
      "position": [x, y, z],
      "velocity": [vx, vy, vz]
    },
    // 其他天体数据...
  },
  ...
]
```

### 包含轨道特性时

当`COMPUTE_ORBITAL_PROPERTIES=True`时，每个时间点还会包含系统的能量和角动量信息：

```json
{
  "orbital_properties": {
    "kinetic_energy": value,
    "potential_energy": value,
    "total_energy": value,
    "angular_momentum": [x, y, z]
  }
}
```

## 性能考虑

- 模拟所有行星需要大量计算，特别是当时间跨度较长时
- 建议从小时间范围开始（如0.1年），逐步增加
- 增加`OUTPUT_INTERVAL`可以减少输出数据量
- 高精度积分方法（如'DOP853'）精度更高但计算更慢
- 对于长时间模拟，可能需要使用更高性能的计算机或云服务 