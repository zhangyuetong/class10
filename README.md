# 三体问题模拟器

这是一个使用Python模拟三体问题（太阳-地球-月球系统）的程序。

## 功能特点

- 基于天文精确数据进行三体运动模拟
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
  - `BODIES`: 模拟的天体列表
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

## 输出数据格式

根据配置，输出的JSON文件包含每个时间点的天体位置数据，格式如下：

### 基本格式（仅位置数据）

```json
[
  {
    "time": "2025-01-01 00:00:00.000",
    "sun": [x, y, z],
    "earth": [x, y, z],
    "moon": [x, y, z]
  },
  ...
]
```

### 包含速度数据时

```json
[
  {
    "time": "2025-01-01 00:00:00.000",
    "sun": {
      "position": [x, y, z],
      "velocity": [vx, vy, vz]
    },
    "earth": {
      "position": [x, y, z],
      "velocity": [vx, vy, vz]
    },
    "moon": {
      "position": [x, y, z],
      "velocity": [vx, vy, vz]
    }
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

## 添加其他天体

在`config.py`文件中，可以通过修改`BODIES`列表和`MASSES`字典来添加更多天体。例如添加火星：

```python
BODIES = ['sun', 'earth', 'moon', 'mars']
MASSES = {
    'sun': M_sun.value,
    'earth': M_earth.value,
    'moon': 7.34767309e22,
    'mars': 6.4171e23
}
``` 