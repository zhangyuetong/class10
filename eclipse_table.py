#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 读取日食月食数据
with open('eclipse_events.json', 'r', encoding='utf-8') as f:
    eclipse_data = json.load(f)

# 定义函数将相邻时间的相同类型日食/月食合并为一个事件
def group_eclipse_events(events_list):
    if not events_list:
        return []
    
    # 按时间排序
    sorted_events = sorted(events_list, key=lambda x: x['time'])
    
    grouped_events = []
    current_event = {
        'start_time': sorted_events[0]['time'],
        'end_time': sorted_events[0]['time'],
        'types': {sorted_events[0]['type']}
    }
    
    for i in range(1, len(sorted_events)):
        current_time = datetime.fromisoformat(sorted_events[i]['time'].replace('.000', ''))
        last_time = datetime.fromisoformat(sorted_events[i-1]['time'].replace('.000', ''))
        
        # 如果时间间隔小于24小时，认为是同一事件
        if (current_time - last_time).total_seconds() <= 24 * 3600:
            current_event['end_time'] = sorted_events[i]['time']
            current_event['types'].add(sorted_events[i]['type'])
        else:
            # 开始新事件
            grouped_events.append(current_event)
            current_event = {
                'start_time': sorted_events[i]['time'],
                'end_time': sorted_events[i]['time'],
                'types': {sorted_events[i]['type']}
            }
    
    # 添加最后一个事件
    grouped_events.append(current_event)
    
    return grouped_events

# 处理日食和月食事件
solar_eclipses = group_eclipse_events(eclipse_data['solar'])
lunar_eclipses = group_eclipse_events(eclipse_data['lunar'])

# 创建结果数据
result_data = []

# 添加日食事件
for eclipse in solar_eclipses:
    start_dt = datetime.fromisoformat(eclipse['start_time'].replace('.000', ''))
    end_dt = datetime.fromisoformat(eclipse['end_time'].replace('.000', ''))
    duration = (end_dt - start_dt).total_seconds() / 3600  # 持续时间（小时）
    
    # 类型信息
    type_info = ""
    if 'Total' in eclipse['types']:
        if 'Partial' in eclipse['types']:
            type_info = "日全食/偏食"
        else:
            type_info = "日全食"
    elif 'Annular' in eclipse['types']:
        if 'Partial' in eclipse['types']:
            type_info = "日环食/偏食"
        else:
            type_info = "日环食"
    else:
        type_info = "日偏食"
    
    # 格式化持续时间区间
    time_range = f"{start_dt.strftime('%Y-%m-%d %H:%M')} - {end_dt.strftime('%H:%M')}"
    
    result_data.append({
        'date': start_dt.strftime('%Y-%m-%d'),
        'event_type': '日食',
        'specific_type': type_info,
        'time_range': time_range,
        'duration_hours': duration
    })

# 添加月食事件
for eclipse in lunar_eclipses:
    start_dt = datetime.fromisoformat(eclipse['start_time'].replace('.000', ''))
    end_dt = datetime.fromisoformat(eclipse['end_time'].replace('.000', ''))
    duration = (end_dt - start_dt).total_seconds() / 3600  # 持续时间（小时）
    
    # 类型信息
    type_info = ""
    if 'Total' in eclipse['types']:
        if 'Partial' in eclipse['types'] and 'Penumbral' in eclipse['types']:
            type_info = "月全食/偏食/半影食"
        elif 'Partial' in eclipse['types']:
            type_info = "月全食/偏食"
        elif 'Penumbral' in eclipse['types']:
            type_info = "月全食/半影食"
        else:
            type_info = "月全食"
    elif 'Partial' in eclipse['types']:
        if 'Penumbral' in eclipse['types']:
            type_info = "月偏食/半影食"
        else:
            type_info = "月偏食"
    else:
        type_info = "月半影食"
    
    # 格式化持续时间区间
    time_range = f"{start_dt.strftime('%Y-%m-%d %H:%M')} - {end_dt.strftime('%H:%M')}"
    
    result_data.append({
        'date': start_dt.strftime('%Y-%m-%d'),
        'event_type': '月食',
        'specific_type': type_info,
        'time_range': time_range,
        'duration_hours': duration
    })

# 创建DataFrame并按日期排序
df = pd.DataFrame(result_data).sort_values('date')

# 显示前20行结果
print("日月食事件表格（显示前20个事件）:")
print(df.head(20).to_string(index=False))

# 保存完整结果到Excel
df.to_excel('eclipse_events_table.xlsx', index=False, sheet_name='日月食事件')

# 统计不同类型食的数量
event_counts = df['event_type'].value_counts()
specific_counts = df['specific_type'].value_counts()

print("\n事件类型统计:")
print(event_counts)
print("\n具体食类型统计:")
print(specific_counts)

# 绘制不使用中文的图表版本
plt.figure(figsize=(12, 5))

# 创建英文标题版本
plt.subplot(1, 2, 1)
event_counts.plot.pie(autopct='%1.1f%%', textprops={'fontsize': 14})
plt.title('Solar vs Lunar Eclipse Ratio', fontsize=16)

plt.subplot(1, 2, 2)
specific_counts.head(6).plot.bar(color='skyblue')
plt.title('Main Eclipse Types', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 保存图表
plt.savefig('eclipse_events_stats_en.png', dpi=300, bbox_inches='tight')

try:
    # 尝试使用中文绘图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    event_counts.plot.pie(autopct='%1.1f%%', textprops={'fontsize': 14})
    plt.title('日食与月食比例', fontsize=16)
    
    plt.subplot(1, 2, 2)
    specific_counts.head(6).plot.bar(color='skyblue')
    plt.title('主要食类型分布', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # 保存中文图表
    plt.savefig('eclipse_events_stats_cn.png', dpi=300, bbox_inches='tight')
except Exception as e:
    print(f"中文图表生成失败: {e}")

print("\n已生成日月食事件表格和统计图表") 