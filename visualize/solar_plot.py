def plot_eclipse_geometry_3d(earth_pos, moon_pos, sun_pos, R_earth, R_moon, formatted_time):
    """
    绘制日食几何关系图（在日月地三者确定的平面上）

    参数:
    earth_pos - 地球位置向量（3D坐标，单位：米）
    moon_pos  - 月球位置向量（3D坐标，单位：米）
    sun_pos   - 太阳位置向量（3D坐标，单位：米）
    R_earth   - 地球半径（单位：米）
    R_moon    - 月球半径（单位：米）
    formatted_time - 格式化的时间字符串，用于图标题及文件名
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    # -------------------------------
    # 1. 定义日月地平面坐标系
    # -------------------------------
    # 计算平面法向量（月地向量 × 月日向量）
    moon_to_earth = earth_pos - moon_pos
    moon_to_sun = sun_pos - moon_pos
    normal_vector = np.cross(moon_to_earth, moon_to_sun)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 单位化

    # 定义平面内的x轴（月地向量方向）
    x_axis = moon_to_earth
    x_axis = x_axis / np.linalg.norm(x_axis)

    # 定义平面内的y轴（与x轴和法向量垂直）
    y_axis = np.cross(normal_vector, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # -------------------------------
    # 2. 将各天体位置投影到平面坐标系
    # -------------------------------
    def project_to_plane(pos):
        # 计算相对于月球位置的向量
        vec = pos - moon_pos
        # 计算在平面坐标系中的坐标
        x = np.dot(vec, x_axis)
        y = np.dot(vec, y_axis)
        return np.array([x, y])

    earth_plane = project_to_plane(earth_pos)
    moon_plane = project_to_plane(moon_pos)  # 月球在原点
    sun_plane = project_to_plane(sun_pos)

    # -------------------------------
    # 3. 计算太阳光方向（在平面坐标系中）
    # -------------------------------
    sun_direction = sun_plane - moon_plane
    sun_direction = sun_direction / np.linalg.norm(sun_direction)
    base_vector = -sun_direction  # 影锥延伸方向

    # -------------------------------
    # 4. 计算地球与月球在平面内的距离
    # -------------------------------
    earth_moon_dist_plane = np.linalg.norm(earth_plane - moon_plane)

    # -------------------------------
    # 5. 绘制地球和月球的圆形表示
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    earth_circle = Circle(earth_plane, R_earth, color='blue', alpha=0.7, label='地球')
    ax.add_patch(earth_circle)
    moon_circle = Circle(moon_plane, R_moon, color='gray', alpha=0.7, label='月球')
    ax.add_patch(moon_circle)

    # -------------------------------
    # 6. 计算影锥的角度
    # -------------------------------
    R_sun = 695700e3  # 太阳半径
    D_sm = np.linalg.norm(sun_pos - moon_pos)  # 月日距离

    alpha_umbra = np.arcsin((R_sun - R_moon) / D_sm)   # 本影锥半顶角
    alpha_penumbra = np.arcsin((R_sun + R_moon) / D_sm)  # 半影锥半顶角

    # -------------------------------
    # 7. 计算在平面内垂直于影锥主轴的方向
    # -------------------------------
    perp_vector = np.array([-base_vector[1], base_vector[0]])

    # -------------------------------
    # 8. 设定延伸线的长度
    # -------------------------------
    line_length = earth_moon_dist_plane * 2.0

    # -------------------------------
    # 9. 绘制本影锥边界
    # -------------------------------
    umbra_vector1 = base_vector * np.cos(alpha_umbra) + perp_vector * np.sin(alpha_umbra)
    umbra_vector2 = base_vector * np.cos(alpha_umbra) - perp_vector * np.sin(alpha_umbra)
    
    umbra_end1 = moon_plane + umbra_vector1 * line_length
    umbra_end2 = moon_plane + umbra_vector2 * line_length
    
    ax.plot([moon_plane[0], umbra_end1[0]], [moon_plane[1], umbra_end1[1]], 'r-', lw=2, label='本影锥边界')
    ax.plot([moon_plane[0], umbra_end2[0]], [moon_plane[1], umbra_end2[1]], 'r-', lw=2)

    # -------------------------------
    # 10. 绘制半影锥边界
    # -------------------------------
    penumbra_vector1 = base_vector * np.cos(alpha_penumbra) + perp_vector * np.sin(alpha_penumbra)
    penumbra_vector2 = base_vector * np.cos(alpha_penumbra) - perp_vector * np.sin(alpha_penumbra)
    
    penumbra_end1 = moon_plane + penumbra_vector1 * line_length
    penumbra_end2 = moon_plane + penumbra_vector2 * line_length
    
    ax.plot([moon_plane[0], penumbra_end1[0]], [moon_plane[1], penumbra_end1[1]], 'y-', lw=2, label='半影锥边界')
    ax.plot([moon_plane[0], penumbra_end2[0]], [moon_plane[1], penumbra_end2[1]], 'y-', lw=2)

    # -------------------------------
    # 11. 绘制辅助线
    # -------------------------------
    # 绘制月地连线
    ax.plot([moon_plane[0], earth_plane[0]], [moon_plane[1], earth_plane[1]], 'k--', lw=1)
    
    # 绘制太阳方向（反向延长）
    sun_line_end = moon_plane - sun_direction * line_length * 0.5
    ax.plot([moon_plane[0], sun_line_end[0]], [moon_plane[1], sun_line_end[1]], 'k:', lw=1, label='太阳方向')

    # -------------------------------
    # 12. 设置图表属性及保存图像
    # -------------------------------
    ax.set_title(f'日食几何关系图（日月地平面） - {formatted_time}')
    ax.set_xlabel('X坐标 (米)')
    ax.set_ylabel('Y坐标 (米)')
    ax.axis('equal')
    ax.legend()
    ax.grid(True)
    
    # 添加坐标系说明
    ax.text(0.05, 0.95, "坐标系说明:\nX轴: 月地方向\nY轴: 平面内垂直方向", 
            transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # 保存图像
    fig.savefig(f'eclipse_geometry_3d_{formatted_time.replace(":", "_")}.png', dpi=300)
    plt.close(fig)