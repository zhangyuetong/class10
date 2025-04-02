import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_eclipse_geometry_3d(earth_pos, moon_pos, sun_pos, R_earth, R_moon, formatted_time):
    """
    日食几何关系图（正确方向+完整显示）
    包含三个子图：全局视图、月球特写、地球特写
    """
    # ========================
    # 1. 坐标系定义与投影
    # ========================
    # 计算月地向量和月日向量
    moon_to_earth = earth_pos - moon_pos
    moon_to_sun = sun_pos - moon_pos
    
    # 建立观测平面坐标系
    normal_vector = np.cross(moon_to_earth, moon_to_sun)
    normal_vector /= np.linalg.norm(normal_vector)  # 归一化

    x_axis = moon_to_earth / np.linalg.norm(moon_to_earth)  # 指向地球的方向
    y_axis = np.cross(normal_vector, x_axis)  # 平面内垂直方向
    y_axis /= np.linalg.norm(y_axis)

    # 坐标投影函数
    def project_to_plane(pos):
        vec = pos - moon_pos
        return np.array([np.dot(vec, x_axis), np.dot(vec, y_axis)])

    # 投影关键点
    earth_plane = project_to_plane(earth_pos)
    moon_plane = np.array([0, 0])  # 月球设为原点
    sun_plane = project_to_plane(sun_pos)

    # ========================
    # 2. 影锥计算
    # ========================
    R_sun = 695700e3  # 太阳半径
    D_sm = np.linalg.norm(sun_pos - moon_pos)  # 月日距离
    
    # 计算太阳方向（注意方向定义）
    sun_direction = (sun_plane - moon_plane) / np.linalg.norm(sun_plane - moon_plane)
    shadow_direction = -sun_direction  # 影锥正确延伸方向

    # 垂直方向向量
    perp_vector = np.array([-shadow_direction[1], shadow_direction[0]])
    
    # 月球边缘切点（上下）
    moon_edge_top = moon_plane + perp_vector * R_moon
    moon_edge_bottom = moon_plane - perp_vector * R_moon

    # 关键角度计算
    alpha_umbra = np.arctan((R_sun - R_moon) / D_sm)  # 本影锥半角
    alpha_penumbra = np.arctan((R_sun + R_moon) / D_sm)  # 半影锥半角

    # 生成影锥边界向量
    def get_shadow_vector(alpha):
        return [
            shadow_direction * np.cos(alpha) + perp_vector * np.sin(alpha),
            shadow_direction * np.cos(alpha) - perp_vector * np.sin(alpha)
        ]
    
    umbra_vectors = get_shadow_vector(alpha_umbra)
    penumbra_vectors = get_shadow_vector(alpha_penumbra)

    # 本影锥长度计算
    umbra_length = R_moon / np.tan(alpha_umbra)
    umbra_tip = moon_plane + shadow_direction * umbra_length

    # ========================
    # 3. 绘图设置
    # ========================
    fig = plt.figure(figsize=(20, 7))
    fig.suptitle(f'日食几何分析 - {formatted_time}', y=1.05, fontsize=16, fontweight='bold')

    # 通用绘图元素
    def plot_common(ax):
        # 天体绘制
        earth = Circle(earth_plane, R_earth, color='#1f77b4', alpha=0.8, label='地球')
        moon = Circle(moon_plane, R_moon, color='#7f7f7f', alpha=0.9, label='月球')
        ax.add_patch(earth)
        ax.add_patch(moon)
        
        # 辅助线
        ax.plot([moon_plane[0], earth_plane[0]], [moon_plane[1], earth_plane[1]], 
                '--', color='#2ca02c', lw=1.5, alpha=0.6, label='月地连线')
        
        # 影锥绘制
        def plot_cone(start, vectors, color, label, length_scale=5):
            for vec in vectors:
                end = start + vec * length_scale * np.linalg.norm(earth_plane - moon_plane)
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                        color=color, lw=1, solid_capstyle='round', label=label)
                label = None  # 避免重复图例
        
        # 本影锥（深红色）
        plot_cone(moon_edge_top, umbra_vectors, '#d62728', '本影锥')
        plot_cone(moon_edge_bottom, umbra_vectors, '#d62728', None)
        
        # 半影锥（橙色）
        plot_cone(moon_edge_top, penumbra_vectors, '#ff7f0e', '半影锥') 
        plot_cone(moon_edge_bottom, penumbra_vectors, '#ff7f0e', None)
        
        ax.axis('equal')
        ax.grid(True, which='both', linestyle=':', alpha=0.7)
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    # ========================
    # 4. 三视图布局
    # ========================
    earth_moon_dist = np.linalg.norm(earth_plane - moon_plane)
    
    # 子图1：全局视图（包括日月和所有线）
    ax1 = fig.add_subplot(131)
    plot_common(ax1)
    ax1.set_title('全局几何关系', pad=15, fontsize=14)
    ax1.set_xlabel('X坐标 (m)', fontsize=12)
    ax1.set_ylabel('Y坐标 (m)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    
    # 确保全局视图显示所有线和日月
    max_dist = earth_moon_dist * 2.0  # 增大显示范围以确保显示完整影锥
    ax1.set_xlim(-max_dist/2, max_dist/2)
    ax1.set_ylim(-max_dist/2, max_dist/2)

    # 子图2：月球特写（只包括月球）
    ax2 = fig.add_subplot(132)
    plot_common(ax2)
    zoom_size = 2 * R_moon  # 更接近月球范围
    ax2.set_xlim(-zoom_size, zoom_size)
    ax2.set_ylim(-zoom_size, zoom_size)
    ax2.set_title('月球附近影锥结构', pad=15, fontsize=14)
    ax2.set_xlabel('X坐标 (m)', fontsize=12)
    
    # 强调切线接触点
    ax2.plot(moon_edge_top[0], moon_edge_top[1], 'o', 
            color='red', markersize=6, markeredgewidth=1.5, 
            markerfacecolor='none', label='切点')
    ax2.plot(moon_edge_bottom[0], moon_edge_bottom[1], 'o', 
            color='red', markersize=6, markeredgewidth=1.5, 
            markerfacecolor='none')
    ax2.legend(loc='upper right', fontsize=10)

    # 子图3：地球特写（只包括地球和本影锥顶点）
    ax3 = fig.add_subplot(133)
    plot_common(ax3)
    
    # 计算地球与本影锥顶点之间的距离
    earth_to_umbra_tip = np.linalg.norm(umbra_tip - earth_plane)
    
    # 调整地球特写的视图范围，确保地球和本影锥顶点都可见
    center_x = (earth_plane[0] + umbra_tip[0]) / 2
    center_y = (earth_plane[1] + umbra_tip[1]) / 2
    
    # 确定适当的显示范围，以包含地球和本影锥顶点
    margin = max(1.2 * R_earth, 0.1 * earth_to_umbra_tip)
    half_width = earth_to_umbra_tip / 2 + margin
    
    ax3.set_xlim(center_x - half_width, center_x + half_width)
    ax3.set_ylim(center_y - half_width, center_y + half_width)
    ax3.set_title('地球附近影锥分布', pad=15, fontsize=14)
    ax3.set_xlabel('X坐标 (m)', fontsize=12)
    
    # 标注本影锥顶点
    ax3.plot(umbra_tip[0], umbra_tip[1], '*', 
            color='darkred', markersize=15, markeredgewidth=1,
            label=f'本影顶点\n(距地心:{np.linalg.norm(umbra_tip-earth_plane)/1000:,.1f} km)')
    
    # 添加地球表面参考线
    theta = np.linspace(0, 2*np.pi, 100)
    ax3.plot(earth_plane[0] + R_earth*np.cos(theta), 
            earth_plane[1] + R_earth*np.sin(theta),
            '--', color='blue', alpha=0.5, lw=1.5, label='地球表面')
    
    ax3.legend(loc='upper right', fontsize=10)

    # ========================
    # 5. 输出设置
    # ========================
    plt.tight_layout()
    
    # 添加图示说明，先提及本影锥
    fig.text(0.02, 0.98, "天文图示说明：\n• 红色线：本影锥（完全遮挡）\n• 橙色线：半影锥（部分遮挡）", 
            ha='left', va='top', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 保存图表
    output_path = f'Eclipse_Geometry_{formatted_time.replace(":","_")}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"图表已保存至: {output_path}")