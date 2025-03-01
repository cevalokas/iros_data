import numpy as np
import torch
import pandas as pd
from pathlib import Path
from model_trainer import ImprovedMarkerPredictor as MarkerPredictor, preprocess_coordinates
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_model_and_scalers():
    # 加载模型
    checkpoint = torch.load('marker_predictor.pth')
    
    # 获取正确的输入维度
    input_size = checkpoint['input_size']
    camera_dim = 512
    imu_dim = input_size - camera_dim
    
    # 使用正确的维度初始化模型
    model = MarkerPredictor(camera_dim=camera_dim, imu_dim=imu_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)  # 将模型移到正确的设备上
    model.eval()
    
    # 加载标准化器
    X_scaler = joblib.load('X_scaler.joblib')
    y_scaler = joblib.load('y_scaler.joblib')
    
    return model, X_scaler, y_scaler

def plot_trajectory_comparison(y_test, y_pred, save_dir):
    """绘制每个点的真实轨迹和预测轨迹对比"""
    coords = ['X', 'Y', 'Z']
    
    # 已经是相对坐标了，直接绘制
    for point_idx in range(9):
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle(f'Point {point_idx} Relative Position Trajectory')
        
        for i, coord in enumerate(coords):
            true_vals = y_test[:, point_idx*3 + i]
            pred_vals = y_pred[:, point_idx*3 + i]
            
            axes[i].plot(true_vals, label='True', alpha=0.7)
            axes[i].plot(pred_vals, label='Predicted', alpha=0.7)
            axes[i].set_title(f'Relative {coord} Position to Point 0')
            axes[i].set_xlabel('Frame')
            axes[i].set_ylabel(f'Relative {coord} Position (mm)')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'point_{point_idx}_relative_trajectory.png')
        plt.close()

def visualize_3d_comparison(y_test, y_pred, frame_idx, save_dir):
    """3D可视化某一帧的真实位置和预测位置"""
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取当前帧的真实值和预测值（已经是相对坐标）
    true_points = y_test[frame_idx].reshape(-1, 3)
    pred_points = y_pred[frame_idx].reshape(-1, 3)
    
    # 绘制真实点
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], 
              c='blue', marker='o', s=100, label='True Relative Position')
    
    # 绘制预测点
    ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
              c='red', marker='^', s=100, label='Predicted Relative Position')
    
    # 连接对应的点
    for i in range(9):
        ax.plot([true_points[i, 0], pred_points[i, 0]],
                [true_points[i, 1], pred_points[i, 1]],
                [true_points[i, 2], pred_points[i, 2]],
                'g--', alpha=0.3)
    
    ax.set_xlabel('Relative X (mm)')
    ax.set_ylabel('Relative Y (mm)')
    ax.set_zlabel('Relative Z (mm)')
    ax.set_title(f'Frame {frame_idx} - True vs Predicted Relative Positions')
    ax.legend()
    
    plt.savefig(save_dir / f'3d_comparison_frame_{frame_idx}.png')
    plt.close()

def calculate_shape_metrics(true_pos, pred_pos):
    """计算形状相似度指标"""
    # 将坐标重塑为(n_frames, n_points, 3)
    true_pos = true_pos.reshape(-1, 9, 3)
    pred_pos = pred_pos.reshape(-1, 9, 3)
    
    # 计算点之间的距离矩阵
    def get_distance_matrix(positions):
        n_points = positions.shape[1]
        dist_matrix = np.zeros((positions.shape[0], n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                dist_matrix[:, i, j] = np.linalg.norm(positions[:, i] - positions[:, j], axis=1)
        return dist_matrix
    
    true_dist = get_distance_matrix(true_pos)
    pred_dist = get_distance_matrix(pred_pos)
    
    # 计算相对距离误差
    rel_dist_error = np.abs(true_dist - pred_dist) / (true_dist + 1e-6)
    mean_rel_error = np.mean(rel_dist_error)
    
    return {
        'mean_relative_distance_error': mean_rel_error,
        'distance_matrices': (true_dist, pred_dist)
    }

def visualize_predictions(true_positions, pred_positions, output_path, interval=50):
    """创建真实位置和预测位置的对比动画"""
    fig = plt.figure(figsize=(20, 10))
    
    views = [
        (30, 45, "Perspective View"),
        (0, 0, "Front View (XY)")
    ]
    
    def update(frame):
        for ax_idx, (elev, azim, title) in enumerate(views):
            ax = fig.add_subplot(1, 2, ax_idx+1, projection='3d')
            ax.clear()
            
            # 获取当前帧的点
            true_frame = true_positions[frame].reshape(-1, 3)
            pred_frame = pred_positions[frame].reshape(-1, 3)
            
            # 绘制点
            ax.scatter(true_frame[:, 0], true_frame[:, 1], true_frame[:, 2],
                      c='blue', marker='o', s=100, label='True')
            ax.scatter(pred_frame[:, 0], pred_frame[:, 1], pred_frame[:, 2],
                      c='red', marker='^', s=100, label='Predicted')
            
            # 添加点的标签
            for i in range(9):
                ax.text(true_frame[i, 0], true_frame[i, 1], true_frame[i, 2],
                       f'{i}', color='blue')
                ax.text(pred_frame[i, 0], pred_frame[i, 1], pred_frame[i, 2],
                       f'{i}', color='red')
            
            # 设置视角和标签
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title(f'{title}\nFrame {frame}')
            
            if ax_idx == 0:
                ax.legend()
    
    anim = FuncAnimation(fig, update, frames=range(len(true_positions)),
                        interval=interval)
    
    plt.tight_layout()
    anim.save(output_path, writer='pillow', dpi=100)
    plt.close()

def plot_point_trajectories(y_test, y_pred, output_path):
    """为每个点生成轨迹对比图"""
    fig, axes = plt.subplots(3, 3, figsize=(20, 20))
    fig.suptitle('Trajectory Comparison for Each Point', fontsize=16)
    
    # 将坐标重塑为(n_frames, n_points, 3)
    y_test = y_test.reshape(-1, 9, 3)
    y_pred = y_pred.reshape(-1, 9, 3)
    
    # 为每个点创建子图
    for point_idx in range(9):
        row = point_idx // 3
        col = point_idx % 3
        ax = axes[row, col]
        
        # 获取当前点的真实轨迹和预测轨迹
        true_traj = y_test[:, point_idx]
        pred_traj = y_pred[:, point_idx]
        
        # 绘制X,Y,Z坐标随时间的变化
        time = np.arange(len(true_traj))
        
        # 绘制三个坐标分量
        ax.plot(time, true_traj[:, 0], 'b-', label='True X', alpha=0.7)
        ax.plot(time, pred_traj[:, 0], 'b--', label='Pred X', alpha=0.7)
        
        ax.plot(time, true_traj[:, 1], 'g-', label='True Y', alpha=0.7)
        ax.plot(time, pred_traj[:, 1], 'g--', label='Pred Y', alpha=0.7)
        
        ax.plot(time, true_traj[:, 2], 'r-', label='True Z', alpha=0.7)
        ax.plot(time, pred_traj[:, 2], 'r--', label='Pred Z', alpha=0.7)
        
        # 设置标题和标签
        ax.set_title(f'Point {point_idx}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Position (mm)')
        ax.grid(True)
        
        # 计算每个坐标轴的误差
        errors = np.mean(np.abs(true_traj - pred_traj), axis=0)
        ax.text(0.02, 0.98, f'Mean Errors:\nX: {errors[0]:.2f}mm\nY: {errors[1]:.2f}mm\nZ: {errors[2]:.2f}mm',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 只在第一个子图显示图例
        if point_idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_scenario(test_path, model, X_scaler, y_scaler):
    """评估单个场景的性能"""
    X_test = np.load(test_path / "processed_X.npy")
    y_test = np.load(test_path / "processed_y.npy")
    
    # 1. 先转换为相对坐标
    y_test_relative = preprocess_coordinates(y_test)
    
    # 2. 然后进行标准化
    X_test_scaled = X_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test_relative)
    
    # 3. 预测
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_pred_scaled = model(X_test_tensor).numpy()
    
    # 4. 反标准化
    y_pred_relative = y_scaler.inverse_transform(y_pred_scaled)
    
    # 5. 计算距离误差和相对误差
    distance_error = np.mean(np.sqrt(np.sum((y_test_relative - y_pred_relative) ** 2, axis=1)))
    relative_error = distance_error / np.mean(np.sqrt(np.sum(y_test_relative ** 2, axis=1))) * 100
    
    # 6. 计算数据量
    data_volume = len(X_test)
    
    return {
        'data_volume': data_volume,
        'distance_error': distance_error,
        'relative_error': relative_error
    }

def calculate_point_cloud_metrics(y_true, y_pred):
    """计算点云相关的评估指标"""
    # 将坐标重塑为(n_frames, n_points, 3)的形状
    y_true = y_true.reshape(-1, 9, 3)
    y_pred = y_pred.reshape(-1, 9, 3)
    
    # 计算每帧的距离指标
    point_distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=2))  # (n_frames, n_points)
    avg_distances_per_frame = np.mean(point_distances, axis=1)  # (n_frames,)
    max_distances_per_frame = np.max(point_distances, axis=1)   # (n_frames,)
    
    # Chamfer距离
    def compute_chamfer(pc1, pc2):
        distances1 = np.sqrt(np.min([np.sum((p1 - pc2) ** 2, axis=1) for p1 in pc1], axis=1))
        distances2 = np.sqrt(np.min([np.sum((p2 - pc1) ** 2, axis=1) for p2 in pc1], axis=1))
        return np.mean(distances1) + np.mean(distances2)
    
    chamfer_distances = [compute_chamfer(true_frame, pred_frame) 
                        for true_frame, pred_frame in zip(y_true, y_pred)]
    
    # Hausdorff距离
    def compute_hausdorff(pc1, pc2):
        forward = np.max([np.min([np.sqrt(np.sum((p1 - pc2) ** 2, axis=1)) for p1 in pc1], axis=1)])
        backward = np.max([np.min([np.sqrt(np.sum((p2 - pc1) ** 2, axis=1)) for p2 in pc1], axis=1)])
        return max(forward, backward)
    
    hausdorff_distances = [compute_hausdorff(true_frame, pred_frame) 
                          for true_frame, pred_frame in zip(y_true, y_pred)]
    
    return {
        'avg_distance': np.mean(avg_distances_per_frame),
        'max_distance': np.max(max_distances_per_frame),
        'chamfer_distance': np.mean(chamfer_distances),
        'hausdorff_distance': np.max(hausdorff_distances),
        'avg_distance_var': np.var(avg_distances_per_frame),
        'max_distance_var': np.var(max_distances_per_frame),
        'chamfer_distance_var': np.var(chamfer_distances),
        'hausdorff_distance_var': np.var(hausdorff_distances)
    }

def print_perception_table(results):
    """打印不同感知方式的性能表格"""
    print("\nTABLE II - PERFORMANCE METRICS FOR DIFFERENT PERCEPTIONS")
    print("="*240)  # 增加宽度
    print(f"{'Perception':<15} {'Data Volume':<12} {'Distance Error':<15} {'Error Variance':<15} "
          f"{'Relative Error':<15} {'Avg Distance':<15} {'Avg Dist Var':<15} {'Max Distance':<15} "
          f"{'Max Dist Var':<15} {'Chamfer Dist':<15} {'Chamfer Var':<15} {'Hausdorff Dist':<15} "
          f"{'Hausdorff Var':<15}")
    print("-"*240)
    
    for perception, metrics in results.items():
        print(f"{perception:<15} {metrics['data_volume']:<12d} "
              f"{metrics['distance_error']:<15.2f} {metrics['error_variance']:<15.2f} "
              f"{metrics['relative_error']:<15.2f} {metrics['avg_distance']:<15.2f} "
              f"{metrics['avg_distance_var']:<15.2f} {metrics['max_distance']:<15.2f} "
              f"{metrics['max_distance_var']:<15.2f} {metrics['chamfer_distance']:<15.2f} "
              f"{metrics['chamfer_distance_var']:<15.2f} {metrics['hausdorff_distance']:<15.2f} "
              f"{metrics['hausdorff_distance_var']:<15.2f}")
    print("-"*240)

def print_scenario_table(results):
    """打印不同场景的性能表格"""
    print("\nTABLE III - PERFORMANCE METRICS FOR DIFFERENT SCENARIOS")
    print("="*240)
    print(f"{'Scenario':<20} {'Data Volume':<12} {'Distance Error':<15} {'Error Variance':<15} "
          f"{'Relative Error':<15} {'Avg Distance':<15} {'Avg Dist Var':<15} {'Max Distance':<15} "
          f"{'Max Dist Var':<15} {'Chamfer Dist':<15} {'Chamfer Var':<15} {'Hausdorff Dist':<15} "
          f"{'Hausdorff Var':<15}")
    print("-"*240)
    
    total_volume = 0
    weighted_metrics = {
        'distance_error': 0, 'error_variance': 0, 'relative_error': 0,
        'avg_distance': 0, 'max_distance': 0, 'chamfer_distance': 0, 'hausdorff_distance': 0
    }
    
    for scenario, metrics in results.items():
        if scenario != 'Average':
            print(f"{scenario:<20} {metrics['data_volume']:<12d} "
                  f"{metrics['distance_error']:<15.2f} {metrics['error_variance']:<15.2f} "
                  f"{metrics['relative_error']:<15.2f} {metrics['avg_distance']:<15.2f} "
                  f"{metrics['avg_distance_var']:<15.2f} {metrics['max_distance']:<15.2f} "
                  f"{metrics['max_distance_var']:<15.2f} {metrics['chamfer_distance']:<15.2f} "
                  f"{metrics['chamfer_distance_var']:<15.2f} {metrics['hausdorff_distance']:<15.2f} "
                  f"{metrics['hausdorff_distance_var']:<15.2f}")
            
            # 更新加权和
            total_volume += metrics['data_volume']
            for key in weighted_metrics:
                weighted_metrics[key] += metrics[key] * metrics['data_volume']
    
    # 计算加权平均值
    print("-"*240)
    print(f"{'Average':<20} {total_volume:<12d} "
          f"{weighted_metrics['distance_error']/total_volume:<15.2f} "
          f"{weighted_metrics['error_variance']/total_volume:<15.2f} "
          f"{weighted_metrics['relative_error']/total_volume:<15.2f} "
          f"{weighted_metrics['avg_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['avg_distance_var']/total_volume:<15.2f} "
          f"{weighted_metrics['max_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['max_distance_var']/total_volume:<15.2f} "
          f"{weighted_metrics['chamfer_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['chamfer_distance_var']/total_volume:<15.2f} "
          f"{weighted_metrics['hausdorff_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['hausdorff_distance_var']/total_volume:<15.2f}")
    print("="*240)

def evaluate_perception(X_test, y_test, model, X_scaler, y_scaler, feature_type='all'):
    """评估特定感知方式的性能"""
    # 准备输入特征
    if feature_type == 'camera':
        X_test_part = X_test[:, :512]  # 只使用相机特征
        X_test_full = np.zeros((X_test_part.shape[0], 516))
        X_test_full[:, :512] = X_test_part
    elif feature_type == 'imu':
        X_test_part = X_test[:, 512:]  # 只使用IMU特征
        X_test_full = np.zeros((X_test_part.shape[0], 516))
        X_test_full[:, 512:] = X_test_part
    else:
        X_test_full = X_test
    
    # 标准化输入
    X_test_scaled = X_scaler.transform(X_test_full)
    
    # 预测
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)  # 添加.to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()  # 添加.cpu()
    
    # 反标准化预测结果得到相对坐标
    y_pred_relative = y_scaler.inverse_transform(y_pred_scaled)
    
    # 将预测的相对坐标转换回绝对坐标
    y_pred_absolute = np.zeros_like(y_pred_relative)
    y_test_relative = preprocess_coordinates(y_test)  # 获取真实值的相对坐标用于评估
    
    # 使用真实值的参考点（点0）来转换预测值到绝对坐标
    for i in range(len(y_test)):
        # 获取真实值中点0的绝对坐标作为参考点
        ref_point = y_test[i, :3]  # 点0的x,y,z坐标
        
        # 将每个点的相对坐标转换为绝对坐标
        for j in range(9):  # 9个点
            idx = j * 3
            y_pred_absolute[i, idx:idx+3] = y_pred_relative[i, idx:idx+3] + ref_point
    
    # 计算每帧的误差
    frame_errors = np.sqrt(np.sum((y_test_relative - y_pred_relative) ** 2, axis=1))
    
    # 计算统计指标
    distance_error = np.mean(frame_errors)
    error_variance = np.var(frame_errors)  # 添加方差计算
    relative_error = distance_error / np.mean(np.sqrt(np.sum(y_test_relative ** 2, axis=1))) * 100
    
    # 计算点云相关指标
    point_cloud_metrics = calculate_point_cloud_metrics(y_test_relative, y_pred_relative)
    
    return (distance_error, relative_error, y_pred_absolute, error_variance,
            point_cloud_metrics['avg_distance'], point_cloud_metrics['max_distance'],
            point_cloud_metrics['chamfer_distance'], point_cloud_metrics['hausdorff_distance'],
            point_cloud_metrics['avg_distance_var'], point_cloud_metrics['max_distance_var'],
            point_cloud_metrics['chamfer_distance_var'], point_cloud_metrics['hausdorff_distance_var'])

def plot_perception_comparison(test_scenarios, model, X_scaler, y_scaler):
    """绘制不同感知方式的相对误差随时间变化的对比图"""
    plt.figure(figsize=(12, 6))
    
    # 收集所有场景的数据
    all_X = []
    all_y = []
    for paths in test_scenarios.values():
        for path in paths:
            path = Path(path)
            all_X.append(np.load(path / "processed_X.npy"))
            all_y.append(np.load(path / "processed_y.npy"))
    
    X_test = np.concatenate(all_X, axis=0)
    y_test = np.concatenate(all_y, axis=0)
    
    # 只取200-800帧的数据
    start_frame = 200
    end_frame = 800
    X_test = X_test[start_frame:end_frame]
    y_test = y_test[start_frame:end_frame]
    
    # 计算每种感知方式的相对误差
    perception_types = ['Only Camera', 'Only IMU', 'Camera + IMU']
    feature_types = ['camera', 'imu', 'all']
    colors = ['b', 'g', 'r']  # 蓝色表示Camera，绿色表示IMU，红色表示IMU+Camera
    
    for perception, feature_type, color in zip(perception_types, feature_types, colors):
        # 预测并计算相对误差
        if feature_type == 'camera':
            X_test_part = X_test[:, :512]
            X_test_full = np.zeros((X_test_part.shape[0], 516))
            X_test_full[:, :512] = X_test_part
        elif feature_type == 'imu':
            X_test_part = X_test[:, 512:]
            X_test_full = np.zeros((X_test_part.shape[0], 516))
            X_test_full[:, 512:] = X_test_part
        else:
            X_test_full = X_test
            
        X_test_scaled = X_scaler.transform(X_test_full)
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
        
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = preprocess_coordinates(y_test)
        
        # 计算每帧的相对误差
        frame_errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
        relative_errors = frame_errors / np.sqrt(np.sum(y_true ** 2, axis=1)) * 100
        
        # 绘制误差曲线
        frames = np.arange(len(relative_errors))
        plt.plot(frames, relative_errors, color, 
                label=perception, linewidth=2)
    
    plt.xlabel('Frame (200-800)')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error Comparison: Camera vs IMU vs IMU+Camera')
    plt.legend()
    plt.grid(True)
    plt.savefig('perception_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_scenario_trajectories(test_scenarios, model, X_scaler, y_scaler):
    """绘制不同场景的预测轨迹和真实轨迹对比"""
    fig = plt.figure(figsize=(15, 10))
    
    for i, (scenario_name, paths) in enumerate(test_scenarios.items(), 1):
        # 获取该场景的数据
        X_test_list = []
        y_test_list = []
        for path in paths:
            path = Path(path)
            X_test_list.append(np.load(path / "processed_X.npy"))
            y_test_list.append(np.load(path / "processed_y.npy"))
        
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # 预测
        X_test_scaled = X_scaler.transform(X_test)
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
        
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = preprocess_coordinates(y_test)
        
        # 创建子图
        ax = fig.add_subplot(2, 3, i, projection='3d')
        
        # 绘制真实轨迹和预测轨迹
        # 选择中间帧进行可视化
        frame_idx = len(y_true) // 2
        
        # 重塑坐标为(n_points, 3)
        true_points = y_true[frame_idx].reshape(-1, 3)
        pred_points = y_pred[frame_idx].reshape(-1, 3)
        
        # 绘制点和连线
        ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], 
                  c='b', marker='o', label='Ground Truth')
        ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                  c='r', marker='^', label='Prediction')
        
        # 连接点以显示形状
        for points in [true_points, pred_points]:
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    ax.plot([points[i,0], points[j,0]], 
                          [points[i,1], points[j,1]], 
                          [points[i,2], points[j,2]], 
                          'k-', alpha=0.2)
        
        ax.set_title(scenario_name)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('scenario_trajectories.png')
    plt.close()

def plot_scenario_errors(test_scenarios, model, X_scaler, y_scaler):
    """绘制不同场景的相对误差随时间变化的对比图"""
    plt.figure(figsize=(12, 6))
    
    # 为每个场景设置不同的颜色和标记
    colors = ['b', 'g', 'r', 'c', 'm']
    
    for (scenario_name, paths), color in zip(test_scenarios.items(), colors):
        # 获取该场景的数据
        X_test_list = []
        y_test_list = []
        for path in paths:
            path = Path(path)
            X_test_list.append(np.load(path / "processed_X.npy"))
            y_test_list.append(np.load(path / "processed_y.npy"))
        
        if len(X_test_list) == 0:
            continue
            
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # 预测
        X_test_scaled = X_scaler.transform(X_test)
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
        
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = preprocess_coordinates(y_test)
        
        # 计算每帧的相对误差
        frame_errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
        relative_errors = frame_errors / np.sqrt(np.sum(y_true ** 2, axis=1)) * 100
        
        # 绘制误差曲线
        frames = np.arange(len(relative_errors))
        plt.plot(frames, relative_errors, color, label=scenario_name, alpha=0.7)
    
    plt.xlabel('Frame')
    plt.ylabel('Relative Error (%)')
    plt.title('Relative Error Over Time for Different Scenarios')
    plt.legend()
    plt.grid(True)
    plt.savefig('scenario_errors.png')
    plt.close()

def plot_point_position_comparison(y_true, y_pred, save_path):
    """绘制每个点的实际位置和预测位置随时间的变化"""
    # 重塑数据为(n_frames, n_points, 3)
    y_true = y_true.reshape(-1, 9, 3)
    y_pred = y_pred.reshape(-1, 9, 3)
    
    # 创建3x3的子图，每个点一个子图
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Point Position Comparison Over Time', fontsize=16)
    
    # 为每个坐标轴设置不同的颜色
    coords = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    for point_idx in range(9):
        row = point_idx // 3
        col = point_idx % 3
        ax = axes[row, col]
        
        # 绘制每个坐标轴的真实值和预测值
        frames = np.arange(len(y_true))
        for coord_idx, (coord, color) in enumerate(zip(coords, colors)):
            # 真实值
            ax.plot(frames, y_true[:, point_idx, coord_idx], 
                   color=color, linestyle='-', label=f'{coord} True',
                   alpha=0.7)
            # 预测值
            ax.plot(frames, y_pred[:, point_idx, coord_idx], 
                   color=color, linestyle='--', label=f'{coord} Pred',
                   alpha=0.7)
        
        ax.set_title(f'Point {point_idx}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Position (mm)')
        ax.grid(True)
        
        # 只在第一个子图显示图例
        if point_idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_dynamic_scenario_comparison(test_scenarios, model, X_scaler, y_scaler):
    """创建动态的3D场景对比图"""
    
    # 为每个场景设置不同的颜色
    colors = {
        'No obstruction': 'blue',
        'Cubic Rigid Objects': 'green',
        'Irregular Rigid Objects': 'red',
        'Soft Objects': 'purple',
        'Artificial Disturbed': 'orange'
    }
    
    # 收集所有场景的数据
    scenario_data = {}
    for scenario_name, paths in test_scenarios.items():
        X_test_list = []
        y_test_list = []
        for path in paths:
            path = Path(path)
            X_test_list.append(np.load(path / "processed_X.npy"))
            y_test_list.append(np.load(path / "processed_y.npy"))
        
        if len(X_test_list) > 0:
            X_test = np.concatenate(X_test_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)
            
            # 获取预测值
            X_test_scaled = X_scaler.transform(X_test)
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                y_pred_scaled = model(X_test_tensor).cpu().numpy()
            y_pred = y_scaler.inverse_transform(y_pred_scaled)
            
            # 重塑为(n_frames, n_points, 3)
            y_true = preprocess_coordinates(y_test).reshape(-1, 9, 3)
            y_pred = y_pred.reshape(-1, 9, 3)
            
            scenario_data[scenario_name] = (y_true, y_pred)
    
    # 创建动画
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        ax.clear()
        
        for scenario_name, (y_true, y_pred) in scenario_data.items():
            color = colors[scenario_name]
            
            # 绘制真实值
            true_points = y_true[frame]
            ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2],
                      c=color, marker='o', label=f'{scenario_name} (True)',
                      alpha=0.7)
            
            # 绘制预测值
            pred_points = y_pred[frame]
            ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                      c=color, marker='^', label=f'{scenario_name} (Pred)',
                      alpha=0.4)
            
            # 连接点以显示形状
            for points in [true_points, pred_points]:
                for i in range(len(points)):
                    for j in range(i+1, len(points)):
                        ax.plot([points[i,0], points[j,0]],
                              [points[i,1], points[j,1]],
                              [points[i,2], points[j,2]],
                              c=color, alpha=0.2)
        
        ax.set_title(f'Frame {frame}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    # 获取最短序列长度
    min_frames = min(y_true.shape[0] for y_true, _ in scenario_data.values())
    
    anim = animation.FuncAnimation(fig, update, frames=min_frames,
                                 interval=50, blit=False)
    
    # 保存为GIF
    anim.save('scenario_comparison.gif', writer='pillow')
    plt.close()

def plot_scenario_snapshots(test_scenarios, model, X_scaler, y_scaler):
    """创建多时间点的场景对比快照"""
    n_snapshots = 5  # 每个场景显示5个时间点
    
    fig = plt.figure(figsize=(20, 4*len(test_scenarios)))
    
    for row, (scenario_name, paths) in enumerate(test_scenarios.items()):
        X_test_list = []
        y_test_list = []
        for path in paths:
            path = Path(path)
            X_test_list.append(np.load(path / "processed_X.npy"))
            y_test_list.append(np.load(path / "processed_y.npy"))
        
        if len(X_test_list) == 0:
            continue
            
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # 获取预测值
        X_test_scaled = X_scaler.transform(X_test)
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        
        # 重塑为(n_frames, n_points, 3)
        y_true = preprocess_coordinates(y_test).reshape(-1, 9, 3)
        y_pred = y_pred.reshape(-1, 9, 3)
        
        # 选择均匀分布的时间点
        frames = np.linspace(0, len(y_true)-1, n_snapshots, dtype=int)
        
        for col, frame in enumerate(frames):
            ax = fig.add_subplot(len(test_scenarios), n_snapshots, row*n_snapshots + col + 1, projection='3d')
            
            # 绘制真实值和预测值
            true_points = y_true[frame]
            pred_points = y_pred[frame]
            
            ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2],
                      c='b', marker='o', label='True', alpha=0.7)
            ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                      c='r', marker='^', label='Pred', alpha=0.7)
            
            if col == 0:
                ax.set_ylabel(scenario_name)
            if row == 0:
                ax.set_title(f'Frame {frame}')
            
            ax.grid(True)
            if col == n_snapshots-1:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig('scenario_snapshots.png')
    plt.close()

def plot_3d_trajectory_comparison(test_scenarios, model, X_scaler, y_scaler):
    """创建3D轨迹对比图，点的颜色随打印顺序加深"""
    fig = plt.figure(figsize=(20, 12))
    
    # 创建红色和蓝色的渐变色映射
    def create_color_gradient(n_points, color):
        if color == 'red':
            return np.array([(1 - 0.8*i/n_points, 0, 0, 0.75) for i in range(n_points)])
        else:  # blue
            return np.array([(0, 0, 1 - 0.8*i/n_points, 0.75) for i in range(n_points)])
    
    for idx, (scenario_name, paths) in enumerate(test_scenarios.items(), 1):
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        
        # 获取该场景的数据
        X_test_list = []
        y_test_list = []
        for path in paths:
            path = Path(path)
            X_test_list.append(np.load(path / "processed_X.npy"))
            y_test_list.append(np.load(path / "processed_y.npy"))
        
        if len(X_test_list) == 0:
            continue
            
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # 预测
        X_test_scaled = X_scaler.transform(X_test)
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
        
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = preprocess_coordinates(y_test)
        
        # 重塑为(n_frames, n_points, 3)
        y_true = y_true.reshape(-1, 9, 3)
        y_pred = y_pred.reshape(-1, 9, 3)
        
        # 创建颜色渐变
        n_total_points = len(y_true) * 9
        true_colors = create_color_gradient(n_total_points, 'red')
        pred_colors = create_color_gradient(n_total_points, 'blue')
        
        # 绘制所有点
        point_count = 0
        for t in range(len(y_true)):
            for i in range(9):
                # 真实值的点
                ax.scatter(y_true[t, i, 0], 
                         y_true[t, i, 1],
                         y_true[t, i, 2],
                         color=true_colors[point_count],
                         marker='o',
                         s=20)
                
                # 预测值的点
                ax.scatter(y_pred[t, i, 0],
                         y_pred[t, i, 1],
                         y_pred[t, i, 2],
                         color=pred_colors[point_count],
                         marker='^',
                         s=20)
                point_count += 1
        
        ax.set_title(f'{scenario_name}\nRed: True (o), Blue: Predicted (^)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 添加两个颜色条来分别表示真实值和预测值的时间流逝
        # 真实值的颜色条（红色渐变）
        sm_true = plt.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list('', 
            [(1,0,0,0.75), (0.2,0,0,0.75)]))
        sm_true.set_array([])
        cbar_true = plt.colorbar(sm_true, ax=ax, location='left')
        cbar_true.set_label('Earlier → Later (True)')
        
        # 预测值的颜色条（蓝色渐变）
        sm_pred = plt.cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list('', 
            [(0,0,1,0.75), (0,0,0.2,0.75)]))
        sm_pred.set_array([])
        cbar_pred = plt.colorbar(sm_pred, ax=ax)
        cbar_pred.set_label('Earlier → Later (Pred)')
    
    plt.tight_layout()
    plt.savefig('3d_trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_heatmaps(test_scenarios, model, X_scaler, y_scaler):
    """创建预测误差和位置变化的热力图"""
    fig = plt.figure(figsize=(20, 15))
    
    # 创建5x2的子图布局
    gs = plt.GridSpec(5, 2, figure=fig)
    
    for idx, (scenario_name, paths) in enumerate(test_scenarios.items()):
        # 获取数据
        X_test_list = []
        y_test_list = []
        for path in paths:
            path = Path(path)
            X_test_list.append(np.load(path / "processed_X.npy"))
            y_test_list.append(np.load(path / "processed_y.npy"))
        
        if len(X_test_list) == 0:
            continue
            
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        # 预测
        X_test_scaled = X_scaler.transform(X_test)
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
        
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = preprocess_coordinates(y_test)
        
        # 重塑为(n_frames, n_points, 3)
        y_true = y_true.reshape(-1, 9, 3)
        y_pred = y_pred.reshape(-1, 9, 3)
        
        # 计算每个点在每一帧的误差
        point_errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=2))  # (n_frames, n_points)
        
        # 计算每个点在每一帧的位移
        point_displacements = np.sqrt(np.sum(y_true**2, axis=2))  # (n_frames, n_points)
        
        # 创建误差热力图
        ax1 = fig.add_subplot(gs[idx, 0])
        sns.heatmap(point_errors.T, 
                   cmap='YlOrRd',
                   xticklabels=50,  # 每50帧显示一个刻度
                   yticklabels=[f'Point {i}' for i in range(9)],
                   cbar_kws={'label': 'Error (mm)'},
                   ax=ax1)
        ax1.set_title(f'{scenario_name} - Prediction Error')
        ax1.set_xlabel('Frame')
        
        # 创建位移热力图
        ax2 = fig.add_subplot(gs[idx, 1])
        sns.heatmap(point_displacements.T,
                   cmap='viridis',
                   xticklabels=50,
                   yticklabels=[f'Point {i}' for i in range(9)],
                   cbar_kws={'label': 'True vs Pred Position (mm)'},
                   ax=ax2)
        ax2.set_title(f'{scenario_name} - Position Comparison')
        ax2.set_xlabel('Frame')
        
        # 添加文本标注：平均误差和最大误差
        mean_error = np.mean(point_errors)
        max_error = np.max(point_errors)
        ax1.text(-0.2, -0.2, 
                f'Mean Error: {mean_error:.2f}mm\nMax Error: {max_error:.2f}mm',
                transform=ax1.transAxes)
    
    plt.tight_layout()
    plt.savefig('error_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_avg_distance_comparison(test_scenarios, model, X_scaler, y_scaler):
    """绘制不同感知方式的平均距离随时间变化的对比图"""
    plt.figure(figsize=(12, 6))
    
    # 收集所有场景的数据
    all_X = []
    all_y = []
    for paths in test_scenarios.values():
        for path in paths:
            path = Path(path)
            all_X.append(np.load(path / "processed_X.npy"))
            all_y.append(np.load(path / "processed_y.npy"))
    
    X_test = np.concatenate(all_X, axis=0)
    y_test = np.concatenate(all_y, axis=0)
    
    # 只取200-800帧的数据
    start_frame = 200
    end_frame = 800
    X_test = X_test[start_frame:end_frame]
    y_test = y_test[start_frame:end_frame]
    
    # 计算每种感知方式的平均距离
    perception_types = ['Only Camera', 'Only IMU', 'Camera + IMU']
    feature_types = ['camera', 'imu', 'all']
    colors = ['b', 'g', 'r']  # 蓝色表示Camera，绿色表示IMU，红色表示IMU+Camera
    
    for perception, feature_type, color in zip(perception_types, feature_types, colors):
        # 预测并计算平均距离
        if feature_type == 'camera':
            X_test_part = X_test[:, :512]
            X_test_full = np.zeros((X_test_part.shape[0], 516))
            X_test_full[:, :512] = X_test_part
        elif feature_type == 'imu':
            X_test_part = X_test[:, 512:]
            X_test_full = np.zeros((X_test_part.shape[0], 516))
            X_test_full[:, 512:] = X_test_part
        else:
            X_test_full = X_test
            
        X_test_scaled = X_scaler.transform(X_test_full)
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            y_pred_scaled = model(X_test_tensor).cpu().numpy()
        
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = preprocess_coordinates(y_test)
        
        # 重塑为(n_frames, n_points, 3)
        y_true = y_true.reshape(-1, 9, 3)
        y_pred = y_pred.reshape(-1, 9, 3)
        
        # 计算每帧的平均距离
        point_distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=2))  # (n_frames, n_points)
        avg_distances = np.mean(point_distances, axis=1)  # (n_frames,)
        
        # 绘制平均距离曲线
        frames = np.arange(len(avg_distances))
        plt.plot(frames, avg_distances, color, 
                label=perception, linewidth=2)
    
    plt.xlabel('Frame (200-800)')
    plt.ylabel('Average Distance (mm)')
    plt.title('Average Distance Comparison: Camera vs IMU vs IMU+Camera')
    plt.legend()
    plt.grid(True)
    plt.savefig('avg_distance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 加载模型和标准化器
    model, X_scaler, y_scaler = load_model_and_scalers()
    
    # 定义测试场景，每个场景都使用列表格式
    test_scenarios = {
        'No obstruction': [
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-21 03.15.24 PM",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-25 05.37.57 PMVoid",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-25 05.42.18 PMvoid2",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-27 05.33.57 PMvoid",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-27 06.49.54 PMvoid",
            "/home/zfb/Grounded-SAM-2/Take 2025-02-27 06.56.51 PMvoid",
        ],
        'Cubic Rigid Objects': [
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-24 03.18.43 PM movingbox",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-25 11.25.27 AMbox",
            "/home/zfb/Grounded-SAM-2/Take 2025-02-25 11.29.36 AMbox2",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-25 05.45.23 PMboxnew"
        ],
        'Irregular Rigid Objects': [
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-22 11.26.11 AM",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-21 03.39.06 PM",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-22 11.33.27 AM",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-25 11.14.27 AMpineapple",
            "/home/zfb/Grounded-SAM-2/Take 2025-02-25 11.20.22 AMpineaple2"
        ],
        'Soft Objects': [
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-23 02.01.39 PM haimian",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-23 02.13.51 PMmovehaimian",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-24 03.22.59 PMmovingHaimian",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-24 08.30.48 PMHaimian2",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-24 08.36.26 PMHaimian3",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-24 08.41.33 PMmovingHaimian2",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-24 08.44.39 PMmovingHaimian3",
            "/home/zfb/Grounded-SAM-2/Take 2025-02-24 08.47.50 PMmovingHaimian4"
        ],
        'Artificial Disturbed': [
            "/home/zfb/Grounded-SAM-2/Take 2025-02-23 02.34.27 PMmovehardbottle"
        ]
    }
    
    # 评估不同感知方式和场景
    perception_results = {
        'Only Camera': {
            'data_volume': 0, 'distance_error': 0, 'error_variance': 0, 
            'relative_error': 0, 'avg_distance': 0, 'max_distance': 0,
            'chamfer_distance': 0, 'hausdorff_distance': 0,
            'avg_distance_var': 0, 'max_distance_var': 0,
            'chamfer_distance_var': 0, 'hausdorff_distance_var': 0
        },
        'Only IMU': {
            'data_volume': 0, 'distance_error': 0, 'error_variance': 0,
            'relative_error': 0, 'avg_distance': 0, 'max_distance': 0,
            'chamfer_distance': 0, 'hausdorff_distance': 0,
            'avg_distance_var': 0, 'max_distance_var': 0,
            'chamfer_distance_var': 0, 'hausdorff_distance_var': 0
        },
        'Camera + IMU': {
            'data_volume': 0, 'distance_error': 0, 'error_variance': 0,
            'relative_error': 0, 'avg_distance': 0, 'max_distance': 0,
            'chamfer_distance': 0, 'hausdorff_distance': 0,
            'avg_distance_var': 0, 'max_distance_var': 0,
            'chamfer_distance_var': 0, 'hausdorff_distance_var': 0
        }
    }
    
    scenario_results = {}
    total_samples = 0
    
    for name, paths in test_scenarios.items():
        # 合并同一场景的所有数据
        X_test_list = []
        y_test_list = []
        for path in paths:
            path = Path(path)
            X_test_list.append(np.load(path / "processed_X.npy"))
            y_test_list.append(np.load(path / "processed_y.npy"))
            
            # 为每个数据集生成轨迹图
            X_test_single = X_test_list[-1]
            y_test_single = y_test_list[-1]
            
            # 评估单个数据集
            metrics = evaluate_perception(
                X_test_single, y_test_single, model, X_scaler, y_scaler, 'all'
            )
            all_pred_single = metrics[2]  # y_pred_absolute
            
            # 保存轨迹图
            trajectory_path = path / "point_trajectories.png"
            plot_point_trajectories(y_test_single, all_pred_single, trajectory_path)
            
            # 添加点位置对比图
            position_comparison_path = path / "point_positions_comparison.png"
            plot_point_position_comparison(y_test_single, all_pred_single, position_comparison_path)
            print(f"Saved position comparison plot to {position_comparison_path}")
        
        # 合并数据进行整体评估
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        n_samples = len(X_test)
        total_samples += n_samples
        
        # 评估每种感知方式
        camera_metrics = evaluate_perception(X_test, y_test, model, X_scaler, y_scaler, 'camera')
        imu_metrics = evaluate_perception(X_test, y_test, model, X_scaler, y_scaler, 'imu')
        all_metrics = evaluate_perception(X_test, y_test, model, X_scaler, y_scaler, 'all')
        
        # 更新场景结果
        scenario_results[name] = {
            'data_volume': n_samples,
            'distance_error': all_metrics[0],
            'error_variance': all_metrics[3],  # 修正索引
            'relative_error': all_metrics[1],  # 修正索引
            'avg_distance': all_metrics[4],
            'max_distance': all_metrics[5],
            'chamfer_distance': all_metrics[6],
            'hausdorff_distance': all_metrics[7],
            'avg_distance_var': all_metrics[8],
            'max_distance_var': all_metrics[9],
            'chamfer_distance_var': all_metrics[10],
            'hausdorff_distance_var': all_metrics[11]
        }
        
        # 更新感知方式结果
        for perception, metrics in zip(['Only Camera', 'Only IMU', 'Camera + IMU'],
                                     [camera_metrics, imu_metrics, all_metrics]):
            perception_results[perception]['data_volume'] += n_samples
            # 累加需要平均的指标
            for key, idx in [
                ('distance_error', 0),
                ('relative_error', 1),
                ('error_variance', 3),
                ('avg_distance', 4),
                ('chamfer_distance', 6),
                ('avg_distance_var', 8),
                ('chamfer_distance_var', 10)
            ]:
                perception_results[perception][key] += metrics[idx] * n_samples
            
            # 对最大值类指标取最大值
            for key, idx in [
                ('max_distance', 5),
                ('hausdorff_distance', 7),
                ('max_distance_var', 9),
                ('hausdorff_distance_var', 11)
            ]:
                perception_results[perception][key] = max(
                    perception_results[perception][key], metrics[idx]
                )
    
    # 计算平均值
    for perception in perception_results.keys():
        total_samples = perception_results[perception]['data_volume']
        # 对需要平均的指标进行平均
        for key in ['distance_error', 'relative_error', 'error_variance', 
                   'avg_distance', 'chamfer_distance',
                   'avg_distance_var', 'chamfer_distance_var']:  # 添加方差指标到平均计算中
            perception_results[perception][key] /= total_samples
        # max_distance, hausdorff_distance 及其方差已经是最大值，不需要平均
    
    # 打印表格
    print("\n" + "="*80)
    print("TABLE II - PERFORMANCE METRICS FOR DIFFERENT PERCEPTIONS")
    print("="*80)
    print(f"{'Perception':<15} {'Data Volume':<12} {'Distance Error':<15} {'Error Variance':<15} "
          f"{'Relative Error':<15} {'Avg Distance':<15} {'Avg Dist Var':<15} {'Max Distance':<15} "
          f"{'Max Dist Var':<15} {'Chamfer Dist':<15} {'Chamfer Var':<15} {'Hausdorff Dist':<15} "
          f"{'Hausdorff Var':<15}")
    print("-"*80)
    
    for perception, metrics in perception_results.items():
        print(f"{perception:<15} {metrics['data_volume']:<12d} "
              f"{metrics['distance_error']:<15.2f} {metrics['error_variance']:<15.2f} "
              f"{metrics['relative_error']:<15.2f} {metrics['avg_distance']:<15.2f} "
              f"{metrics['avg_distance_var']:<15.2f} {metrics['max_distance']:<15.2f} "
              f"{metrics['max_distance_var']:<15.2f} {metrics['chamfer_distance']:<15.2f} "
              f"{metrics['chamfer_distance_var']:<15.2f} {metrics['hausdorff_distance']:<15.2f} "
              f"{metrics['hausdorff_distance_var']:<15.2f}")
    
    print("\n" + "="*80)
    print("TABLE III - PERFORMANCE METRICS FOR DIFFERENT SCENARIOS")
    print("="*80)
    print(f"{'Scenario':<25} {'Data Volume':<12} {'Distance Error':<15} {'Error Variance':<15} "
          f"{'Relative Error':<15} {'Avg Distance':<15} {'Avg Dist Var':<15} {'Max Distance':<15} "
          f"{'Max Dist Var':<15} {'Chamfer Dist':<15} {'Chamfer Var':<15} {'Hausdorff Dist':<15} "
          f"{'Hausdorff Var':<15}")
    print("-"*80)
    
    total_volume = 0
    weighted_metrics = {
        'distance_error': 0, 'error_variance': 0, 'relative_error': 0,
        'avg_distance': 0, 'max_distance': 0, 'chamfer_distance': 0, 'hausdorff_distance': 0,
        'avg_distance_var': 0, 'max_distance_var': 0,
        'chamfer_distance_var': 0, 'hausdorff_distance_var': 0
    }
    
    for scenario, metrics in scenario_results.items():
        if scenario != 'Average':
            print(f"{scenario:<25} {metrics['data_volume']:<12d} "
                  f"{metrics['distance_error']:<15.2f} {metrics['error_variance']:<15.2f} "
                  f"{metrics['relative_error']:<15.2f} {metrics['avg_distance']:<15.2f} "
                  f"{metrics['avg_distance_var']:<15.2f} {metrics['max_distance']:<15.2f} "
                  f"{metrics['max_distance_var']:<15.2f} {metrics['chamfer_distance']:<15.2f} "
                  f"{metrics['chamfer_distance_var']:<15.2f} {metrics['hausdorff_distance']:<15.2f} "
                  f"{metrics['hausdorff_distance_var']:<15.2f}")
            
            total_volume += metrics['data_volume']
            for key in weighted_metrics:
                weighted_metrics[key] += metrics[key] * metrics['data_volume']
    
    # 计算加权平均值
    print("-"*240)
    print(f"{'Average':<25} {total_volume:<12d} "
          f"{weighted_metrics['distance_error']/total_volume:<15.2f} "
          f"{weighted_metrics['error_variance']/total_volume:<15.2f} "
          f"{weighted_metrics['relative_error']/total_volume:<15.2f} "
          f"{weighted_metrics['avg_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['avg_distance_var']/total_volume:<15.2f} "
          f"{weighted_metrics['max_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['max_distance_var']/total_volume:<15.2f} "
          f"{weighted_metrics['chamfer_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['chamfer_distance_var']/total_volume:<15.2f} "
          f"{weighted_metrics['hausdorff_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['hausdorff_distance_var']/total_volume:<15.2f}")
    print("="*240)

    # 添加新的可视化
    print("\nGenerating perception comparison plot...")
    plot_perception_comparison(test_scenarios, model, X_scaler, y_scaler)
    print("Saved perception comparison plot as 'perception_comparison.png'")

    # 添加新的可视化
    print("\nGenerating 3D trajectory comparison plot...")
    plot_3d_trajectory_comparison(test_scenarios, model, X_scaler, y_scaler)
    print("Saved 3D trajectory comparison plot as '3d_trajectory_comparison.png'")

    # 添加新的可视化
    print("\nGenerating error heatmaps...")
    plot_error_heatmaps(test_scenarios, model, X_scaler, y_scaler)
    print("Saved error heatmaps as 'error_heatmaps.png'")

    # 添加新的可视化
    print("\nGenerating average distance comparison plot...")
    plot_avg_distance_comparison(test_scenarios, model, X_scaler, y_scaler)
    print("Saved average distance comparison plot as 'avg_distance_comparison.png'")

if __name__ == "__main__":
    main() 