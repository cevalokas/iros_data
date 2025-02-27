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
    
    # 1. 平均欧几里德距离 (mm)
    point_distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=2))  # (n_frames, n_points)
    avg_distance = np.mean(point_distances)
    
    # 2. 最大欧几里德距离 (mm)
    max_distance = np.max(point_distances)
    
    # 3. Chamfer距离
    def compute_chamfer(pc1, pc2):
        # 计算每个点到另一个点云中最近点的距离
        distances1 = np.sqrt(np.min([np.sum((p1 - pc2) ** 2, axis=1) for p1 in pc1], axis=1))
        distances2 = np.sqrt(np.min([np.sum((p2 - pc1) ** 2, axis=1) for p2 in pc1], axis=1))
        return np.mean(distances1) + np.mean(distances2)
    
    chamfer_distances = [compute_chamfer(true_frame, pred_frame) 
                        for true_frame, pred_frame in zip(y_true, y_pred)]
    chamfer_distance = np.mean(chamfer_distances)
    
    # 4. Hausdorff距离
    def compute_hausdorff(pc1, pc2):
        # 计算从pc1到pc2的最大最小距离
        forward = np.max([np.min([np.sqrt(np.sum((p1 - pc2) ** 2, axis=1)) for p1 in pc1], axis=1)])
        backward = np.max([np.min([np.sqrt(np.sum((p2 - pc1) ** 2, axis=1)) for p2 in pc1], axis=1)])
        return max(forward, backward)
    
    hausdorff_distances = [compute_hausdorff(true_frame, pred_frame) 
                          for true_frame, pred_frame in zip(y_true, y_pred)]
    hausdorff_distance = np.mean(hausdorff_distances)
    
    return {
        'avg_distance': avg_distance,
        'max_distance': max_distance,
        'chamfer_distance': chamfer_distance,
        'hausdorff_distance': hausdorff_distance
    }

def print_perception_table(results):
    """打印不同感知方式的性能表格"""
    print("\nTABLE II - PERFORMANCE METRICS FOR DIFFERENT PERCEPTIONS")
    print("="*160)
    print(f"{'Perception':<15} {'Data Volume':<12} {'Distance Error':<15} {'Error Variance':<15} "
          f"{'Relative Error':<15} {'Avg Distance':<15} {'Max Distance':<15} "
          f"{'Chamfer Dist':<15} {'Hausdorff Dist':<15}")
    print("-"*160)
    
    for perception, metrics in results.items():
        print(f"{perception:<15} {metrics['data_volume']:<12d} "
              f"{metrics['distance_error']:<15.2f} {metrics['error_variance']:<15.2f} "
              f"{metrics['relative_error']:<15.2f} {metrics['avg_distance']:<15.2f} "
              f"{metrics['max_distance']:<15.2f} {metrics['chamfer_distance']:<15.2f} "
              f"{metrics['hausdorff_distance']:<15.2f}")
    print("-"*160)

def print_scenario_table(results):
    """打印不同场景的性能表格"""
    print("\nTABLE III - PERFORMANCE METRICS FOR DIFFERENT SCENARIOS")
    print("="*160)
    print(f"{'Scenario':<20} {'Data Volume':<12} {'Distance Error':<15} {'Error Variance':<15} "
          f"{'Relative Error':<15} {'Avg Distance':<15} {'Max Distance':<15} "
          f"{'Chamfer Dist':<15} {'Hausdorff Dist':<15}")
    print("-"*160)
    
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
                  f"{metrics['max_distance']:<15.2f} {metrics['chamfer_distance']:<15.2f} "
                  f"{metrics['hausdorff_distance']:<15.2f}")
            
            # 更新加权和
            total_volume += metrics['data_volume']
            for key in weighted_metrics:
                weighted_metrics[key] += metrics[key] * metrics['data_volume']
    
    # 计算加权平均值
    print("-"*160)
    print(f"{'Average':<20} {total_volume:<12d} "
          f"{weighted_metrics['distance_error']/total_volume:<15.2f} "
          f"{weighted_metrics['error_variance']/total_volume:<15.2f} "
          f"{weighted_metrics['relative_error']/total_volume:<15.2f} "
          f"{weighted_metrics['avg_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['max_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['chamfer_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['hausdorff_distance']/total_volume:<15.2f}")
    print("="*160)

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
            point_cloud_metrics['chamfer_distance'], point_cloud_metrics['hausdorff_distance'])

def main():
    # 加载模型和标准化器
    model, X_scaler, y_scaler = load_model_and_scalers()
    
    # 定义测试场景，每个场景都使用列表格式
    test_scenarios = {
        'No obstruction': [
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-21 03.15.24 PM",
        #    "/home/zfb/Grounded-SAM-2/Take 2025-02-25 05.37.57 PMVoid",
            "/home/zfb/Grounded-SAM-2/Take 2025-02-25 05.42.18 PMvoid2"
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
        'Only Camera': {'data_volume': 0, 'distance_error': 0, 'error_variance': 0, 
                       'relative_error': 0, 'avg_distance': 0, 'max_distance': 0,
                       'chamfer_distance': 0, 'hausdorff_distance': 0},
        'Only IMU': {'data_volume': 0, 'distance_error': 0, 'error_variance': 0,
                    'relative_error': 0, 'avg_distance': 0, 'max_distance': 0,
                    'chamfer_distance': 0, 'hausdorff_distance': 0},
        'Camera + IMU': {'data_volume': 0, 'distance_error': 0, 'error_variance': 0,
                        'relative_error': 0, 'avg_distance': 0, 'max_distance': 0,
                        'chamfer_distance': 0, 'hausdorff_distance': 0}
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
            
            # 评估单个数据集 - 修复这里的解包
            metrics = evaluate_perception(
                X_test_single, y_test_single, model, X_scaler, y_scaler, 'all'
            )
            # 只使用需要的值：distance_error, relative_error, y_pred_absolute, error_variance
            all_pred_single = metrics[2]  # y_pred_absolute 在第3个位置
            
            # 保存轨迹图
            trajectory_path = path / "point_trajectories.png"
            plot_point_trajectories(y_test_single, all_pred_single, trajectory_path)
            print(f"\nSaved trajectory comparison plot to {trajectory_path}")
        
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
            'hausdorff_distance': all_metrics[7]
        }
        
        # 更新感知方式结果
        for perception, metrics in zip(['Only Camera', 'Only IMU', 'Camera + IMU'],
                                     [camera_metrics, imu_metrics, all_metrics]):
            perception_results[perception]['data_volume'] += n_samples
            perception_results[perception]['distance_error'] += metrics[0] * n_samples
            perception_results[perception]['relative_error'] += metrics[1] * n_samples
            perception_results[perception]['error_variance'] += metrics[3] * n_samples
            perception_results[perception]['avg_distance'] += metrics[4] * n_samples
            perception_results[perception]['max_distance'] = max(
                perception_results[perception]['max_distance'], metrics[5]
            )  # 对最大距离取最大值而不是加权平均
            perception_results[perception]['chamfer_distance'] += metrics[6] * n_samples
            perception_results[perception]['hausdorff_distance'] = max(
                perception_results[perception]['hausdorff_distance'], metrics[7]
            )  # 对Hausdorff距离取最大值而不是加权平均
    
    # 计算平均值
    for perception in perception_results.keys():
        total_samples = perception_results[perception]['data_volume']
        # 对需要平均的指标进行平均
        for key in ['distance_error', 'relative_error', 'error_variance', 
                   'avg_distance', 'chamfer_distance']:
            perception_results[perception][key] /= total_samples
        # max_distance 和 hausdorff_distance 已经是最大值，不需要平均
    
    # 打印表格
    print("\n" + "="*80)
    print("TABLE II - PERFORMANCE METRICS FOR DIFFERENT PERCEPTIONS")
    print("="*80)
    print(f"{'Perception':<15} {'Data Volume':<12} {'Distance Error':<15} {'Error Variance':<15} "
          f"{'Relative Error':<15} {'Avg Distance':<15} {'Max Distance':<15} "
          f"{'Chamfer Dist':<15} {'Hausdorff Dist':<15}")
    print("-"*80)
    
    for perception, metrics in perception_results.items():
        print(f"{perception:<15} {metrics['data_volume']:<12d} "
              f"{metrics['distance_error']:<15.2f} {metrics['error_variance']:<15.2f} "
              f"{metrics['relative_error']:<15.2f} {metrics['avg_distance']:<15.2f} "
              f"{metrics['max_distance']:<15.2f} {metrics['chamfer_distance']:<15.2f} "
              f"{metrics['hausdorff_distance']:<15.2f}")
    
    print("\n" + "="*80)
    print("TABLE III - PERFORMANCE METRICS FOR DIFFERENT SCENARIOS")
    print("="*80)
    print(f"{'Scenario':<25} {'Data Volume':<12} {'Distance Error':<15} {'Error Variance':<15} "
          f"{'Relative Error':<15} {'Avg Distance':<15} {'Max Distance':<15} "
          f"{'Chamfer Dist':<15} {'Hausdorff Dist':<15}")
    print("-"*80)
    
    total_volume = 0
    weighted_metrics = {
        'distance_error': 0, 'error_variance': 0, 'relative_error': 0,
        'avg_distance': 0, 'max_distance': 0, 'chamfer_distance': 0, 'hausdorff_distance': 0
    }
    
    for scenario, metrics in scenario_results.items():
        print(f"{scenario:<25} {metrics['data_volume']:<12d} "
              f"{metrics['distance_error']:<15.2f} {metrics['error_variance']:<15.2f} "
              f"{metrics['relative_error']:<15.2f} {metrics['avg_distance']:<15.2f} "
              f"{metrics['max_distance']:<15.2f} {metrics['chamfer_distance']:<15.2f} "
              f"{metrics['hausdorff_distance']:<15.2f}")
        
        total_volume += metrics['data_volume']
        for key in weighted_metrics:
            weighted_metrics[key] += metrics[key] * metrics['data_volume']
    
    # 计算加权平均值
    print("-"*80)
    print(f"{'Average':<25} {total_volume:<12d} "
          f"{weighted_metrics['distance_error']/total_volume:<15.2f} "
          f"{weighted_metrics['error_variance']/total_volume:<15.2f} "
          f"{weighted_metrics['relative_error']/total_volume:<15.2f} "
          f"{weighted_metrics['avg_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['max_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['chamfer_distance']/total_volume:<15.2f} "
          f"{weighted_metrics['hausdorff_distance']/total_volume:<15.2f}")
    print("="*80)

if __name__ == "__main__":
    main() 