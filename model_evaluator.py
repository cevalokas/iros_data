import numpy as np
import torch
import pandas as pd
from pathlib import Path
from model_trainer import MarkerPredictor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import seaborn as sns
import joblib

def load_model_and_scalers():
    # 加载模型
    checkpoint = torch.load('marker_predictor.pth')
    model = MarkerPredictor(checkpoint['input_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载标准化器
    X_scaler = joblib.load('X_scaler.joblib')
    y_scaler = joblib.load('y_scaler.joblib')
    
    return model, X_scaler, y_scaler

def plot_trajectory_comparison(y_test, y_pred, save_dir):
    """绘制每个点的真实轨迹和预测轨迹对比"""
    coords = ['X', 'Y', 'Z']
    
    for point_idx in range(9):
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.suptitle(f'Point {point_idx} Trajectory Comparison')
        
        for i, coord in enumerate(coords):
            true_vals = y_test[:, point_idx*3 + i]
            pred_vals = y_pred[:, point_idx*3 + i]
            
            axes[i].plot(true_vals, label='True', alpha=0.7)
            axes[i].plot(pred_vals, label='Predicted', alpha=0.7)
            axes[i].set_title(f'{coord} Coordinate')
            axes[i].set_xlabel('Frame')
            axes[i].set_ylabel('Position (mm)')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'point_{point_idx}_trajectory.png')
        plt.close()

def visualize_3d_comparison(y_test, y_pred, frame_idx, save_dir):
    """3D可视化某一帧的真实位置和预测位置"""
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取当前帧的真实值和预测值
    true_points = y_test[frame_idx].reshape(-1, 3)
    pred_points = y_pred[frame_idx].reshape(-1, 3)
    
    # 绘制真实点
    ax.scatter(true_points[:, 0], true_points[:, 1], true_points[:, 2], 
              c='blue', marker='o', s=100, label='True Position')
    
    # 绘制预测点
    ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
              c='red', marker='^', s=100, label='Predicted Position')
    
    # 连接对应的真实点和预测点
    for i in range(9):
        ax.plot([true_points[i, 0], pred_points[i, 0]],
                [true_points[i, 1], pred_points[i, 1]],
                [true_points[i, 2], pred_points[i, 2]],
                'g--', alpha=0.3)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Frame {frame_idx} - True vs Predicted Positions')
    ax.legend()
    
    plt.savefig(save_dir / f'3d_comparison_frame_{frame_idx}.png')
    plt.close()

def evaluate_model(test_path):
    # 加载模型和标准化器
    model, X_scaler, y_scaler = load_model_and_scalers()
    
    # 加载测试数据
    X_test = np.load(test_path / "processed_X.npy")
    y_test = np.load(test_path / "processed_y.npy")
    
    # 标准化测试数据
    X_test_scaled = X_scaler.transform(X_test)
    
    # 预测
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_pred_scaled = model(X_test_tensor).numpy()
    
    # 反标准化预测结果
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    # 创建保存结果的目录
    save_dir = Path("evaluation_results")
    save_dir.mkdir(exist_ok=True)
    
    # 绘制轨迹对比图
    plot_trajectory_comparison(y_test, y_pred, save_dir)
    
    # 绘制3D对比图 (选择几个关键帧)
    frame_indices = [0, len(y_test)//4, len(y_test)//2, 3*len(y_test)//4, -1]
    for idx in frame_indices:
        visualize_3d_comparison(y_test, y_pred, idx, save_dir)
    
    # 计算误差统计
    errors = y_test - y_pred
    
    # 创建误差数据框并绘制小提琴图
    error_data = []
    for point_idx in range(9):
        for coord_idx, coord in enumerate(['X', 'Y', 'Z']):
            idx = point_idx * 3 + coord_idx
            error_data.extend([{
                'Point': f'Point {point_idx}',
                'Coordinate': coord,
                'Error': err
            } for err in errors[:, idx]])
    
    error_df = pd.DataFrame(error_data)
    
    # 绘制小提琴图
    plt.figure(figsize=(15, 8))
    sns.violinplot(data=error_df, x='Point', y='Error', hue='Coordinate', split=True)
    plt.title('Prediction Errors Distribution')
    plt.xlabel('Marker Points')
    plt.ylabel('Error (mm)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / 'error_distribution.png')
    plt.close()
    
    # 打印误差统计信息
    print("\nError Statistics (mm):")
    for point_idx in range(9):
        point_errors = errors[:, point_idx*3:(point_idx+1)*3]
        mean_error = np.mean(np.abs(point_errors))
        max_error = np.max(np.abs(point_errors))
        print(f"Point {point_idx}:")
        print(f"  Mean absolute error: {mean_error:.2f}")
        print(f"  Max absolute error: {max_error:.2f}")

def main():
    # 测试数据路径
    test_path = Path("/home/zfb/Grounded-SAM-2/Take 2025-02-21 03.39.06 PM")
    
    # 评估模型
    print("Evaluating model...")
    evaluate_model(test_path)
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 