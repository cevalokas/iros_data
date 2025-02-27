"""
Data Preprocessor Module for Grounded-SAM-2

This module provides functionality for data preprocessing and manipulation in the Grounded-SAM-2 project.
It includes a DataPreprocessor class that handles various data processing tasks such as:
- Data loading and transformation
- Data visualization
- Data analysis and preprocessing operations

Author: Unknown
Date: 2025-02-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DataPreprocessor:
    def __init__(self):
        pass
        
    def load_mocap_data(self, csv_path):
        """Load and preprocess motion capture data"""
        # Read the metadata rows
        with open(csv_path, 'r') as f:
            header_rows = [next(f) for _ in range(6)]
            
        print("\nHeader information:")
        for i, row in enumerate(header_rows):
            print(f"Row {i}: {row.strip()}")
            
        # 从header中获取实际的帧率
        header_info = header_rows[0].split(',')
        capture_fps = float(header_info[header_info.index('Capture Frame Rate')+1])
        print(f"\nDetected capture frame rate: {capture_fps} fps")
        
        # Read the actual data
        df = pd.read_csv(csv_path, skiprows=6)
        print("\nAvailable columns:", df.columns.tolist())
        
        # 使用实际的帧率
        frame_time = 1.0 / capture_fps
        
        # 创建帧边界
        start_time = df['Time (Seconds)'].min()
        end_time = df['Time (Seconds)'].max()
        frame_boundaries = np.arange(start_time, end_time + frame_time, frame_time)
        
        # 初始化存储结果的字典
        processed_data = {
            'Frame': [],
            'Time': []
        }
        
        # 为每个标记点初始化存储空间
        for i in range(9):  # 9个标记点
            processed_data[f'Marker_{i}_X'] = []
            processed_data[f'Marker_{i}_Y'] = []
            processed_data[f'Marker_{i}_Z'] = []
        
        # 标记点的列映射
        marker_columns = {
            0: ['X.12', 'Y.12', 'Z.12'],  # Unlabeled 2398 -> 0号点
            1: ['X.11', 'Y.11', 'Z.11'],  # Unlabeled 2397 -> 1号点
            2: ['X.9', 'Y.9', 'Z.9'],  # Unlabeled 2395 -> 2号点
            3: ['X.16', 'Y.16', 'Z.16'],  # Unlabeled 2402 -> 3号点
            4: ['X.8', 'Y.8', 'Z.8'],  # Unlabeled 2394 -> 4号点
            5: ['X.14', 'Y.14', 'Z.14'],  # Unlabeled 2400 -> 5号点
            6: ['X.10', 'Y.10', 'Z.10'],     # Unlabeled 2396 -> 6号点
            7: ['X.15', 'Y.15', 'Z.15'],     # Unlabeled 2401 -> 7号点
            8: ['X.13', 'Y.13', 'Z.13']   # Unlabeled 2399 -> 8号点
        }
        
        # 按帧处理数据
        for frame_idx, (start, end) in enumerate(zip(frame_boundaries[:-1], frame_boundaries[1:])):
            # 获取当前帧时间范围内的数据
            frame_data = df[(df['Time (Seconds)'] >= start) & (df['Time (Seconds)'] < end)]
            
            if not frame_data.empty:
                # 添加帧号和时间
                processed_data['Frame'].append(frame_idx)
                processed_data['Time'].append(start)
                
                # 处理每个标记点
                for marker_idx, cols in marker_columns.items():
                    # 计算平均位置
                    x_mean = frame_data[cols[0]].mean()
                    y_mean = frame_data[cols[1]].mean()
                    z_mean = frame_data[cols[2]].mean()
                    
                    # 存储结果
                    processed_data[f'Marker_{marker_idx}_X'].append(x_mean)
                    processed_data[f'Marker_{marker_idx}_Y'].append(y_mean)
                    processed_data[f'Marker_{marker_idx}_Z'].append(z_mean)
        
        # 创建数据框
        processed_df = pd.DataFrame(processed_data)
        
        # 保存处理后的数据
        output_path = Path(csv_path).parent / "processed_mocap.csv"
        processed_df.to_csv(output_path, index=False)
        print(f"\nSaved processed data to {output_path}")
        print(f"Processed {len(processed_df)} frames")
        
        return processed_df
    
    def load_imu_data(self, csv_path):
        """Load IMU data"""
        df = pd.read_csv(csv_path)
        print("\nIMU data columns:", df.columns.tolist())
        
        # 只保留 sensor_id 为 2 的数据
        df = df[df['sensor_id'] == 2].copy()
        
        # 将时间戳转换为从0开始的秒数
        if 'timestamp' in df.columns:
            try:
                # 转换时间戳为datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # 计算相对于第一个时间戳的秒数
                start_time = df['timestamp'].min()
                df['seconds'] = (df['timestamp'] - start_time).dt.total_seconds()
            except:
                print("Warning: Could not parse timestamp as datetime, trying direct conversion")
                try:
                    # 如果已经是秒数格式，确保从0开始
                    start_time = df['timestamp'].min()
                    df['seconds'] = df['timestamp'] - start_time
                except:
                    print("Error: Could not process timestamp column")
                    return None
        
        print(f"\nIMU data time range: {df['seconds'].min():.2f}s to {df['seconds'].max():.2f}s")
        return df
    
    def load_features(self, npy_path):
        """Load extracted features"""
        try:
            return np.load(npy_path)
        except FileNotFoundError:
            print(f"Warning: Features file {npy_path} not found. Using dummy data.")
            return np.zeros((100, 512))  # Assuming feature size of 512
    
    def process_data(self, mocap_path, imu_path, features_path, video_dir, target_fps=5):
        """处理和对齐所有数据"""
        # 加载数据
        mocap_df = self.load_mocap_data(mocap_path)
        imu_df = self.load_imu_data(imu_path)
        features = np.load(features_path)
        
        # 找到所有数据的共同时间范围
        start_time = max(
            mocap_df['Time'].min(),
            imu_df['seconds'].min(),
            0  # 视频从0开始
        )
        end_time = min(
            mocap_df['Time'].max(),
            imu_df['seconds'].max(),
            len(features) / target_fps  # 视频结束时间
        )
        
        print(f"\nTime Range Analysis:")
        print(f"MoCap time range: {mocap_df['Time'].min():.2f}s to {mocap_df['Time'].max():.2f}s")
        print(f"IMU time range: {imu_df['seconds'].min():.2f}s to {imu_df['seconds'].max():.2f}s")
        print(f"Video time range: 0.00s to {len(features)/target_fps:.2f}s")
        print(f"Common time range: {start_time:.2f}s to {end_time:.2f}s")
        
        # 计算每个时间窗口的大小
        window_size = 1.0/target_fps  # 0.2秒
        
        # 生成时间戳，确保5fps
        timestamps = np.arange(start_time, end_time, window_size)
        
        print(f"\nSampling Configuration:")
        print(f"Target FPS: {target_fps}")
        print(f"Window size: {window_size:.3f}s")
        print(f"Number of windows: {len(timestamps)}")
        
        processed_data = []
        
        for i, center_time in enumerate(timestamps):
            # 定义时间窗口
            window_start = center_time - window_size/2
            window_end = center_time + window_size/2
            
            # 获取这个窗口内的所有数据
            mocap_frame = mocap_df[(mocap_df['Time'] >= window_start) & 
                                  (mocap_df['Time'] < window_end)]
            
            imu_frame = imu_df[(imu_df['seconds'] >= window_start) & 
                              (imu_df['seconds'] < window_end)]
            
            # 获取对应的视频特征
            feature_idx = int((center_time - start_time) * target_fps)
            if feature_idx >= len(features):
                break
            
            frame_data = {
                'frame': i,
                'timestamp': center_time,
                'window_start': window_start,
                'window_end': window_end
            }
            
            # 处理MoCap数据
            if not mocap_frame.empty:
                for marker in range(9):
                    frame_data.update({
                        f'target_marker_{marker}_x': mocap_frame[f'Marker_{marker}_X'].mean(),
                        f'target_marker_{marker}_y': mocap_frame[f'Marker_{marker}_Y'].mean(),
                        f'target_marker_{marker}_z': mocap_frame[f'Marker_{marker}_Z'].mean()
                    })
            else:
                continue
            
            # 处理IMU数据
            if not imu_frame.empty:
                imu_means = imu_frame.mean(numeric_only=True)
                frame_data.update({
                    'imu_acc_x': imu_means['x'],
                    'imu_acc_y': imu_means['y'],
                    'imu_acc_z': imu_means['z'],
                    'imu_gyro_x': imu_means['w']
                })
            else:
                continue
            
            # 添加视频特征
            frame_data.update({
                f'video_feature_{i}': val 
                for i, val in enumerate(features[feature_idx].flatten())
            })
            
            processed_data.append(frame_data)
        
        # 转换为DataFrame并保存
        result_df = pd.DataFrame(processed_data)
        
        # 保存处理后的数据
        output_csv = Path(mocap_path).parent / "processed_data.csv"
        result_df.to_csv(output_csv, index=False)
        print(f"Saved processed data to {output_csv}")
        
        # 分离特征和目标
        X_columns = ([col for col in result_df.columns if col.startswith('imu_')] + 
                     [col for col in result_df.columns if col.startswith('video_feature_')])
        y_columns = [col for col in result_df.columns if col.startswith('target_marker_')]
        
        X = result_df[X_columns].values
        y = result_df[y_columns].values
        
        # 保存为numpy数组
        base_path = Path(mocap_path).parent
        np.save(base_path / "processed_X.npy", X)
        np.save(base_path / "processed_y.npy", y)
        
        # 打印最终统计信息
        print("\nFinal Statistics:")
        print(f"Total frames: {len(result_df)}")
        print(f"Actual FPS: {len(result_df)/(timestamps[-1] - timestamps[0]):.1f}")
        print(f"Time range: {result_df['timestamp'].min():.2f}s to {result_df['timestamp'].max():.2f}s")
        print("\nFeature dimensions:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        return X, y, result_df
    
    def visualize_markers(self, processed_df, frame_idx=None, save_path=None):
        """
        Visualize marker positions in 3D space from three different angles
        """
        if frame_idx is None:
            frame_idx = len(processed_df) // 2  # 使用中间帧
        
        # 获取所有标记点列
        markers = {}
        for i in range(9):  # 9个标记点
            markers[f'Marker_{i}'] = [
                f'Marker_{i}_X',
                f'Marker_{i}_Y',
                f'Marker_{i}_Z'
            ]
        
        # 计算显示范围
        all_coords = []
        for marker_name, cols in markers.items():
            for col in cols:
                all_coords.extend(processed_df[col].values)
        
        max_range = max(abs(min(all_coords)), abs(max(all_coords)))
        max_range *= 1.2  # 添加20%的边距
        
        # 创建图形
        fig = plt.figure(figsize=(20, 8))
        
        # 定义三个视角
        views = [
            (30, 45, "Perspective View"),
            (0, 0, "Front View (XY)"),
            (0, 90, "Side View (YZ)")
        ]
        
        # 颜色映射
        colors = plt.cm.rainbow(np.linspace(0, 1, len(markers)))
        
        # 创建三个子图，不同视角
        for i, (elev, azim, title) in enumerate(views, 1):
            ax = fig.add_subplot(1, 3, i, projection='3d')
            
            # 绘制每个标记点
            for (marker_name, cols), color in zip(markers.items(), colors):
                x = processed_df[cols[0]].iloc[frame_idx]
                y = processed_df[cols[1]].iloc[frame_idx]
                z = processed_df[cols[2]].iloc[frame_idx]
                ax.scatter(x, y, z, c=[color], s=100, marker='o', label=marker_name)
            
            # 设置标签和标题
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_zlabel('Z (mm)')
            ax.set_title(f'{title}\n(Frame {frame_idx})')
            
            # 设置一致的坐标轴范围
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
            
            # 添加网格
            ax.grid(True)
            
            # 设置视角
            ax.view_init(elev=elev, azim=azim)
            
            # 调整相机距离
            ax.dist = 11
            
            # 只在第一个子图添加图例
            if i == 1:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # 调整布局
        plt.tight_layout(w_pad=4, rect=[0, 0, 0.98, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_animation(self, processed_df, output_path, frames=None, interval=50):
        """
        Create an animation of marker movements with two views
        """
        from matplotlib.animation import FuncAnimation
        
        if frames is None:
            frames = range(0, len(processed_df), 10)  # 每10帧取一帧
        
        # 创建图形
        fig = plt.figure(figsize=(20, 10))
        
        # 获取标记点
        markers = {}
        for i in range(9):  # 9个标记点
            markers[f'Marker_{i}'] = [
                f'Marker_{i}_X',
                f'Marker_{i}_Y',
                f'Marker_{i}_Z'
            ]
        
        # 颜色映射
        colors = plt.cm.rainbow(np.linspace(0, 1, len(markers)))
        
        # 定义两个视角
        views = [
            (0, 0, "Front View (XY)"),
            (0, 90, "Side View (YZ)")
        ]
        
        # 计算坐标轴范围
        all_coords = []
        for marker_name, cols in markers.items():
            for col in cols:
                all_coords.extend(processed_df[col].values)
        
        max_range = max(abs(min(all_coords)), abs(max(all_coords)))
        max_range *= 1.2  # 添加20%的边距
        
        # 创建两个子图
        axes = [fig.add_subplot(1, 2, i+1, projection='3d') for i in range(2)]
        
        def update(frame):
            for ax, (elev, azim, title) in zip(axes, views):
                ax.clear()
                
                # 绘制每个标记点
                for (marker_name, cols), color in zip(markers.items(), colors):
                    x = processed_df[cols[0]].iloc[frame]
                    y = processed_df[cols[1]].iloc[frame]
                    z = processed_df[cols[2]].iloc[frame]
                    ax.scatter(x, y, z, c=[color], s=100, marker='o', label=marker_name)
                
                # 设置标签和标题
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.set_title(f'{title}\n(Frame {frame})')
                
                # 设置一致的坐标轴范围
                ax.set_xlim(-max_range, max_range)
                ax.set_ylim(-max_range, max_range)
                ax.set_zlim(-max_range, max_range)
                
                # 添加网格
                ax.grid(True)
                
                # 设置视角
                ax.view_init(elev=elev, azim=azim)
                
                # 调整相机距离
                ax.dist = 11
                
                # 只在第一个子图添加图例
                if ax == axes[0]:
                    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # 创建动画
        anim = FuncAnimation(fig, update, frames=frames, interval=interval)
        
        # 调整布局
        plt.tight_layout(w_pad=4, rect=[0, 0, 0.98, 1])
        
        # 保存动画
        anim.save(output_path, writer='pillow', dpi=100)
        plt.close()

def main():
    preprocessor = DataPreprocessor()
    
    # 更新数据路径
    base_path = Path("/home/cch/Grounded-SAM-2/iros_data/Take 2025-02-28 12.35.38 PMnew")
    mocap_path = base_path / "Take 2025-02-28 12.35.38 PM.csv"
    imu_path = base_path / "imu_data_1511.csv"
    features_path = base_path / "features/feature2.npy"
    video_dir = base_path / "frames"
    
    # First process the mocap data and create visualizations
    print("Processing mocap data...")
    mocap_df = preprocessor.load_mocap_data(mocap_path)
    
    # Print the actual data to debug
    print("\nFirst few rows of processed data:")
    print(mocap_df.head())
    print("\nColumns in processed data:")
    print(mocap_df.columns.tolist())
    
    # Check if we have any data in the relative position columns
    # rel_cols = [col for col in mocap_df.columns if '_rel_' in col]
    # print("\nChecking relative position data:")
    # for col in rel_cols:
    #     data = mocap_df[col]
    #     print(f"{col}: Range [{data.min():.2f}, {data.max():.2f}], Mean: {data.mean():.2f}")
    
    # # Create visualization for a single frame
    # vis_path = Path(mocap_path).parent / "marker_positions.png"
    # preprocessor.visualize_markers(mocap_df, frame_idx=1000, save_path=vis_path)
    # print(f"Saved static visualization to {vis_path}")
    
    # # Create animation
    # anim_path = Path(mocap_path).parent / "marker_animation.gif"
    # preprocessor.create_animation(mocap_df, anim_path, interval=50)
    # print(f"Saved animation to {anim_path}")
    
    # # Process the rest of the data
    # print("\nProcessing IMU and feature data...")
    # X, y, result_df = preprocessor.process_data(mocap_path, imu_path, features_path, video_dir)
    
    # print("\nData saved to:")
    # print(f"- {base_path / 'processed_data.csv'}")
    # print(f"- {base_path / 'processed_X.npy'}")
    # print(f"- {base_path / 'processed_y.npy'}")
    # print("Done!")

if __name__ == "__main__":
    main() 