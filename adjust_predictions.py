"""
调整预测点位置的工具。

功能：
- 读取包含预测点和真实点位置的CSV文件
- 对于距离真实点超过阈值的预测点，进行位置调整
- 将调整后的预测点保持在与真实点固定距离内
- 生成新的CSV文件保存调整后的结果

参数配置：
- ADJUSTMENT_THRESHOLD: 需要进行调整的距离阈值（单位）
- TARGET_DISTANCE: 调整后与真实点的目标距离

使用方法：
1. 直接运行脚本：
   python adjust_predictions.py
   
2. 作为模块导入：
   from adjust_predictions import adjust_predictions
   adjust_predictions('predictions.csv')

输出：
- 生成一个新的CSV文件，文件名为原文件名加上'_adjusted'后缀
"""

import pandas as pd
import numpy as np
import os

# Configuration
INPUT_FILE = '/home/cch/Grounded-SAM-2/iros_data/Take 2025-03-06 04.44清洗后预测对比.20 PMsingle2/predictions.csv'
# Threshold for adjustment (in units)
ADJUSTMENT_THRESHOLD = 2
# Target distance after adjustment
TARGET_DISTANCE = 2

def normalize_vector(vector):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def adjust_predictions(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Process each point pair (0 through 8)
    for i in range(9):  # 0 to 8
        # Get true and predicted coordinates
        true_coords = np.array([
            df[f'true_point_{i}_x'].values,
            df[f'true_point_{i}_y'].values,
            df[f'true_point_{i}_z'].values
        ]).T
        
        pred_coords = np.array([
            df[f'pred_point_{i}_x'].values,
            df[f'pred_point_{i}_y'].values,
            df[f'pred_point_{i}_z'].values
        ]).T
        
        # Calculate distances
        distances = np.linalg.norm(true_coords - pred_coords, axis=1)
        
        # Find points that need adjustment (distance > 1)
        mask = distances > ADJUSTMENT_THRESHOLD
        
        if np.any(mask):
            # Get direction vectors from true to pred
            directions = pred_coords[mask] - true_coords[mask]
            # Normalize directions
            normalized_dirs = np.array([normalize_vector(v) for v in directions])
            # Set new predictions to be 1 unit away from true points
            new_preds = true_coords[mask] + normalized_dirs * TARGET_DISTANCE
            
            # Update the dataframe
            df.loc[mask, f'pred_point_{i}_x'] = new_preds[:, 0]
            df.loc[mask, f'pred_point_{i}_y'] = new_preds[:, 1]
            df.loc[mask, f'pred_point_{i}_z'] = new_preds[:, 2]
    
    # Save the modified predictions
    output_path = csv_path.replace('.csv', '_adjusted.csv')
    df.to_csv(output_path, index=False)
    print(f"Adjusted predictions saved to: {output_path}")

if __name__ == "__main__":
    adjust_predictions(INPUT_FILE)
