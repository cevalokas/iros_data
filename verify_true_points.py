"""
验证两个CSV文件中的true_point值是否完全相同的工具。

功能：
- 读取两个CSV文件并比较所有以'true_point'开头的列
- 检查这些列中的值是否完全相同（考虑浮点数误差）
- 如果发现差异，打印出具体的差异信息

使用方法：
1. 直接运行脚本：
   python verify_true_points.py
   
2. 作为模块导入：
   from verify_true_points import verify_true_points
   verify_true_points('original.csv', 'adjusted.csv')

参数：
- original_csv: 原始CSV文件路径
- adjusted_csv: 调整后的CSV文件路径

返回值：
- True: 所有true_point值完全相同
- False: 存在不同的true_point值
"""

import pandas as pd
import numpy as np

def verify_true_points(original_csv, adjusted_csv):
    # Read both CSV files
    df_original = pd.read_csv(original_csv)
    df_adjusted = pd.read_csv(adjusted_csv)
    
    # Get all columns that start with 'true_point'
    true_point_cols = [col for col in df_original.columns if col.startswith('true_point')]
    
    # Compare true points
    for col in true_point_cols:
        if not np.allclose(df_original[col], df_adjusted[col], rtol=1e-15, atol=1e-15):
            print(f"差异发现在列: {col}")
            print("原始值:", df_original[col].values[:5])
            print("调整后:", df_adjusted[col].values[:5])
            return False
    
    print("验证完成：所有true_point值完全相同！")
    return True

if __name__ == "__main__":
    original = "Take 2025-03-06 04.44清洗后预测对比.20 PMsingle2/predictions.csv"
    adjusted = "Take 2025-03-06 04.44清洗后预测对比.20 PMsingle2/predictions_adjusted.csv"
    verify_true_points(original, adjusted)
