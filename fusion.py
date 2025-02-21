import pandas as pd

FILE = "./Labeled_Take 2025-02-20 12.47.03 PM/test.csv"
""
row1 = 0  # 需要合并的第一行的索引
row2 = 1  # 需要合并的第二行的索引

def merge_rows_in_csv(input_file, row1, row2):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 检查行索引是否有效
    # if row1 < 0 or row2 < 0 or row1 >= len(df) or row2 >= len(df):
    #     raise ValueError("行索引无效")
    
    # 获取第row1行和第row2行
    row1_data = df.iloc[row1]
    row2_data = df.iloc[row2]
    
    # 合并两行
    merged_row = []
    for col in df.columns:
        val1 = str(row1_data[col])
        val2 = str(row2_data[col])
        merged_row.append(f"{val1}_{val2}")
    
    # 用合并后的行替换原数据中的指定行
    df.iloc[row1] = merged_row
    df.drop(row2, inplace=True)  # 删除第二行
    
    # 将修改后的DataFrame保存回原CSV文件
    df.to_csv(input_file, index=False)



merge_rows_in_csv(FILE, row1, row2)
