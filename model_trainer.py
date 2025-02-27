import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MarkerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AttentionLayer(nn.Module):
    """自注意力层"""
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 计算注意力权重
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(x.shape[-1], dtype=torch.float32))
        attention = F.softmax(scores, dim=-1)
        
        # 应用注意力
        return torch.matmul(attention, V)

class FeatureFusionModule(nn.Module):
    """特征融合模块"""
    def __init__(self, camera_dim, imu_dim):
        super().__init__()
        self.camera_attention = AttentionLayer(camera_dim)
        self.imu_attention = AttentionLayer(imu_dim)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(camera_dim + imu_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, camera_features, imu_features):
        # 应用注意力机制
        camera_att = self.camera_attention(camera_features)
        imu_att = self.imu_attention(imu_features)
        
        # 连接特征
        combined = torch.cat([camera_att, imu_att], dim=-1)
        return self.fusion(combined)

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, input_dim)
        )
        
    def forward(self, x):
        return F.relu(x + self.block(x))

class ImprovedMarkerPredictor(nn.Module):
    def __init__(self, camera_dim=512, imu_dim=4, hidden_dim=256, num_points=9):
        super().__init__()
        
        # 相机特征处理分支 - 使用注意力机制
        self.camera_branch = nn.Sequential(
            nn.Linear(camera_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            AttentionLayer(hidden_dim),  # 只给相机特征添加注意力层
            ResidualBlock(hidden_dim)
        )
        
        # IMU特征处理分支 - 简单的全连接层
        self.imu_branch = nn.Sequential(
            nn.Linear(imu_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResidualBlock(hidden_dim)
        )
        
        # 特征融合 - 简化版本
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 时序处理
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(256, 128),  # 256是因为双向LSTM
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_points * 3)  # 每个点3个坐标
        )
    
    def forward(self, x):
        # 分离相机和IMU特征
        camera_features = x[:, :512]
        imu_features = x[:, 512:]
        
        # 处理相机特征
        camera_features = camera_features.unsqueeze(1)  # 添加时序维度
        camera_processed = self.camera_branch(camera_features)
        
        # 处理IMU特征
        imu_features = imu_features.unsqueeze(1)
        imu_processed = self.imu_branch(imu_features)
        
        # 特征融合 - 直接拼接后通过全连接层
        fused_features = torch.cat([camera_processed, imu_processed], dim=-1)
        fused_features = self.fusion(fused_features)
        
        # LSTM处理时序信息
        lstm_out, _ = self.lstm(fused_features)
        
        # 生成预测
        predictions = self.output(lstm_out).squeeze(1)
        return predictions

# 在所有类定义之后，设置别名
MarkerPredictor = ImprovedMarkerPredictor

def load_multiple_datasets(data_paths):
    """加载多个数据集并合并"""
    all_X = []
    all_y = []
    
    for path in data_paths:
        base_path = Path(path)
        X = np.load(base_path / "processed_X.npy")
        y = np.load(base_path / "processed_y.npy")
        all_X.append(X)
        all_y.append(y)
    
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

def preprocess_coordinates(data):
    """将坐标转换为相对于点0的相对位置"""
    # data shape: (n_samples, 27) - 每9个值为一组xyz坐标
    n_samples = data.shape[0]
    processed_data = np.zeros_like(data)
    
    for i in range(n_samples):
        # 获取点0的坐标作为参考点
        ref_x = data[i, 0]
        ref_y = data[i, 1]
        ref_z = data[i, 2]
        
        # 计算其他点相对于点0的位置
        for j in range(9):  # 9个点
            idx = j * 3
            processed_data[i, idx] = data[i, idx] - ref_x
            processed_data[i, idx + 1] = data[i, idx + 1] - ref_y
            processed_data[i, idx + 2] = data[i, idx + 2] - ref_z
    
    return processed_data

def train_model(train_paths, batch_size=32, num_epochs=200, learning_rate=0.001, val_ratio=0.3):
    """改进的训练函数，添加验证集划分"""
    # 加载数据
    print("Loading training data...")
    X_all, y_all = load_multiple_datasets(train_paths)
    
    # 转换为相对坐标
    y_all = preprocess_coordinates(y_all)
    
    # 随机打乱数据
    indices = np.random.permutation(len(X_all))
    split_idx = int(len(X_all) * (1 - val_ratio))
    
    # 划分训练集和验证集
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    X_train, y_train = X_all[train_indices], y_all[train_indices]
    X_val, y_val = X_all[val_indices], y_all[val_indices]
    
    print(f"\nData split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # 标准化数据
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    
    # 对验证集使用相同的标准化器
    X_val = X_scaler.transform(X_val)
    y_val = y_scaler.transform(y_val)
    
    # 保存标准化器
    joblib.dump(X_scaler, 'X_scaler.joblib')
    joblib.dump(y_scaler, 'y_scaler.joblib')
    
    # 创建数据加载器
    train_dataset = MarkerDataset(X_train, y_train)
    val_dataset = MarkerDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    input_size = X_train.shape[1]
    camera_dim = 512
    imu_dim = input_size - camera_dim
    print(f"Initializing model with input_size={input_size}, camera_dim={camera_dim}, imu_dim={imu_dim}")
    
    model = ImprovedMarkerPredictor(camera_dim=camera_dim, imu_dim=imu_dim)
    model = model.to(device)
    
    # 定义优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_size': input_size,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch
            }, 'marker_predictor.pth')
            print("Saved best model checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    return model

def main():
    # 设置训练数据路径
    train_paths = [
        "/home/zfb/Grounded-SAM-2/Take 2025-02-21 03.15.24 PM",  # 无负载
        "/home/zfb/Grounded-SAM-2/Take 2025-02-25 05.37.57 PMVoid",
        "/home/zfb/Grounded-SAM-2/Take 2025-02-26 03.11.43 PMpic",
    #    "/home/zfb/Grounded-SAM-2/Take 2025-02-25 05.42.18 PMvoid2",
        "/home/zfb/Grounded-SAM-2/Take 2025-02-24 03.18.43 PM movingbox",    # Cubic Rigid Object
        "/home/zfb/Grounded-SAM-2/Take 2025-02-25 05.45.23 PMboxnew",
    #    "/home/zfb/Grounded-SAM-2/Take 2025-02-25 11.29.36 AMbox2",
        "/home/zfb/Grounded-SAM-2/Take 2025-02-25 11.25.27 AMbox",           # Cubic Rigid Object
        "/home/zfb/Grounded-SAM-2/Take 2025-02-22 11.26.11 AM",            # Irregular Rigid Object
        "/home/zfb/Grounded-SAM-2/Take 2025-02-21 03.39.06 PM",
        "/home/zfb/Grounded-SAM-2/Take 2025-02-25 11.14.27 AMpineapple",
    #    "/home/zfb/Grounded-SAM-2/Take 2025-02-25 11.20.22 AMpineaple2",
        "/home/zfb/Grounded-SAM-2/Take 2025-02-22 11.33.27 AM",            # Irregular Rigid Object
    #    "/home/zfb/Grounded-SAM-2/Take 2025-02-23 02.01.39 PM haimian",
        "/home/zfb/Grounded-SAM-2/Take 2025-02-23 02.13.51 PMmovehaimian",
        "/home/zfb/Grounded-SAM-2/Take 2025-02-24 03.22.59 PMmovingHaimian",
        "/home/zfb/Grounded-SAM-2/Take 2025-02-24 08.30.48 PMHaimian2",   # Soft Object
        "/home/zfb/Grounded-SAM-2/Take 2025-02-24 08.36.26 PMHaimian3",
        "/home/zfb/Grounded-SAM-2/Take 2025-02-24 08.41.33 PMmovingHaimian2", # Soft Object
        "/home/zfb/Grounded-SAM-2/Take 2025-02-24 08.44.39 PMmovingHaimian3"
    #    "/home/zfb/Grounded-SAM-2/Take 2025-02-24 08.47.50 PMmovingHaimian4"
    ]
    
    # 训练模型
    model = train_model(train_paths)
    print("Training completed!")

if __name__ == "__main__":
    main()