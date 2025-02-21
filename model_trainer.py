import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

class MarkerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MarkerPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 27)  # 9个点 × 3个坐标
        )
    
    def forward(self, x):
        return self.network(x)

def load_and_preprocess_data(data_paths):
    """加载并预处理多个数据集"""
    X_list = []
    y_list = []
    
    for path in data_paths:
        X = np.load(path / "processed_X.npy")
        y = np.load(path / "processed_y.npy")
        X_list.append(X)
        y_list.append(y)
    
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    return X, y

def train_model(train_paths, batch_size=32, epochs=100, lr=0.001):
    # 加载训练数据
    X_train, y_train = load_and_preprocess_data(train_paths)
    
    # 标准化数据
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    
    # 保存标准化器
    joblib.dump(X_scaler, 'X_scaler.joblib')
    joblib.dump(y_scaler, 'y_scaler.joblib')
    
    # 创建数据集和数据加载器
    train_dataset = MarkerDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    input_size = X_train.shape[1]
    model = MarkerPredictor(input_size)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
    }, 'marker_predictor.pth')
    
    return model, X_scaler, y_scaler

def main():
    # 训练数据路径
    train_paths = [
        Path("/home/zfb/Grounded-SAM-2/Take 2025-02-21 03.15.24 PM"),
        Path("/home/zfb/Grounded-SAM-2/Take 2025-02-21 03.30.57 PM")
    ]
    
    # 训练模型
    print("Training model...")
    model, X_scaler, y_scaler = train_model(train_paths)
    print("Training completed!")

if __name__ == "__main__":
    main()