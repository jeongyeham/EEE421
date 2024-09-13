import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

# 确保模型和权重保存的目录存在
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

csv_file = './Training data.csv'
model_path = os.path.join(model_dir, 'fully_connected_nn.pth')


# 重新定义一个全连接神经网络模型
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FullyConnectedNN, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, output_size)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x


# 重新定义一个数据读取类
class BuildingEnergyDataset(Dataset):
    def __init__(self, csv_data_file):
        self.data_frame = pd.read_csv(csv_data_file).drop(columns=['Index'])
        self.data_frame.fillna(self.data_frame.mean(), inplace=True)  # 填充缺失值

        self.target = self.data_frame['ENERGY_CONSUMPTION_CURRENT']
        self.features = self.data_frame.drop(columns=['ENERGY_CONSUMPTION_CURRENT'])

        self.data_scaler = StandardScaler()
        self.features = self.data_scaler.fit_transform(self.features)

        #self.target = self.target.to_frame()
        #self.target = self.data_scaler.transform(self.target)
        #self.target = pd.Series(self.target.squeeze())

        self.target = self.data_scaler.fit_transform(self.target.values.reshape(-1, 1)).flatten()
        print(self.target)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        features = self.features[idx]
        target = self.target[idx]

        sample = {'features': torch.tensor(features, dtype=torch.float),
                  'target': torch.tensor(target, dtype=torch.float)}

        return sample


def spilit_dataset(dataset, train_ratio, val_ratio):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    return random_split(dataset, [train_size, val_size])


def load_model(model_path):
    if os.path.exists(model_path):
        print("找到已保存的模型，正在加载...")
        model = FullyConnectedNN(input_size=26, output_size=1)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print("模型加载成功！")
        return model
    else:
        print("未找到模型文件，将训练新模型。")
        return None


def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, error_threshold):
    model.train()
    print("Model is training!")

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        total = 0

        for data in train_loader:
            optimizer.zero_grad()
            output = model(data['features'])
            loss = criterion(output, data['target'].view(-1, 1))
            train_loss += loss.item() * data['features'].size(0)
            total += data['features'].size(0)

            loss.backward()
            optimizer.step()

        # scheduler.step()
        avg_train_loss = train_loss / total
        elapsed_time = time.time() - start_time
        print(
            f'Epoch {epoch + 1}, 学习率: {scheduler.get_last_lr()}, 训练损失: {avg_train_loss:.4f}, 耗时: {elapsed_time:.2f}s')

    return model


def verificate_model(model, val_loader, criterion):
    # 验证阶段
    model.eval()
    total_loss = 0
    total_rmse = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            output = model(data['features'])
            loss = criterion(output, data['target'].view(-1, 1))
            total_loss += loss.item() * data['features'].size(0)
            total_rmse += torch.sqrt(loss).sum().item()
            total += data['features'].size(0)

    avg_val_loss = total_loss / total
    avg_val_rmse = total_rmse / total
    print(f'平均验证损失: {avg_val_loss:.4f}, 平均验证RMSE: {avg_val_rmse:.4f}')


if __name__ == "__main__":
    dataset = BuildingEnergyDataset(csv_file)
    train_dataset, val_dataset = spilit_dataset(dataset, train_ratio=0.8, val_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    criterion = nn.MSELoss()

    model = load_model(model_path)
    if model is None:
        model = FullyConnectedNN(input_size=26, output_size=1)
        criterion = nn.MSELoss()

        params = [
            {'params': model.fc1.parameters(), 'lr': 0.02},
            {'params': model.fc2.parameters(), 'lr': 0.02},
            {'params': model.fc3.parameters(), 'lr': 0.02},
            {'params': model.fc4.parameters(), 'lr': 0.02}
        ]

        optimizer = optim.Adam(params)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

        model = train_model(model, train_loader, criterion, optimizer, scheduler, 500, 10)
        verificate_model(model, val_loader, criterion)
        torch.save(model.state_dict(), model_path)
    else:
        continue_training = input("是否继续训练模型？(yes/no): ")
        if continue_training.lower() == 'yes':
            params = [
                {'params': model.fc1.parameters(), 'lr': 0.2},
                {'params': model.fc2.parameters(), 'lr': 0.1},
                {'params': model.fc3.parameters(), 'lr': 0.2},
                {'params': model.fc4.parameters(), 'lr': 0.50}
            ]

            optimizer = optim.Adam(params)
            scheduler = StepLR(optimizer, step_size=30, gamma=0.001)

            num_epochs = int(input("输入要训练的epoch数: "))
            model = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, 10)
            torch.save(model.state_dict(), model_path)
        else:
            print("模型训练完成！")
            verificate_model(model, val_loader, criterion)
