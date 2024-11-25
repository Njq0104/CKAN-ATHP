# -*- coding: utf-8 -*-
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.nn import functional as F
import os
from kan_convolutional.KANConv import KAN_Convolutional_Layer

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# 定义模型结构
class KANC_MLP(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        # 卷积层
        self.conv2 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            device=device
        )

        # 池化层
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(2, 2))
        self.flat = nn.Flatten()

        # 使用 LazyLinear 自动推断输入尺寸
        self.linear1 = nn.LazyLinear(256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, x1, x2, x3):
        # 拼接
        x_combined = torch.cat((x1, x2, x3), dim=3)

        # 卷积和池化
        x = self.conv2(x_combined)
        x = self.pool2(x)

        # 展平后通过全连接层
        x = self.flat(x)

        x = self.linear1(x)

        x = self.linear2(x)

        # 应用 softmax
        x = F.log_softmax(x, dim=1)
        return x


# 加载保存的模型
model = KANC_MLP(device=device).to(device)
model.load_state_dict(torch.load('CKAN_1_stratify_87_79.pth', map_location=device))
model.eval()  # 切换到评估模式

# 加载新的特征矩阵
X1_new = np.load("DistilProtBert_test-0.5.npy")
X2_new = np.load("PubChem10M_test.npy")
X3_new = np.load("t5_test.npy")

# 加载新的序列和标签（如果有标签文件）
sequences_new_df = pd.read_csv('filtered_file.csv')
sequences_new = sequences_new_df.iloc[:, 0].values
labels_new = torch.tensor(sequences_new_df.iloc[:, 1].values, dtype=torch.long).to(device)  # 假设有真实标签

# 将特征矩阵转换为PyTorch张量并调整形状
X1_new = torch.tensor(X1_new, dtype=torch.float32).unsqueeze(1).unsqueeze(2).to(device)
X2_new = torch.tensor(X2_new, dtype=torch.float32).unsqueeze(1).unsqueeze(2).to(device)
X3_new = torch.tensor(X3_new, dtype=torch.float32).unsqueeze(1).unsqueeze(2).to(device)

# 创建新的DataLoader
new_dataset = TensorDataset(X1_new, X2_new, X3_new, labels_new)  # 如果没有标签，可以移除 labels_new
new_loader = DataLoader(new_dataset, batch_size=64, shuffle=False)

# 用于存储预测结果
all_new_predictions = []
all_new_labels = []
all_new_sequences = []

# 跟踪当前批次的起始索引
current_index = 0
total_samples = len(new_dataset)

# 进行预测
with torch.no_grad():
    for batch in new_loader:
        x1, x2, x3, y = batch  # 如果没有标签，移除 y
        x1, x2, x3 = x1.to(device), x2.to(device), x3.to(device)

        # 模型预测
        outputs = model(x1, x2, x3)
        probabilities = torch.exp(outputs)  # 转换为概率
        predictions = probabilities.argmax(dim=1)  # 获取预测类别

        batch_size = x1.size(0)
        # 获取当前批次对应的序列
        batch_sequences = sequences_new[current_index:current_index + batch_size]
        all_new_sequences.extend(batch_sequences)
        current_index += batch_size

        # 保存结果
        all_new_predictions.extend(probabilities.cpu().numpy())  # 保存预测概率
        all_new_labels.extend(y.cpu().numpy())  # 保存真实标签，如果没有标签，跳过这一步

# 将预测概率、预测标签和真实标签转换为DataFrame
df_new_predictions = pd.DataFrame(all_new_predictions, columns=['Prob_Class_0', 'Prob_Class_1'])
df_new_predictions['Predicted_Label'] = np.argmax(all_new_predictions, axis=1)  # 添加预测标签
df_new_predictions['True_Label'] = all_new_labels  # 添加真实标签
df_new_predictions['Sequence'] = all_new_sequences  # 添加序列

# 重新排列列顺序
df_new_predictions = df_new_predictions[['Sequence', 'Predicted_Label', 'True_Label', 'Prob_Class_0', 'Prob_Class_1']]

# 保存为CSV文件
df_new_predictions.to_csv("new_predictions.csv", index=False)

print("Prediction complete. Results saved to new_predictions.csv.")
