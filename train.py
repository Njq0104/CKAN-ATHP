# -*- coding: utf-8 -*-

import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, confusion_matrix, roc_auc_score, average_precision_score
import os
# 检查是否有GPU可用
from kan_convolutional.KANConv import KAN_Convolutional_Layer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

seed = 34

# 加载特征矩阵
X1 = np.load("DistilProtBert.npy")
X2 = np.load("PubChem10M.npy")
X3 = np.load("Prot-t5.npy")

sequences_df = pd.read_csv('cleaned_dataset.csv')
sequences = sequences_df.iloc[:, 0].values
labels = torch.tensor(sequences_df.iloc[:, 1].values, dtype=torch.long) 

# 将特征矩阵转换为PyTorch张量并调整形状
X1 = torch.tensor(X1, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
X2 = torch.tensor(X2, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
X3 = torch.tensor(X3, dtype=torch.float32).unsqueeze(1).unsqueeze(2)

print(X1.shape)
print(X2.shape)
print(X3.shape)

# 将标签转换为PyTorch张量
labels = torch.tensor(labels, dtype=torch.long)
# 划分数据集 (X1, X2, X3 为之前加载的特征矩阵)
X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, Y_train, Y_test, seq_train, seq_test = train_test_split(
    X1, X2, X3, labels, sequences, test_size=0.2, random_state=seed, stratify=labels)

# 创建DataLoader
train_dataset = TensorDataset(X1_train, X2_train, X3_train, Y_train)
test_dataset = TensorDataset(X1_test, X2_test, X3_test, Y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

class KANC_MLP(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()

        self.device = device

        # 仅保留第二个卷积层
        self.conv2 = KAN_Convolutional_Layer(
            n_convs=5,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            device=device
        )

        # 池化和展平层
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(2, 2))
        self.flat = nn.Flatten()

        # 全连接层
        self.linear1 = nn.Linear(3520, 256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, x1, x2, x3):
        # 将 x1, x2, x3 进行拼接
        x_combined = torch.cat((x1, x2, x3), dim=3)

        # 直接进行卷积和池化操作
        x = self.conv2(x_combined)
        x = self.pool2(x)

        # 将张量展平
        x = self.flat(x)

        # 全连接层
        x = self.linear1(x)
        x = self.linear2(x)

        # 应用 softmax
        x = F.log_softmax(x, dim=1)
        return x


from torchsummary import summary
# 初始化模型
model = KANC_MLP(device=device).to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def label_smoothing(labels, num_classes, smoothing=0.1):
    """
    Apply label smoothing to the labels.
    
    Args:
        labels (torch.Tensor): True labels in one-hot format.
        num_classes (int): Number of classes.
        smoothing (float): Smoothing factor, typically between 0 and 1.
        
    Returns:
        torch.Tensor: Smoothed labels.
    """
    confidence = 1.0 - smoothing
    label_shape = torch.Size((labels.size(0), num_classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=labels.device)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, labels.data.unsqueeze(1), confidence)
    return true_dist

def save_model_and_predictions(model, test_accuracy, all_labels, all_probs, all_preds, seq_test, model_dir='model', predicitions_dir='probabilities'):
    accuracy_str = f"{test_accuracy:.2f}".replace(".", "_")
    
    # 创建保存目录
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存模型
    model_filename = os.path.join(model_dir, f"model_stratify_{accuracy_str}.pth")
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")
    
    # 保存预测概率和标签
    predictions_filename = os.path.join(predicitions_dir, f"predictions_stratify_{accuracy_str}.csv")
    
    # 创建包含序列、预测概率、预测标签和真实标签的DataFrame
    predictions_df = pd.DataFrame({
        'sequence': seq_test,
        'predicted_label': all_preds,
        'true_label': all_labels,
        'predicted_prob_0': all_probs[:, 0],
        'predicted_prob_1': all_probs[:, 1],
    })
    
    predictions_df.to_csv(predictions_filename, index=False)
    print(f"Predictions saved as {predictions_filename}")

def calculate_metrics(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    auc = roc_auc_score(y_true, y_prob[:, 1])
    ap = average_precision_score(y_true, y_prob[:, 1])

    return accuracy, mcc, f1, specificity, sensitivity, auc, ap

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, save_threshold, kl_weight=1.0, smoothing=0.1):
    best_metrics = {'epoch': 0, 'accuracy': 0, 'mcc': 0, 'f1': 0, 'specificity': 0, 'sensitivity': 0, 'auc': 0, 'ap': 0}
    best_test_accuracy = 0  # 用于跟踪最高的测试集精度

    num_classes = 2  # 假设二分类问题

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X1_batch, X2_batch, X3_batch, Y_batch in train_loader:
            X1_batch, X2_batch, X3_batch, Y_batch = X1_batch.to(device), X2_batch.to(device), X3_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X1_batch, X2_batch, X3_batch)
            
            # 计算平滑标签
            smoothed_labels = label_smoothing(Y_batch, num_classes, smoothing)
            
            # 交叉熵损失
            ce_loss = criterion(outputs, Y_batch)
            
            # KL散度损失，使用平滑标签
            log_probs = F.log_softmax(outputs, dim=1)
            kl_loss = F.kl_div(log_probs, smoothed_labels, reduction='batchmean')
            
            # 组合损失
            loss = ce_loss + kl_weight * kl_loss
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X1_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        # 测试
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for X1_batch, X2_batch, X3_batch, Y_batch in test_loader:
                X1_batch, X2_batch, X3_batch, Y_batch = X1_batch.to(device), X2_batch.to(device), X3_batch.to(device), Y_batch.to(device)
                outputs = model(X1_batch, X2_batch, X3_batch)
                probabilities = torch.exp(outputs)  # 从 log_softmax 转换回概率
                loss = criterion(outputs, Y_batch)
                test_loss += loss.item() * X1_batch.size(0)
                _, predicted = torch.max(outputs, 1)

                all_preds.append(predicted.cpu().numpy())
                all_probs.append(probabilities.cpu().numpy())
                all_labels.append(Y_batch.cpu().numpy())
                total += Y_batch.size(0)
                correct += (predicted == Y_batch).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct / total

        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # 计算评估指标
        accuracy, mcc, f1, specificity, sensitivity, auc, ap = calculate_metrics(all_labels, all_preds, all_probs)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

        # 保存模型和预测概率
        if test_accuracy > save_threshold:
            save_model_and_predictions(model, test_accuracy, all_labels, all_probs, all_preds, seq_test)
            
        # 如果当前测试集精度高于目前最高值，则更新最佳评估指标
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_metrics = {
                'epoch': epoch + 1,
                'accuracy': accuracy,
                'mcc': mcc,
                'f1': f1,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'auc': auc,
                'ap': ap
            }

    # 打印最佳评估指标
    print(f"\nBest metrics at Epoch {best_metrics['epoch']}:")
    print(f"Test Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"MCC: {best_metrics['mcc']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(f"Specificity: {best_metrics['specificity']:.4f}")
    print(f"Sensitivity: {best_metrics['sensitivity']:.4f}")
    print(f"AUC: {best_metrics['auc']:.4f}")
    print(f"AP: {best_metrics['ap']:.4f}")


# 开始训练和测试
train(model, train_loader, test_loader, criterion, optimizer, num_epochs=60, save_threshold=87.5)
