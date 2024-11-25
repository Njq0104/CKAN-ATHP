import numpy as np
from sklearn.model_selection import train_test_split

# 第一步：加载特征文件
F = np.load("feature.npy")

# 获取形状信息
num_sequences, feature_dim = F.shape  # 输入已经是二维特征 (序列数量, 特征维度)

# 第二步：划分数据集为训练集和测试集
train_data, test_data = train_test_split(F, test_size=0.2, random_state=34)

# 第三步：定义数据增强方法，针对训练集进行增强
def augment_training_data(train_data, perturbation_factor, seed=None):
    if seed is not None:
        np.random.seed(seed)

    augmented_samples = []

    for i in range(len(train_data)):
        # 当前样本的特征向量
        original_sample = train_data[i]

        # 生成扰动向量 V，与特征向量形状一致
        V = np.random.uniform(0, 1, original_sample.shape)

        # 对所有特征进行数据增强
        F_new = original_sample + V * perturbation_factor * original_sample
        augmented_samples.append(F_new)

    return np.array(augmented_samples)

# 设置扰动系数
perturbation_factor = 0.2

# 对训练集进行数据增强
augmented_train_data = augment_training_data(train_data, perturbation_factor)

# 第四步：合并增强后的训练集和原始的测试集
final_dataset = np.vstack([augmented_train_data, test_data])

# 保存最终的二维数据集到新文件
np.save("feature-0.2.npy", final_dataset)

print(f"增强后的二维特征文件已保存")
