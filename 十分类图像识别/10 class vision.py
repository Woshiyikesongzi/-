import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
#准备数据集
train_data=torchvision.datasets.CIFAR10(root=r"C:\Users\张\Desktop\文件\学习\python",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root=r"C:\Users\张\Desktop\文件\学习\python",train=False,transform=torchvision.transforms.ToTensor(),download=True)
train_data_size=len(train_data)
test_data_size=len(test_data)
print(f'训练数据集的长度：{train_data_size}')
print(f'训练数据集的长度：{test_data_size}')
#利用dataloader加载数据集
train_dataloader=DataLoader(train_data,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=64,shuffle=False)
# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# 构造CNN网络
class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        x = self.model(x)
        return x
# 加载模型并评估
cnnnet = CNNnet().to(device)
cnnnet.load_state_dict(torch.load("best_model.pth"))
cnnnet.eval()

features = []
labels = []
with torch.no_grad():
    for x_test, y_test in test_dataloader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        outputs = cnnnet(x_test)
        features.append(outputs)
        labels.append(y_test)

# 将特征和标签转换为 CPU 张量
features = torch.cat(features, dim=0).cpu().numpy()
labels = torch.cat(labels, dim=0).cpu().numpy()

# t-SNE 可视化（带 PCA 降维）
# 计算特征的实际维度
n_samples, n_features = features.shape
n_components = min(50, n_features, n_samples - 1)  # 保证 n_components 不超过 n_samples 或 n_features

# PCA 预降维
pca = PCA(n_components=n_components)
features_pca = pca.fit_transform(features)

# 使用 t-SNE 降维至 2 维
tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
tsne_results = tsne.fit_transform(features_pca)

# 可视化结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE Visualization after Model Training')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(handles=scatter.legend_elements()[0], labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
plt.show()

# 测试集计算各种指标
y_pred = np.argmax(features, axis=1)
acc = accuracy_score(labels, y_pred)
pre = precision_score(labels, y_pred, average='macro')
recall = recall_score(labels, y_pred, average='macro')
f1score = f1_score(labels, y_pred, average='macro')
print('计算指标结果：\nAcc: %.2f%% \nPre: %.2f%% \nRecall: %.2f%% \nF1-score: %.2f%% ' % (100*acc, 100*pre, 100*recall, 100*f1score))
# 绘制混淆矩阵
ConfusionMatrixDisplay(confusion_matrix(labels, y_pred), display_labels=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']).plot()
plt.show()
