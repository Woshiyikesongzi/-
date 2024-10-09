import torchvision
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
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
cnnnet = CNNnet().to(device)
optimizer = torch.optim.Adam(cnnnet.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
loss_func=loss_func.to(device)
#设置训练网络的一些参数
#记录训练次数
total_train_step=0
#记录测试次数
total_test_step=0
#训练的轮数
epoch=20
best_acc = 0.0
for i in range(epoch):
    print(f'第{i}轮训练开始')
    #训练步骤开始
    cnnnet.train()
    for imgs,targets in train_dataloader:
        imgs,targets = imgs.to(device),targets.to(device)
        outputs=cnnnet(imgs)
        loss=loss_func(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()#将所有模型参数的梯度缓存清零
        loss.backward()#计算损失函数相对于模型参数的梯度(反向传播)
        optimizer.step()#根据计算出的梯度更新模型参数
        total_train_step+=1
        if total_train_step%100==0:
            print(f"训练次数{total_train_step}，loss{loss.item()}]")
    #验证步骤开始
    cnnnet.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for imgs,targets in test_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs=cnnnet(imgs)
            loss=loss_func(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accuracy
    avg_test_loss = total_test_loss / len(test_dataloader)
    avg_accuracy = total_accuracy / test_data_size
    print(f'整体测试集上的loss：{total_test_loss}')
    print(f'整体测试集上的正确率：{100*avg_accuracy:.2f}%')
    total_test_step+=1
    if avg_accuracy>best_acc:
        best_acc=avg_accuracy
        print(f'目前最好的正确率：{best_acc}')
        torch.save(cnnnet.state_dict(), "best_model.pth")
        #torch.save(cnnnet, f"cnnnet{i}.pth")
        #官方推荐的保存方式：
        #torch.save((cnnnet.state_dict(),f"cnnnet{i}.pth"))
        print(f'Epoch{i+1}:模型已保存,验证准确率：{100*avg_accuracy:.2f}%')




