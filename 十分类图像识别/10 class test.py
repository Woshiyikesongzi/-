import torch
import  torchvision
from PIL import Image
from torch import nn
image_path='test image/fish.png'
image=Image.open(image_path)
print(image)
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
image=transform(image)
print(image.shape)
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
# 实例化模型并加载权重
model = CNNnet()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 将图像调整为模型输入的形状
image = torch.reshape(image, (1, 3, 32, 32))
# 定义类别映射字典
class_names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

# 进行预测
with torch.no_grad():
    output = model(image)
predicted_class = output.argmax(1).item()
print(f'预测结果: {class_names[predicted_class]}')
