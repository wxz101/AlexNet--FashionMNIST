import torch
import torchvision
import torchvision.transforms as transforms
import os

# 设置数据保存路径
data_dir = r"E:\FashionMNIST"
os.makedirs(data_dir, exist_ok=True)

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # AlexNet输入尺寸为224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 单通道归一化
])

# 下载数据集
train_dataset = torchvision.datasets.FashionMNIST(
    root=data_dir, train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root=data_dir, train=False, download=True, transform=transform
)

print("数据集已下载到:", data_dir)