import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import time

#config---超参数设置：
batch_size = 64  # 批处理大小,即一次训练所使用的样本数量
learning_rate = 0.01  # 学习率
num_epochs = 10

#数据路径：
data_dir = r"E:\FashionMNIST"

#数据加载：
transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))  #  单通道归一化
    ]
)

train_dataset = torchvision.datasets.FashionMNIST(
    root = data_dir,
    train = True,
    transform = transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root = data_dir,
    train = False,
    transform = transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False
)

#定义AlexNet模型：
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),  # 修改为256 * 6 * 6
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


#初始化模型，损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

#训练函数：
def train(model,train_loader,criterion,optimizer,epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx,(inputs,targets) in enumerate(train_loader):
        inputs,targets = inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output,targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _,predicted = output.max(1)  # 返回最大值对应的索引
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 99:
            print(f'Epoch: {epoch + 1} | Batch: {batch_idx + 1} | Loss: {running_loss / 100:.3f}')
            running_loss = 0.0
    train_acc = 100. * correct / total
    print(f'Train Accuracy: {train_acc:.2f}%')
    return train_acc

def test(model,test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,targets in test_loader:
            inputs,targets = inputs.to(device),targets.to(device)
            outputs = model(inputs)
            _,predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total
        print(f'Test Accuracy: {test_acc:.2f}%')
        return test_acc

#训练和测试：
train_accs,test_accs = [],[]
start_time = time.time()
for epoch in range(num_epochs):
    train_acc = train(model, train_dataloader, criterion, optimizer, epoch)
    test_acc = test(model, test_dataloader)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

print(f'Total Training Time: {time.time() - start_time:.2f}s')

#绘制准确度曲线：
plt.plot(train_accs,label = 'Train Accuracy')
plt.plot(test_accs,label = 'Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()