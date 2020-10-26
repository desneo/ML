```
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from torch import optim
import glob
import os
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torch

trainPath = "G:\\practice\\pycharm\\datasets\\mnistasjpg\\trainingSet\\trainingSet\\*\\*.jpg"
trainTestPath = "G:\\practice\\pycharm\\datasets\\mnistasjpg\\trainingSample\\trainingSample\\*\\*.jpg"


# 定义数据源
class MnistDataset(Dataset):
    def __init__(self, root_dir):
        self.files = glob.glob(root_dir)
        self.x = 123

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.files[idx]))
        label = os.path.abspath(self.files[idx]).split("\\")[-2]
        return img,label


trainMiniSet = MnistDataset(trainPath)
trainTestMiniSet = MnistDataset(trainTestPath)
trainDataLoader = DataLoader(trainMiniSet, batch_size=64, num_workers=0, shuffle=True)
trainTestDataLoader = DataLoader(trainTestMiniSet, batch_size=64, num_workers=0, shuffle=True)


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super.__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.ReLU(nn.MaxPool2d(self.conv1(x), 2))
        x = nn.ReLU(nn.MaxPool2d(self.conv2_drop(self.conv2(x)), 2))
        x = nn.ReLU(self.fc1(x))
        x = nn.Dropout


# 定义网络结构
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2))

        self.dense = nn.Sequential(
            nn.Linear(14 * 14 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense(x)
        return x


model = Model()
cost = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

print(model)

n_epochs = 5

for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    for data, label in trainDataLoader:
        x_train = data
        y_train = label
        x_train = Variable(x_train)
        # x_train = Variable(x_train)
        outputs = model(x_train)
        _,pred = torch.max(outputs, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)

        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_correct += torch.sum((pred == y_train.data))

    testing_correct = 0
    for data in trainTestDataLoader:
        x_test,y_test = data
        x_test, y_test = Variable(x_test), Variable(y_test)
        outputs = model(x_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct = torch.sum(pred == y_test.data)

    # print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss/len()))
    print(" Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(100*running_correct / len(trainMiniSet), 100*testing_correct/len(trainTestMiniSet)))
```