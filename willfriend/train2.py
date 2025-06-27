import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import time


class WillohNet(nn.Module):
    def __init__(self):
        super(WillohNet, self).__init__()
        self.fc1 = nn.Linear(1203, 1203)
        self.fc2 = nn.Linear(1203, 300)
        self.fc3 = nn.Linear(300, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)

        self.fc6 = nn.Linear(1203,1)

    def forward(self, x):
        y = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x) + self.fc6(y)
        return x

data = torch.load("data_new_tensor.pth")
labels = torch.load("qs_tensor.pth")

print(data.shape)
print(labels.shape)

device = torch.device('cuda', 0)
willohnet = WillohNet().to(device)
data = torch.load("data_new_tensor.pth").to(device)
labels = torch.load("qs_tensor.pth").to(device)
willohnet.load_state_dict(torch.load('willohnet.pth'))
device = torch.device('cuda', 0)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(willohnet.parameters(), lr=0.00001)
willohnet.train()

for j in range(100000):


    pred = willohnet(data)
    loss = loss_fn(pred, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if j%20==0:
        print(loss.item())
        if j%200==0:
            torch.save(willohnet.state_dict(), 'willohnet.pth')
            print("SAVED!")

