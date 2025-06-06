import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import time

DATA_POINTS = 49990

class WillohNet(nn.Module):
    def __init__(self):
        super(WillohNet, self).__init__()
        self.fc1 = nn.Linear(8, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 100)
        self.fc7 = nn.Linear(100, 10)
        self.fc8 = nn.Linear(10, 1)
        self.fc9 = nn.Linear(8, 1)

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
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x) + self.fc9(y)
        return x


data = torch.load("gs.pth")
labels = torch.load("qs.pth")

print(data.shape)
print(labels.shape)



random_data = torch.rand((8))
willohnet = WillohNet()

willohnet.load_state_dict(torch.load('willohnet.pth'))

device = torch.device('cuda', 0)
print(f"Using {device} device")

result = willohnet(random_data)
print(random_data)
print(result)



def examples(i):

    if i==0:
        return
    for k in range(i):
        u = random.randint(0,len(data) - 1)
        v = random.randint(0, len(data[0]) - 1)
        data_sl = data[u][v]
        label_sl = labels[u][v]

        pred = willohnet(data_sl)

        print(f'Data: {list(data_sl)}\nPrediction: {float(pred)}\nActual: {float(label_sl)}\n\n')

        time.sleep(1)

examples(0)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(willohnet.parameters(), lr=0.00001)

willohnet.train()



for j in range(10000):
    for i in range(len(data[0])):

        k = len(data)

        part = 80

        p = random.randint(0, k - part - 1)

        lower_i = random.randint(p,p+part)


        data_sl = data[lower_i:lower_i+part]
        labels_sl = labels[lower_i:lower_i+part]

        pred = willohnet(data_sl)
        loss = loss_fn(pred, labels_sl)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%1==0:
            print(loss.item() / part)
            torch.save(willohnet.state_dict(), 'willohnet.pth')
    
