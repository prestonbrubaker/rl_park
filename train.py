import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time

DATA_POINTS = 49990

class WillohNet(nn.Module):
    def __init__(self):
        super(WillohNet, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
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
        v = random.randint(0, DATA_POINTS - 1)
        data_sl = data[u][v]
        label_sl = labels[u][v]

        pred = willohnet(data_sl)

        print(f'Data: {list(data_sl)}\nPrediction: {float(pred)}\nActual: {float(label_sl)}\n\n')

        time.sleep(1)

examples(10)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(willohnet.parameters(), lr=0.001)

willohnet.train()



for j in range(10):
    for i in range(DATA_POINTS):
        pred = willohnet(data)
        loss = loss_fn(pred, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%10==0:
            print(loss.item())
        torch.save(willohnet.state_dict(), 'willohnet.pth')
    