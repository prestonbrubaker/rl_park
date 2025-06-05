import torch
import torch.nn as nn
import torch.nn.functional as F
import random

DATA_POINTS = 49990

class WillohNet(nn.Module):
    def __init__(self):
        super(WillohNet, self).__init__()
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


data = torch.load("gs.pth")
labels = torch.load("qs.pth")

print(data.shape)
print(labels.shape)

random_data = torch.rand((8))
willohnet = WillohNet()

willohnet.load_state_dict(torch.load('willohnet.pth'))

result = willohnet(random_data)
print(random_data)
print(result)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(willohnet.parameters(), lr=0.0001)

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
    