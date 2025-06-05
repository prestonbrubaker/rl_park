import torch
import torch.nn as nn
import torch.nn.functional as F
import random

DATA_POINTS = 10000


class WillohNet(nn.Module):
    def __init__(self):
        super(WillohNet, self).__init__()
        self.fc1 = nn.Linear(6, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x



random_data = torch.rand((6))
willohnet = WillohNet()
result = willohnet(random_data)
print(random_data)
print(result)


score = 0
log = []
labels = []
for i in range(DATA_POINTS):
    if i%10==0:
        x = random.uniform(-5,5)
        y = random.uniform(-5,5)
    data = [0] * 4
    r = random.randint(0,3)
    ds = 0
    if (r == 0 and x < 0) or (r == 2 and x >= 0):
        ds = abs(y) / 5
        score += ds
    
    if r == 0:
        x += random.uniform(0, 1)
        if x>5:
            x = 5
    elif r == 2:
        x-= random.uniform(0,1)
        if x<-5:
            x = -5
    if r == 1:
        y += random.uniform(0, 1)
        if y>5:
            y = 5
    elif r == 3:
        y-= random.uniform(0,1)
        if y<-5:
            y = -5
    
    data[r] = 1
    data.append(x)
    data.append(y)
    labels.append([ds])
    log.append(data)

print(log)

d = torch.tensor(log).float()
l = torch.tensor(labels).float()

print(d)
print(labels)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(willohnet.parameters(), lr=0.001)

willohnet.train()

for j in range(10):
    for i in range(DATA_POINTS):
        pred = willohnet(d)
        loss = loss_fn(pred, l)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%10==0:
            print(loss.item())


for i in range(4):
    data = [0] * 4
    data[i] = 1
    data.append(random.uniform(-5,5))
    data.append(random.uniform(-5,5))
    d = torch.tensor(data).float()
    pred = willohnet(d)
    print(f"Input: {data}     Prediction: {pred}")

torch.save(willohnet.state_dict(), 'willohnet.pth')
print("Model saved to 'willohnet.pth'")
