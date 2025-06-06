import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pygame
import time

pygame.init()

window = pygame.display.set_mode((800,800))

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

random_data = torch.rand((8))

willohnet = WillohNet()
willohnet.load_state_dict(torch.load('willohnet.pth'))
willohnet.eval()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(willohnet.parameters(), lr=0.0001)

data = torch.load("gs.pth")
labels = torch.load("qs.pth")

running = True
x = random.uniform(-5,5)
y = random.uniform(-5,5)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    window.fill((0,0,0))

    pygame.draw.rect(window, (255, 255, 0), (400 - 2, 0, 4, 800))

    pygame.draw.rect(window, (255, 24, 11), (x*700/10 + 400-10, y*700/10 + 400-10, 20, 20))

    pygame.display.flip()

    for j in range(10):
        for i in range(DATA_POINTS):
            pred = willohnet(data)
            loss = loss_fn(pred, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if i%10==0:
                print(loss.item())

    time.sleep(0.1)
    print(x)
