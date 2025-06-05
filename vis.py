import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pygame
import time

pygame.init()

window = pygame.display.set_mode((800,800))


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


model = WillohNet()





model.load_state_dict(torch.load('willohnet.pth'))
model.eval()

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

    max_m = 0
    max_i = 0
    for i in range(4):
        data = [0]*4
        data[i] = 1
        data.append(x)
        data.append(y)
        pred = model(torch.tensor(data).float())
        pred = pred.item()
        if pred > max_m:
            max_m = pred
            max_i = i
    
    if max_i == 0:
        x += random.uniform(0, 1)/10
        if x>5:
            x=5
    if max_i == 1:
        y += random.uniform(0, 1)/10
        if y>5:
            y=5
    elif max_i == 2:
        x -= random.uniform(0, 1)/10
        if x<-5:
            x=-5
    elif max_i == 3:
        y -= random.uniform(0,1)/10
        if y<-5:
            y=-5
    time.sleep(0.1)
    print(x)
