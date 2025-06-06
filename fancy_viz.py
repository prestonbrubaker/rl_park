import asyncio
import platform
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

pygame.init()
window = pygame.display.set_mode((800, 800))
FPS = 10

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

def setup():
    model = WillohNet()
    model.load_state_dict(torch.load('willohnet.pth'))
    model.eval()
    return model, random.uniform(-5, 5), random.uniform(-5, 5)

async def update_loop(model, x, y):
    window.fill((0, 0, 0))
    pygame.draw.rect(window, (255, 255, 0), (400 - 2, 0, 4, 800))
    pygame.draw.rect(window, (255, 24, 11), (x * 700 / 10 + 400 - 10, y * 700 / 10 + 400 - 10, 20, 20))
    pygame.display.flip()

    max_m = 0
    max_i = 0
    for i in range(4):
        data = [0] * 8
        data[i] = 1
        data[4] = x
        data[5] = y
        # Pad the last two dimensions with zeros
        data[6] = 0
        data[7] = 0
        pred = model(torch.tensor(data).float())
        pred = pred.item()
        if pred > max_m:
            max_m = pred
            max_i = i

    if max_i == 0:
        x += random.uniform(0, 1) / 10
        if x > 5:
            x = 5
    elif max_i == 1:
        y += random.uniform(0, 1) / 10
        if y > 5:
            y = 5
    elif max_i == 2:
        x -= random.uniform(0, 1) / 10
        if x < -5:
            x = -5
    elif max_i == 3:
        y -= random.uniform(0, 1) / 10
        if y < -5:
            y = -5

    print(x)
    return x, y

async def main():
    model, x, y = setup()
    while True:
        x, y = await update_loop(model, x, y)
        await asyncio.sleep(1.0 / FPS)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
