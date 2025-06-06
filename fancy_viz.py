import pygame
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

# Define the neural network class
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

# Initialize Pygame
pygame.init()
window = pygame.display.set_mode((800, 800))
font = pygame.font.SysFont("roboto", 18)

# Load the trained model
model = WillohNet()
model.load_state_dict(torch.load('willohnet.pth'))
model.eval()

# Initialize positions and score
u = random.randint(0, 800)  # Sun x-coordinate
v = random.randint(0, 800)  # Sun y-coordinate
x = random.randint(0, 800)  # Agent x-coordinate
y = random.randint(0, 800)  # Agent y-coordinate
score = 0.0
temperature = 0.1  # Temperature for softmax action selection

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move the sun randomly
    r = random.randint(0, 3)
    if r == 0 and u < 800:
        u += 1  # Right
    elif r == 1 and v < 800:
        v += 1  # Down
    elif r == 2 and u > 0:
        u -= 1  # Left
    elif r == 3 and v > 0:
        v -= 1  # Up

    # Evaluate all possible actions and compute softmax probabilities
    action_values = []
    state_base = [0.0, 0.0, 0.0, 0.0, float(u - x), float(v - y), float(x), float(y)]
    for a in range(4):
        state = state_base.copy()
        state[a] = 1.0
        pred = model(torch.tensor(state).float()).item()
        action_values.append(pred)
    
    # Apply softmax with temperature
    action_values = np.array(action_values)
    exp_values = np.exp(action_values / temperature)
    softmax_probs = exp_values / np.sum(exp_values)
    
    # Choose an action based on softmax probabilities
    best_a = np.random.choice(4, p=softmax_probs)

    # Move the agent based on the selected action
    if best_a == 0 and y > 0:
        y -= 1  # Up
    elif best_a == 1 and x > 0:
        x -= 1  # Left
    elif best_a == 2 and y < 800:
        y += 1  # Down
    elif best_a == 3 and x < 800:
        x += 1  # Right

    # Calculate the reward
    dist = ((x - u) ** 2 + (y - v) ** 2) ** 0.5
    if x > u - 10 and x < u + 10 and y > v - 10 and y < v + 10:
        reward = 1 - dist / 100000
    else:
        reward = -dist / 100000
    score += reward

    # Render the game
    window.fill((0, 0, 0))  # Black background
    pygame.draw.rect(window, (255, 255, 255), (u - 40, v - 40, 80, 80))  # Sun
    pygame.draw.rect(window, (255, 0, 0), (x - 10, y - 10, 20, 20))  # Agent
    text = font.render(f"Score: {score:.2f}", True, (0, 255, 0))  # Score text
    text_rect = text.get_rect(center=(100, 20))
    window.blit(text, text_rect)
    pygame.display.flip()

    #time.sleep(0.1)  # Control the speed (10 steps per second)

pygame.quit()