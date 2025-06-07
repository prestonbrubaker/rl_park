import pygame
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle

# Initialize Pygame
pygame.init()
window = pygame.display.set_mode((800, 800))
font = pygame.font.SysFont("roboto", 18)

# Game variables
x, y = 0, 400
g, sens = 0.06, 5
yv, xv = 0, 0
plat_x, plat_y = 0, 300
check_c = 10
score = 0

# Platform array
a = [0 if random.uniform(0, 1) < 0.9 else 1 for _ in range(1000)]

# Data collection flag
collect_data = True  # Set to False to disable data collection
dataset = [] if collect_data else None

# CNN model (must match the trained model)
class GameCNN(nn.Module):
    def __init__(self):
        super(GameCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GameCNN().to(device)
model.load_state_dict(torch.load('game_cnn.pth', weights_only=True))
model.eval()

# Transform for screen input (matches training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Temperature for softmax
temperature = 1.0  # Higher values increase randomness, lower values make it more deterministic

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Capture and process screen
    screen_data = pygame.surfarray.array3d(window)  # Shape: (800, 800, 3)
    screen_data = screen_data / 255.0  # Normalize
    screen_tensor = torch.tensor(screen_data, dtype=torch.float32)  # Shape: (800, 800, 3)
    
    # Convert to NumPy for transform
    screen_np = screen_tensor.numpy()  # Convert to (800, 800, 3) NumPy array
    screen_tensor = transform(screen_np).unsqueeze(0).to(device)  # Shape: (1, 3, 64, 64)

    # Get CNN predictions
    with torch.no_grad():
        logits = model(screen_tensor)  # Shape: (1, 4)
        # Apply softmax with temperature
        probs = F.softmax(logits / temperature, dim=1).squeeze(0)  # Shape: (4,)
        probs = probs.cpu().numpy()

    # Sample actions based on probabilities
    control_input = np.random.binomial(1, probs).astype(np.float32)  # Binary vector [w, a, s, d]

    # Store data if enabled
    if collect_data:
        dataset.append({
            'screen': torch.tensor(screen_data, dtype=torch.float32),  # Save original 800x800x3
            'control': torch.tensor(control_input, dtype=torch.float32),
            'score': torch.tensor(score, dtype=torch.float32)
        })

    # Game logic (same as original)
    x_old = x + plat_x

    if control_input[0] and y < plat_y + 10:  # W
        y += sens * 0.1
        yv += 0.8
    if control_input[1]:  # A
        if x > 100:
            x -= sens
        else:
            plat_x -= sens
    if control_input[2]:  # S
        y -= sens
    if control_input[3]:  # D
        if x < 400:
            x += sens
        else:
            plat_x += sens
    
    yv -= g
    y += yv

    if y < plat_y and not a[int((plat_x + x) / 800 * check_c)]:
        y = plat_y
        yv = 0

    ds = x + plat_x - x_old
    if y < plat_y:
        ds -= plat_y - y
    score += ds

    # Render
    window.fill((0, 0, 0))
    for i in range(check_c + 2):
        color = (255, 255, 0) if (i + int(plat_x * check_c / 800)) % 2 == 0 else (255, 255, 255)
        color = (0, 0, 0) if a[i + int(plat_x / 800 * check_c)] else color
        plat_x += 0.001
        pygame.draw.rect(window, color, (i * 800 / check_c - plat_x % (800 / check_c), 800 - plat_y, 800 / check_c, plat_y))
    pygame.draw.rect(window, (255, 0, 0), (x - 25, 800 - y - 50, 50, 50))
    text = font.render(f"Score: {score:.2f}", True, (0, 255, 0))
    text_rect = text.get_rect(center=(100, 20))
    window.blit(text, text_rect)
    pygame.display.flip()

    time.sleep(0.01)

# Save dataset if enabled
if collect_data:
    with open('ai_game_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

pygame.quit()