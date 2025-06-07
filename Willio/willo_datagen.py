import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

# Load the dataset
with open('game_dataset.pkl', 'rb') as f:
    loaded_dataset = pickle.load(f)

print(f"Original dataset size: {len(loaded_dataset)}")

# Parameters
gamma = 0.99  # Discount factor
truncate_frames = min(10, len(loaded_dataset) // 2)  # Adaptive truncation

# Compute Q-values for the dataset
def compute_q_values(dataset, gamma, truncate_frames):
    q_dataset = []
    for i in range(len(dataset) - truncate_frames):
        item = dataset[i]
        screen = item['screen']  # Shape: (800, 800, 3)
        control = item['control']  # Shape: (4,)
        # Compute dS (score differential)
        dS = item['score'].item() if i == 0 else item['score'].item() - dataset[i-1]['score'].item()

        # Compute future discounted reward
        future_reward = 0.0
        for j in range(i + 1, min(i + truncate_frames + 1, len(dataset))):
            future_dS = dataset[j]['score'].item() - dataset[j-1]['score'].item()
            future_reward += (gamma ** (j - i)) * future_dS
        
        # Q-value for each action
        q_values = torch.zeros(4, dtype=torch.float32)
        for action_idx in range(4):
            if control[action_idx] == 1:
                q_values[action_idx] = dS + gamma * future_reward
            else:
                q_values[action_idx] = 0  # No reward for untaken actions

        q_dataset.append({
            'screen': screen,
            'q_values': q_values
        })
    
    print(f"Q-dataset size: {len(q_dataset)}")
    return q_dataset

# Process dataset to include Q-values
q_dataset = compute_q_values(loaded_dataset, gamma, truncate_frames)

# Check if dataset is empty
if len(q_dataset) == 0:
    raise ValueError("Q-dataset is empty. Check dataset size or truncate_frames.")

# Define the custom dataset
class GameDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        screen = item['screen']  # Expected shape: (800, 800, 3)
        q_values = item['q_values']  # Shape: (4,)
        
        # Convert tensor to NumPy for ToPILImage
        screen = screen.numpy()  # Convert to (800, 800, 3) NumPy array
        
        if self.transform:
            screen = self.transform(screen)
        
        return screen, q_values

# Define transform to downsample images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()  # Converts to (3, 64, 64)
])

# Create dataset and dataloader
dataset = GameDataset(q_dataset, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the CNN model
class GameCNN(nn.Module):
    def __init__(self):
        super(GameCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 outputs for Q-values of W, A, S, D
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

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GameCNN().to(device)
criterion = nn.MSELoss()  # Regression loss for Q-values
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (screens, q_values) in enumerate(dataloader):
        screens = screens.to(device)  # Shape: (batch_size, 3, 64, 64)
        q_values = q_values.to(device)  # Shape: (batch_size, 4)
        
        optimizer.zero_grad()
        outputs = model(screens)  # Predicted Q-values
        loss = criterion(outputs, q_values)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 10 == 9:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/10:.4f}')
            running_loss = 0.0

print("Training finished!")
torch.save(model.state_dict(), 'game_cnn.pth')