import pickle
import torch
import torch.nn as nn


with open("./data.pkl", "rb") as f:
    data = pickle.load(f)

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)

print(data_tensor)
print(labels_tensor)

print(data_tensor.shape)
print(labels_tensor.shape)

data_new = []
qs = []

for i in range(len(labels) - 1):
    if i%3000 >= 3000 - 400 - 1:
        continue
    q = 0
    for j in range(400):
        #print(f'{i} {j}')
        q += labels[i + j][0] * 0.99 ** j
    data_new.append(data[i])
    qs.append([q])

data_new_tensor = torch.tensor(data_new, dtype=torch.float32)
qs_tensor = torch.tensor(qs, dtype=torch.float32)

print(data_new_tensor.shape)
print(qs_tensor.shape)

torch.save(data_new_tensor, "data_new_tensor.pth")
torch.save(qs_tensor, "qs_tensor.pth")
