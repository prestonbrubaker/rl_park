import torch

games = torch.load("games.pth")
labels = torch.load("labels.pth")
print(games.shape)
print(labels.shape)

qs = []
gs = []

for k in range(len(labels)):
    gt = []
    qt = []
    for i in range(50000 - 10):
        q = 0
        for j in range(10):
            q += labels[k][i + j] * 0.9 ** j
        qt.append(q)
        gt.append(list(games[k][i]))
    qs.append(qt)
    gs.append(gt)

qs = torch.tensor(qs).float()
gs = torch.tensor(gs).float()

torch.save(qs, "qs.pth")
torch.save(gs, "gs.pth")

print(qs.shape)
print(gs.shape)
