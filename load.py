import torch

games = torch.load("games.pth")
labels = torch.load("labels.pth")
print(games.shape)
print(labels.shape)

qs = []
gs = []

for k in range(len(labels)):
    print(k)
    gt = []
    qt = []
    for i in range(len(games)):
        q = 0
        for j in range(10):
            q += labels[k][i + j] * 0.9 ** j
        qt.append([q])
        gtt = []
        for v in range(8):
            if v <= 3:
                gtt.append(float(games[k][i][v]))
            else:
                gtt.append(float(games[k][i][v]) / 800)
        gt.append(gtt)
    qs.append(qt)
    gs.append(gt)

    qs_t = torch.tensor(qs).float()
    gs_t = torch.tensor(gs).float()

    torch.save(qs_t, "qs.pth")
    torch.save(gs_t, "gs.pth")

#print(qs.shape)
#print(gs.shape)
