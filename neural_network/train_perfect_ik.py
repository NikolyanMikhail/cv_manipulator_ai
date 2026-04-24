import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def get_kinematics(j1, j2, j3, j4):
    r_base = 0.012
    z_base = 0.0595
    
    # Плечо
    r3 = r_base + 0.024 * np.cos(j2) + 0.128 * np.sin(j2)
    z3 = z_base - 0.024 * np.sin(j2) + 0.128 * np.cos(j2)
    
    # Предплечье (Локоть)
    j23 = j2 + j3
    r4 = r3 + 0.124 * np.cos(j23)
    z4 = z3 - 0.124 * np.sin(j23)
    
    # Кисть
    j234 = j2 + j3 + j4
    r_grip = r4 + 0.126 * np.cos(j234)
    z_grip = z4 - 0.126 * np.sin(j234)
    
    x = r_grip * np.cos(j1)
    y = r_grip * np.sin(j1)
    
    return x, y, z_grip, j234

print("Генерация Свободного Датасета (Разгибание руки)...")
num = 8000000 
j1 = np.random.uniform(-3.14, 3.14, num)

j2 = np.random.uniform(0.0, 1.5, num)   # Плечо вперед
j3 = np.random.uniform(-1.5, 0.5, num)  # ЛОКОТЬ МОЖЕТ ВЫПРЯМЛЯТЬСЯ
j4 = np.random.uniform(-1.5, 1.5, num)  # Кисть свободна

x, y, z, p = get_kinematics(j1, j2, j3, j4)
mask = (z > 0.01) & (z < 0.4) & (np.sqrt(x**2 + y**2) > 0.12) & (np.sqrt(x**2 + y**2) < 0.38) & (p > 0.5) & (p < 2.0)

X_t = torch.tensor(np.stack([x[mask], y[mask], z[mask]], axis=1), dtype=torch.float32)
Y_t = torch.tensor(np.stack([j1[mask], j2[mask], j3[mask], j4[mask]], axis=1), dtype=torch.float32)

print(f"Собрано {X_t.shape[0]} свободных поз. Обучаем...")

class IKNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))
    def forward(self, x): return self.net(x)

model = IKNet()
opt = optim.Adam(model.parameters(), lr=0.001)

epochs = 1200
batch_size = 1024
for e in range(epochs):
    opt.zero_grad()
    idx = torch.randperm(X_t.size()[0])[:batch_size]
    loss = nn.MSELoss()(model(X_t[idx]), Y_t[idx])
    loss.backward()
    opt.step()
    if e % 200 == 0: print(f"Эпоха {e}/1200 | Loss: {loss.item():.6f}")

torch.save(model.state_dict(), 'ik_model.pth')
print("реади!")