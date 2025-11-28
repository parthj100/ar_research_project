import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3 import PPO
from envs.gridworld import GridworldEnv
from models.student import StudentPolicy

# 1) Collect teacher dataset
teacher = PPO.load("results/teacher_ppo_gridworld")
env = GridworldEnv()

states, actions = [], []
for ep in range(500):
    obs, _ = env.reset()
    done = False
    while not done:
        act, _ = teacher.predict(obs, deterministic=True)
        states.append(obs.copy())
        actions.append(int(act))
        obs, r, done, _, _ = env.step(act)

states = torch.tensor(np.array(states), dtype=torch.float32)
actions = torch.tensor(np.array(actions), dtype=torch.long)
np.savez("results/teacher_dataset.npz", states=states.numpy(), actions=actions.numpy())

# 2) Train student by imitating teacher actions (cross-entropy)
data = np.load("results/teacher_dataset.npz")
ds = TensorDataset(torch.tensor(data["states"], dtype=torch.float32),
                   torch.tensor(data["actions"], dtype=torch.long))
dl = DataLoader(ds, batch_size=64, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model = StudentPolicy().to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)
ce = nn.CrossEntropyLoss()

for epoch in range(20):
    tot = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}: loss={tot/len(ds):.4f}")

torch.save(model.state_dict(), "results/student_policy.pt")
print("Saved student to results/student_policy.pt")