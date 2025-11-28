import time, numpy as np, torch
from stable_baselines3 import PPO
from envs.gridworld import GridworldEnv
from models.student import StudentPolicy

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def run_teacher_episode(model, env, net_delay_ms=50):
    obs, _ = env.reset(); done = False; steps = 0; total_ms = 0.0
    while not done:
        t0 = time.time()
        time.sleep(net_delay_ms/1000.0)               # send -> server
        action, _ = model.predict(obs, deterministic=True)
        time.sleep(net_delay_ms/1000.0)               # server -> recv
        total_ms += (time.time()-t0)*1000.0
        obs, r, done, _, _ = env.step(action)
        steps += 1
    reached = int(np.array_equal(env.agent, env.goal))
    bytes_per_step = 32  # approx; 4 floats + overhead
    return reached, steps, total_ms/max(steps,1), bytes_per_step*steps

def run_student_episode(student, env):
    obs, _ = env.reset(); done = False; steps = 0; total_ms = 0.0
    while not done:
        t0 = time.time()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = student(x)
            action = int(torch.argmax(logits, dim=1).item())
        total_ms += (time.time()-t0)*1000.0
        obs, r, done, _, _ = env.step(action)
        steps += 1
    reached = int(np.array_equal(env.agent, env.goal))
    return reached, steps, total_ms/max(steps,1), 0

def run_student_with_hints(student, teacher, env, k=10, net_delay_ms=50):
    obs, _ = env.reset(); done = False; steps = 0; total_ms = 0.0; bytes_sent = 0
    while not done:
        # student proposes
        t0 = time.time()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = student(x)
            s_action = int(torch.argmax(logits, dim=1).item())
        total_ms += (time.time()-t0)*1000.0

        # occasional teacher hint
        if steps % k == 0:
            time.sleep(net_delay_ms/1000.0)
            t_action, _ = teacher.predict(obs, deterministic=True)
            time.sleep(net_delay_ms/1000.0)
            bytes_sent += 32   # approx state payload
            action = int(t_action)
        else:
            action = s_action

        obs, r, done, _, _ = env.step(action)
        steps += 1
    reached = int(np.array_equal(env.agent, env.goal))
    return reached, steps, total_ms/max(steps,1), bytes_sent

if __name__ == "__main__":
    env = GridworldEnv()
    teacher = PPO.load("results/teacher_ppo_gridworld")
    student = StudentPolicy().to(device)
    student.load_state_dict(torch.load("results/student_policy.pt", map_location=device))
    student.eval()

    N = 200
    modes = {
        "teacher_offdevice": lambda: run_teacher_episode(teacher, env, net_delay_ms=50),
        "student_ondevice": lambda: run_student_episode(student, env),
        "student_plus_hints": lambda: run_student_with_hints(student, teacher, env, k=10, net_delay_ms=50),
    }

    for name, fn in modes.items():
        succ, steps, lat, bytes_ = [], [], [], []
        for _ in range(N):
            s, st, l, b = fn()
            succ.append(s); steps.append(st); lat.append(l); bytes_.append(b)
        print(f"\n== {name} ==")
        print("Success rate:", np.mean(succ))
        print("Avg steps:", np.mean(steps))
        print("Avg per-step latency (ms):", np.mean(lat))
        print("Bytes/episode (mean):", np.mean(bytes_))