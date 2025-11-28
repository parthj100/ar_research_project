import gymnasium as gym
from stable_baselines3 import PPO
from envs.gridworld import GridworldEnv
import numpy as np

def make_env():
    return GridworldEnv(grid_size=5, max_steps=50)

if __name__ == "__main__":
    env = gym.wrappers.TimeLimit(make_env(), max_episode_steps=50)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
    model.save("results/teacher_ppo_gridworld")

    # Quick eval
    env = make_env()
    successes, steps_list = 0, []
    for _ in range(200):
        obs, _ = env.reset()
        done, steps = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = env.step(action)
            steps += 1
        if np.array_equal(env.agent, env.goal):
            successes += 1
        steps_list.append(steps)
    print("Teacher success rate:", successes/200, "Avg steps:", np.mean(steps_list))