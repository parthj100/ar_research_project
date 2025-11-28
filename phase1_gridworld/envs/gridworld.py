import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridworldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_size=5, max_steps=50, step_penalty=-0.01):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.step_penalty = step_penalty

        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # up, down, left, right, stay
        self._rng = np.random.default_rng()
        self.reset(seed=None)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.agent = self._rng.integers(0, self.grid_size, size=2)
        while True:
            self.goal = self._rng.integers(0, self.grid_size, size=2)
            if not np.array_equal(self.goal, self.agent):
                break
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.agent[0], self.agent[1], self.goal[0], self.goal[1]], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        if action == 0:   self.agent[1] += 1     # up
        elif action == 1: self.agent[1] -= 1     # down
        elif action == 2: self.agent[0] -= 1     # left
        elif action == 3: self.agent[0] += 1     # right
        # 4: stay

        self.agent = np.clip(self.agent, 0, self.grid_size-1)

        done = False
        reward = self.step_penalty

        if np.array_equal(self.agent, self.goal):
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, False, {}