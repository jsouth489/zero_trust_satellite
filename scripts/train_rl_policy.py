from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from gym import Env, spaces
import numpy as np


class SatelliteEnv(Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.state = None

    def reset(self):
        self.state = np.random.rand(10)
        return self.state, {}

    def step(self, action):
        reward = -np.random.rand() * 10 + (action == 0) * 10
        done = False
        return self.state, reward, done, False, {}


def train():
    env = DummyVecEnv([lambda: SatelliteEnv() for _ in range(4)])
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")

    model.learn(total_timesteps=100000)
    model.save("../models/ppo_policy_model")


if __name__ == "__main__":
    train()