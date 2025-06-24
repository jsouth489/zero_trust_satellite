from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from gym import Env, spaces
import numpy as np
import pandas as pd


class SatelliteEnv(Env):
    def __init__(self, policy_data):
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)  # +1 for TPM policy
        self.action_space = spaces.Discrete(5)
        self.data = pd.read_csv(policy_data)
        self.idx = 0

    def reset(self):
        self.idx = np.random.randint(0, len(self.data))
        state = np.array([float(x) for x in self.data.iloc[self.idx]["state"].split(',')] + [
            1 if self.data.iloc[self.idx]["tpm_policy"] == "attest_1s" else 0], dtype=np.float32)
        return state, {}

    def step(self, action):
        reward = -np.random.rand() * 10 + (action == int(self.data.iloc[self.idx]["action"])) * 10
        done = False
        state = np.array([float(x) for x in self.data.iloc[self.idx]["state"].split(',')] + [
            1 if self.data.iloc[self.idx]["tpm_policy"] == "attest_1s" else 0], dtype=np.float32)
        return state, reward, done, False, {}


def train():
    env = DummyVecEnv([lambda: SatelliteEnv("../data/policy_training.csv") for _ in range(4)])
    model = PPO("MlpPolicy", env, verbose=1, device="cuda")

    model.learn(total_timesteps=100000)
    model.save("../models/ppo_policy_model")


if __name__ == "__main__":
    train()