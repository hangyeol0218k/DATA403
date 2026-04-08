import os
import random
from collections import deque

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.wrappers import RecordVideo
from torch.distributions import Normal


"""
TODO:
Train an agent that can reliably land between the flags and achieve a reward greater than 200.

Also, save:
- A video of the agent reliably landing between the flags
- Training logs
- A performance graph showing rewards exceeding 200

You may refer to the lecture notes if needed.
"""


class PolicyNetwork(nn.Module):
    """
    TODO:
    Design your own policy network.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(64, out_dim)
        self.std_layer = nn.Linear(64, out_dim)
    
    def forward(self, x):
        x = self.network(x)
        mean, std = self.mean_layer(x), self.std_layer(x)
        std = F.softplus(std) + 1e+5
        return mean, std        

class ValueNetwork(nn.Module):
    """
    TODO:
    Design your own value network.
    """
    def __init__(self, in_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train(env, M):
    """
    TODO:
    Implement the main training loop of a policy gradient method.

    You are free to design the details, but your implementation should include:
    - Network initialization: a policy network and a value network (baseline)
    - Action selection:
        * Sample actions from a stochastic policy (e.g., Gaussian distribution)
        * Use deterministic actions during evaluation
    - Interaction with the environment
    - Network update:
        * Compute policy loss to maximize expected returns
        * Compute value loss to estimate the expected return of each state
    """

    state_dim = env.observation_space.shape[0]

    # action[0]: Main(bottom) engine throttle (1.0 = maximum thrust, 0.0 = no thrust)
    # action[1]: Side(left/right) engine throttle (1.0 = maximum left thrust, 0.0 = no thrust, -1.0 = maximum right thrust)
    action_low = env.action_space.low
    action_high = env.action_space.high


    for episode in range(1, M+1):
        done = False
        while not done:
            # You can take a step for the next state and get next_state, reward, terminated_flg, and truncated_flg
            next_state, reward, terminated_flg, truncated_flg, _ = env.step(action)
            done = terminated_flg or truncated_flg
    
    env.close()


def main():
    # Create LunarLander environment
    # Action space: env.action_space.n (continuous action space)
    # State space: env.observation_space.shape
    env = gym.make("LunarLander-v3", continuous=True, render_mode='human')
    
    M = MAX_EPISODES

    train(env, M)

if __name__ == "__main__":
    main()