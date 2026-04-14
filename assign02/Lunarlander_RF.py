import glob
import os
import random
import warnings
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
        std = F.softplus(std) + 1e-5
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

def select_action(pol_net, state, stochastic = True):
    state = torch.tensor(state, dtype=torch.float32)
    mean, std = pol_net(state)
    if stochastic:
        dist = Normal(mean, std)
        u = dist.sample()
        a = torch.tanh(u)   # squashing to [-1, 1]

        log_prob = dist.log_prob(u).sum()   # Assume that 2 action spaces are independent
        log_prob -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum()
        return log_prob, a
    else:
        return None, torch.tanh(mean)

def train(env, hyperparams):
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
    M, pol_lr, val_lr, gamma = hyperparams

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # action[0]: Main(bottom) engine throttle (1.0 = maximum thrust, 0.0 = no thrust)
    # action[1]: Side(left/right) engine throttle (1.0 = maximum left thrust, 0.0 = no thrust, -1.0 = maximum right thrust)
    action_low = env.action_space.low
    action_high = env.action_space.high

    pol_net = PolicyNetwork(state_dim, action_dim)
    val_net = ValueNetwork(state_dim)
    pol_optim = optim.Adam(pol_net.parameters(), lr=pol_lr)
    val_optim = optim.Adam(val_net.parameters(), lr=val_lr)

    episode_rewards = []
    pol_losses = []
    val_losses = []

    for episode in range(1, M+1):
        log_probs = []
        rewards = []
        cumul_rewards = []
        baseline = []
        advantages = []

        state, _ = env.reset()
        done = False
        # Collect a trajectory
        while not done:
            # Select a stocahastic action and obtain its log probability
            log_prob, action = select_action(pol_net=pol_net, state=state, stochastic=True)
            log_probs.append(log_prob)
            action = action.detach().numpy()

            # You can take a step for the next state and get next_state, reward, terminated_flg, and truncated_flg
            next_state, reward, terminated_flg, truncated_flg, _ = env.step(action)
            rewards.append(torch.tensor(reward, dtype=torch.float32))

            # Get a baseline value
            b = val_net(torch.tensor(state, dtype=torch.float32))
            baseline.append(b)

            # Progress one time step
            state = next_state
            done = terminated_flg or truncated_flg

        # Compute the cumulative rewards and advantage values
        temp = 0
        for r in rewards[::-1]:
            temp = r + gamma * temp
            cumul_rewards.append(temp)
        cumul_rewards.reverse()

        # Stack all trajectory data into tensors
        baseline      = torch.stack(baseline).squeeze()
        rewards       = torch.stack(rewards).squeeze()
        cumul_rewards = torch.stack(cumul_rewards).squeeze()
        log_probs     = torch.stack(log_probs).squeeze()

        # Compute and normalize advantages
        advantages = (cumul_rewards.detach() - baseline.detach())
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update value network(baseline) parameters
        val_loss = nn.MSELoss()(baseline, cumul_rewards)
        val_optim.zero_grad()
        val_loss.backward()
        val_optim.step()

        # Update policy network parameters
        pol_loss = -(log_probs * advantages).mean()
        pol_optim.zero_grad()
        pol_loss.backward()
        pol_optim.step()

        total_reward = rewards.sum().item()
        episode_rewards.append(total_reward)
        pol_losses.append(pol_loss.item())
        val_losses.append(val_loss.item())
        print(f"Episode {episode:4d} | Total Reward: {total_reward:8.2f} | Pol Loss: {pol_loss.item():8.4f} | Val Loss: {val_loss.item():8.4f}")

    env.close()

    record_best_videos(pol_net, n_eval=30, n_keep=5)
    plot_results(episode_rewards, pol_losses, val_losses, M)


def record_best_videos(pol_net, n_eval=30, n_keep=5, video_folder="./assign02/results"):
    print("\nRecording evaluation episodes...")
    os.makedirs(video_folder, exist_ok=True)
    rec_env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Overwriting existing videos.*")
        rec_env = RecordVideo(rec_env, video_folder=video_folder, episode_trigger=lambda _: True, disable_logger=True)

    eval_rewards = []
    for _ in range(n_eval):
        state, _ = rec_env.reset()
        done = False
        total = 0
        while not done:
            _, action = select_action(pol_net, state, stochastic=False)
            state, reward, terminated_flg, truncated_flg, _ = rec_env.step(action.detach().numpy())
            total += reward
            done = terminated_flg or truncated_flg
        eval_rewards.append(total)
    rec_env.close()

    # Delete all videos except the best n_keep
    best_indices = set(np.argsort(eval_rewards)[-n_keep:].tolist())
    print(f"\nTop {n_keep} episodes kept:")
    for idx in sorted(best_indices, key=lambda i: eval_rewards[i], reverse=True):
        print(f"  Episode {idx}: reward = {eval_rewards[idx]:.2f}")
    video_files = sorted(glob.glob(os.path.join(video_folder, "rl-video-episode-*.mp4")))
    for i, f in enumerate(video_files):
        if i not in best_indices:
            os.remove(f)
    for f in glob.glob(os.path.join(video_folder, "rl-video-episode-*.meta.json")):
        os.remove(f)
    print(f"Saved top {n_keep} videos to {video_folder}/")


def plot_results(episode_rewards, pol_losses, val_losses, M, out_dir="./assign02/results"):
    window = 50
    moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')

    _, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].set_title("REINFORCE on LunarLander-v3 (Continuous)")
    axes[0].plot(episode_rewards, alpha=0.4, color='steelblue', label='Episode Reward')
    axes[0].plot(range(window - 1, M), moving_avg, color='orange', label=f'Moving Avg Reward (window={window})')
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].set_title("Policy Loss")
    axes[1].plot(pol_losses, color='steelblue', label='Policy Loss')
    axes[1].set_ylabel("Value")
    axes[1].legend()

    axes[2].set_title("Value Loss (separate scale)")
    axes[2].plot(val_losses, color='orange', label='Value Loss')
    axes[2].set_ylabel("Value Loss")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Episode")
        ax.grid(True, linestyle='--', alpha=0.5)

    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "analysis.png"))
    plt.show()


def main():
    # Create LunarLander environment
    # Action space: env.action_space.n (continuous action space)
    # State space: env.observation_space.shape
    env = gym.make("LunarLander-v3", continuous=True, render_mode=None)

    M = 3000
    pol_lr = 1e-3
    val_lr = 1e-3
    gamma = 0.99
    hyperparams = (M, pol_lr, val_lr, gamma)

    train(env, hyperparams)

if __name__ == "__main__":
    main()
