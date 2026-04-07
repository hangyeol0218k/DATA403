import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt

from gymnasium.wrappers import RecordVideo
from collections import deque

"""
TODO:
Train an agent that can survive over 400 time steps (~8 seconds).

Also, save:
- A video of the episode in which the agent survived for the longest number of time steps.
- A performance graph (the number of time steps survived in each episode) that shows the learning progress of your agent.
similar to the provided examples.

You may refer to the lecture notes if needed.
"""


class DQN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_features)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def select_action(eps, env, dqn, state):
    x = np.random.random()
    if x < eps:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return dqn(state).argmax().item()

def running_DQL(env, M, T, K, R, B, N, df, lr, er):
    """
    TODO:
    Implement the main loop of Deep Q-Learning.

    You are free to design the details, but your implementation should include:
    - Epsilon-greedy policy
    - Network update via gradient descent with replay memory
    - Replay memory warm-up
    - Fixed Q-targets
    """
    tar_net = DQN(in_features=4, out_features=2)
    pol_net = DQN(in_features=4, out_features=2)
    optimizer = optim.Adam(pol_net.parameters(), lr=lr)
    
    replay_memory = deque(maxlen=R)
    eps, eps_lb, eps_dec = er
    warmup_flg = 0
    total_step = 0

    episode_durations = []
    eps_history = []
    max_duration = 0
    cur_duration = 0
    best_actions = []
    best_seed = None

    for episode in range(1, M+1):
        # Reset environment with randomly generated seed
        seed = random.randint(0, 2**32 - 1)
        state, _ = env.reset(seed=seed)
        state = torch.tensor(state, dtype=torch.float32)
        action_list = []

        for t in range(0, T):
            # You can take a step for the next state and get next_state, reward, terminated_flg, and truncated_flg
            action = select_action(eps, env, pol_net, state)
            action_list.append(action)
            next_state, reward, terminated_flg, truncated_flg, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            replay_memory.append((state, action, reward, next_state, terminated_flg))
            state = next_state
            
            if warmup_flg == 0:
                warmup_flg = 0 if len(replay_memory) < B else 1
            else:
                for _ in range(0, K):
                    batch = random.sample(replay_memory, B)
                    states, actions, rewards, next_states, term_flg = zip(*batch)

                    states = torch.stack(states)
                    actions = torch.tensor(actions).unsqueeze(1)
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                    next_states = torch.stack(next_states)
                    term_flg = torch.tensor(term_flg, dtype=torch.float32)

                    y_pred = pol_net(states).gather(1, actions).squeeze()
                    with torch.no_grad():
                        y_targ = rewards + df * tar_net(next_states).max(dim=1).values * (1 - term_flg)

                    loss = nn.MSELoss()(y_pred, y_targ)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_step += 1
                if total_step % N == 0:
                    tar_net.load_state_dict(pol_net.state_dict())

            if terminated_flg or truncated_flg:
                cur_duration = t + 1
                break
        else:
            cur_duration = T
        
        # update the best episode info
        if cur_duration > max_duration:
            max_duration = cur_duration
            best_actions = action_list[:]
            best_seed = seed

        episode_durations.append(cur_duration)
        eps_history.append(eps)
        print(f"episode {episode:4d} | t = {cur_duration:4d} | eps = {eps:.3f}")

        # update epsilon
        eps = max(eps_lb, eps * eps_dec)

    print(f"max duration: {max_duration}")

    # Replay the best actions in new env with the same seed (same iniital state)
    rec_env = gym.make('CartPole-v1', max_episode_steps=500, render_mode="rgb_array")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Overwriting existing videos.*")
        rec_env = RecordVideo(rec_env, video_folder="./results", episode_trigger=lambda _: True, disable_logger=True)
    rec_env.reset(seed=best_seed)
    for action in best_actions:
        _, _, terminated_flg, truncated_flg, _ = rec_env.step(action)
        if terminated_flg or truncated_flg:
            break
    rec_env.close()
    
    # Result visualization
    episodes = range(1, len(episode_durations) + 1)
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
        # Duration step
    ax1.plot(episodes, episode_durations)
    ax1.axhline(y=400, color='r', linestyle='--', label='target (400)')
    ax1.set_ylabel("Steps")
    ax1.set_title("Episode")
    ax1.legend()

        # Epsilon (exploration rate)
    ax2.plot(episodes, eps_history)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Epsilon Decay")

    plt.tight_layout()
    plt.savefig("./results/analysis.png")
    plt.show()

def main():
    # Create CartPole environment
    # Action space: env.action_space.n (2: push cart to the left, push cart to the right)
    # State space: env.observation_space.shape (4: cart pos, cart vel, pole angle, pole ang vel)
    env = gym.make('CartPole-v1', max_episode_steps=500, render_mode=None)

    M = 200                 # Number of episodes
    T = 500                 # Max time steps for each episode
    K = 4                   # Update-to-date ratio
    R = 500                 # Capacity of the replay memory
    B = 32                  # Minibatch size
    N = 10                  # Target network update frequency
    df = 0.99               # Discount factor
    lr = 1e-4               # Learing rate of the optimzer
    er = [1, 0.01, 0.98]    # Exploration rate (for Epsilon-greedy policy)
    
    running_DQL(env, M, T, K, R, B, N, df, lr, er)
    env.close()

if __name__ == "__main__":
    main()