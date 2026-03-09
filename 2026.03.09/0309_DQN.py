import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)  # Adam Optimizer
replay_buffer = ReplayBuffer(10000)

batch_size = 64
gamma = 0.99     # discount factor
epsilon = 1.0    

reward_list = []

for episode in range(1000):  # episodes
    state = env.reset()
    if isinstance(state, tuple): state = state[0] 
    
    state = torch.FloatTensor(state).to(device)
    total_reward = 0

    for t in range(500):
        if np.random.rand() < epsilon:  # epsilon greedy
            action = np.random.randint(action_dim)
        else:
            with torch.no_grad():
                action = policy_net(state).argmax().item()

        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        
        next_state = torch.FloatTensor(next_state).to(device)
        replay_buffer.push((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        if len(replay_buffer) >= batch_size:
            transitions = replay_buffer.sample(batch_size)
            b_state, b_action, b_reward, b_next_state, b_done = zip(*transitions)

            b_state = torch.stack(b_state).to(device)
            b_action = torch.LongTensor(b_action).unsqueeze(1).to(device)
            b_reward = torch.FloatTensor(b_reward).unsqueeze(1).to(device)
            b_next_state = torch.stack(b_next_state).to(device)
            b_done = torch.FloatTensor(b_done).unsqueeze(1).to(device)

            q_values = policy_net(b_state).gather(1, b_action)
            next_q_values = target_net(b_next_state).max(1)[0].detach().unsqueeze(1)
            target = b_reward + gamma * next_q_values * (1 - b_done)

            loss = F.mse_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(0.01, epsilon * 0.995)

    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    reward_list.append(total_reward)

env.close()


# ------ 그래프 ------
plt.plot(reward_list, alpha=0.4, color='steelblue', label='Episode Reward')

# 100 에피소드 이상부터 이동 평균 추세선 추가
window = 100
if len(reward_list) >= window:
    moving_avg = [
        np.mean(reward_list[i - window:i])
        for i in range(window, len(reward_list) + 1)
    ]
    plt.plot(
        range(window - 1, len(reward_list)),
        moving_avg,
        color='crimson',
        linewidth=2,
        label=f'{window}-Episode Moving Average'
    )

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training - CartPole-v1')
plt.legend()
plt.tight_layout()
plt.show()