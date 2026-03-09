# ────────────────────────────────────────────
# A2C Rollout Size Comparison
# ────────────────────────────────────────────

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# ===== Hyperparamter =====
learning_rate = 0.0002
gamma         = 0.99              # discount factor
entropy_coef  = 0.01                
N_EPISODES    = 1000
ROLLOUT_LIST  = [5, 10, 32, 64]   # 비교할 rollout 크기

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        self.fc1   = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v  = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc_pi(x), dim=softmax_dim)

    def v(self, x):
        x = F.relu(self.fc1(x))
        return self.fc_v(x)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for s, a, r, s_prime, done in self.data:
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_lst.append([0.0 if done else 1.0])
        self.data = []
        return (
            torch.tensor(np.array(s_lst),       dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst,                  dtype=torch.float),
            torch.tensor(np.array(s_prime_lst), dtype=torch.float),
            torch.tensor(done_lst,               dtype=torch.float),
        )

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        advantage  = td_target - self.v(s)
        pi   = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=1, keepdim=True)
        loss = -torch.log(pi_a) * advantage.detach() + \
               F.smooth_l1_loss(self.v(s), td_target.detach()) - \
               entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        return entropy.mean().item()


def run_experiment(n_rollout):
    """단일 rollout 크기로 학습 후 (score_history, entropy_history) 반환"""
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    score_history, entropy_history = [], []

    for n_epi in range(N_EPISODES):
        done = False
        s, _ = env.reset()
        epi_score, epi_entropy = 0.0, []

        while not done:
            for _ in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                a    = Categorical(prob).sample().item()
                s_prime, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                model.put_data((s, a, r, s_prime, done))
                s          = s_prime
                epi_score += r
                if done:
                    break

            if model.data:
                epi_entropy.append(model.train_net())

        score_history.append(epi_score)
        if epi_entropy:
            entropy_history.append(np.mean(epi_entropy))

    env.close()
    return score_history, entropy_history


def moving_average(data, window=100):
    if len(data) < window:
        return [], []
    avg = [np.mean(data[i - window:i]) for i in range(window, len(data) + 1)]
    return list(range(window - 1, len(data))), avg


# ────────────────────────────────────────────
# 실험 실행
# ────────────────────────────────────────────
colors = ['steelblue', 'darkorange', 'green', 'mediumpurple']
results = {}

for rollout in ROLLOUT_LIST:
    print(f"▶ Training with n_rollout={rollout} ...")
    results[rollout] = run_experiment(rollout)

# ────────────────────────────────────────────
# 그래프
# ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for i, rollout in enumerate(ROLLOUT_LIST):
    score_h, ent_h = results[rollout]
    c = colors[i]
    x_avg, avg = moving_average(score_h)
    x_ea,  ea  = moving_average(ent_h)

    # alpha: 원본 투명도, 추세선만 레전드 표시
    ax1.plot(score_h, alpha=0, color=c)
    ax1.plot(x_avg, avg, color=c, linewidth=2, label=f'rollout={rollout}')

    ax2.plot(ent_h, alpha=0, color=c)
    ax2.plot(x_ea, ea, color=c, linewidth=2, label=f'rollout={rollout}')

ax1.set_title('Score (100-ep Moving Average)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Score')
ax1.legend()

ax2.set_title('Entropy H(π) (100-ep Moving Average)')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Entropy')
ax2.axhline(y=np.log(2), color='red', linestyle='--',
            linewidth=1, label='Max H = ln(2)')
ax2.legend()

plt.suptitle('A2C — n_rollout Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()