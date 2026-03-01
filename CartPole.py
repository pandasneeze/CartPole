import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# ──────────────────────────────────────────────
# 1. 환경 및 디바이스 설정
# ──────────────────────────────────────────────
env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


# ──────────────────────────────────────────────
# 2. 화면 전처리
# ──────────────────────────────────────────────
resize = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=InterpolationMode.BICUBIC),
    T.ToTensor()
])

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)

def get_screen():
    screen = env.render().transpose((2, 0, 1))
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)


# ──────────────────────────────────────────────
# 3. Actor-Critic 네트워크
# ──────────────────────────────────────────────
class ActorCritic(nn.Module):
    """
       actor_head  : 정책
       critic_head : 상태 가치 V(s)
    """
    def __init__(self, h, w, n_actions):
        super(ActorCritic, self).__init__()

        # CNN backbone
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # output 크기 계산 헬퍼 함수
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        flat_size = convw * convh * 32

        # Actor 헤드: π_θ(a|s)
        self.actor_head = nn.Linear(flat_size, n_actions) # (input, output)

        # Critic 헤드: 상태 가치 V(s)
        self.critic_head = nn.Linear(flat_size, 1) # output은 V(s) 단일 스칼라
    
    def _backbone(self, x):  # 3 layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.view(x.size(0), -1)   # DQN과 다르게 head에 바로 넣지 않음
                                       # Actor, Critic head 각각 따로기 때문

    def forward(self, x):
        """
        학습 시 사용
        output: (행동 확률 π_θ(a|s), 상태 가치 V(s))
        """
        features = self._backbone(x)
        action_probs = F.softmax(self.actor_head(features), dim=-1)
        state_value = self.critic_head(features)
        return action_probs, state_value
    
    def get_action(self, x):
        """
        추론 시 사용
        확률 분포에서 행동을 샘플링하고 log_prob(손실 계산에 사용)도 함께 반환
        """
        features = self._backbone(x)
        action_probs = F.softmax(self.actor_head(features), dim=-1)
        dist = Categorical(action_probs)   # 분포
        action = dist.sample()             # 행동 선택
        return action, dist.log_prob(action)
    

# ──────────────────────────────────────────────
# 4. 하이퍼파라미터 & 모델 초기화
# ──────────────────────────────────────────────
GAMMA = 0.99          # Discount Factor
LR = 1e-3             # Learning Rate
NUM_EPISODES = 300    # 에피소드 수
ENTROPY_COEF = 0.01   # 엔트로피 보너스 계수
VALUE_COEF = 0.5      # Critic loss 가중치

env.reset()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n

model = ActorCritic(screen_height, screen_width, n_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR) # Adam optimaizer 사용


# ──────────────────────────────────────────────
# 5. 에피소드 수익 계산 (Discounted Return)
# ──────────────────────────────────────────────
def compute_returns(rewards, gamma=GAMMA):
    """
    G_t 계산
    G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    """
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    # 정규화
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# ──────────────────────────────────────────────
# 6. 손실(loss) 계산 및 업데이트
# ──────────────────────────────────────────────
def update(log_probs, values, rewards, entropies):
    """
    Actor-Critic 통합 손실:
      - Actor Loss: -log π(a|s) × Advantage
      - Critic Loss: MSE(V(s), G_t) (*note: MSE = Mean Squared Error)
      - Entropy: -H(π)
    """
    returns = compute_returns(rewards)
    log_probs = torch.stack(log_probs)
    values = torch.stack(values).squeeze()
    entropies = torch.stack(entropies)

    # Advantage = Q - V
    advantage = returns - values.detach()

    actor_loss = -(log_probs * advantage).mean()
    critic_loss = F.mse_loss(values, returns)
    entropy_loss = -entropies.mean()

    total_loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()

    return total_loss.item()


# ──────────────────────────────────────────────
# 7. 시각화
# ──────────────────────────────────────────────
episode_durations = []

def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Actor-Critic Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy(), alpha=0.4, label='Duration')
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100-ep avg')
    plt.legend()
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# ──────────────────────────────────────────────
# 8. 메인 학습 루프
# ──────────────────────────────────────────────
for i_episode in range(NUM_EPISODES):
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    # 에피소드 단위로 경험 수집(DQN의 Replay Memory와 다른 점)
    log_probs = []
    values = []
    rewards = []
    entropies = []

    for t in count():
        # 1. 행동 선택
        actions_probs, state_value = model(state)
        dist = Categorical(actions_probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))
        values.append(state_value)
        entropies.append(dist.entropy())

        # 2. 환경 진행
        _, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        rewards.append(reward)

        # 3. 다음 상태
        last_screen = current_screen
        current_screen = get_screen()
        # 끝나지 않은 경우에만 state 업데이트
        state = (current_screen - last_screen) if not done else None

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

    # 4. 에피소드 종료 후 한 번에 업데이트
    update(log_probs, values, rewards, entropies)


print('Complete')
env.close()
plt.ioff()
plt.show()