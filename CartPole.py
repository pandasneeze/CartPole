import gymnasium as gym
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

    # 카트 위치 기준으로 좌우 중앙 크롭
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

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extrcted screen')
plt.show()


# ──────────────────────────────────────────────
# 3. Actor 네트워크 (θ) — π_θ(a_t | s_t)
# ──────────────────────────────────────────────
class ActorNetwork(nn.Module):
    """
    Actor 전용 CNN backbone
    출력: softmax 확률 벡터 π_θ(a_t | s_t)
    """
    def __init__(self, h, w, n_actions):
        super(ActorNetwork, self).__init__()

        # Actor 전용 CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3   = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        flat_size = convw * convh * 32    

        # 정책 헤드: n_actions개의 행동 확률
        self.policy_head = nn.Linear(flat_size, n_actions)  # (input, output)

    def forward(self, x):
        """π_θ(a|s) 반환 — 확률 분포"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return F.softmax(self.policy_head(x), dim=-1)


# ──────────────────────────────────────────────
# 4. Critic Network (w) — V_w(s_t)
# ──────────────────────────────────────────────
class CriticNetwork(nn.Module):
    """
    Critic 전용 CNN backbone
    출력: 상태 가치 V_w(s) - 1개의 스칼라
    """
    def __init__(self, h, w):
        super(CriticNetwork, self).__init__()
        # Critic 전용 CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3   = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        flat_size = convw * convh * 32

        # 가치 헤드: 스칼라 V(s)
        self.value_head = nn.Linear(flat_size, 1)

    def forward(self, x):
        """V_w(s) 반환 — 스칼라"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.value_head(x) 
    

# ──────────────────────────────────────────────
# 5. 하이퍼파라미터 & 모델 초기화
# ──────────────────────────────────────────────
GAMMA = 0.99          # Discount Factor
ALPHA = 1e-4          # Actor 학습률 α
BETA = 1e-3           # Critic 학습률 β
NUM_EPISODES = 300

env.reset()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n

# 두 네트워크 생성
actor = ActorNetwork(screen_height, screen_width, n_actions).to(device)
critic = CriticNetwork(screen_height, screen_width).to(device)

# α, β의 옵티마이저(Adam Optimizer)
actor_optimizer = optim.Adam(actor.parameters(), lr=ALPHA)
critic_optimizer = optim.Adam(critic.parameters(), lr=BETA)


# ──────────────────────────────────────────────
# 6. 행동 선택 — π_θ(a_t|s_t) 에서 샘플링
# ──────────────────────────────────────────────
def select_action(state):
    """
    Actor가 출력한 확률 분포에서 행동 샘플링
    반환: action, log π_θ(a_t|s_t)
    """
    probs = actor(state)
    dist = Categorical(probs)
    action = dist.sample()
    return action, dist.log_prob(action) # 액션, log π 반환


# ──────────────────────────────────────────────
# 7. TD Advantage 계산 및 업데이트
# ──────────────────────────────────────────────
def compute_advantage(reward, state, next_state, done):
    """
    TD(0) Advantage:
    A(s_t, a_t) = R_{t+1} + γ·V_w(s_{t+1}) - V_w(s_t)

    done=True 이면 s_{t+1}이 없으므로 V(s_{t+1}) = 0
    """
    v_s = critic(state)  # V_w(s_t)

    with torch.no_grad():
        if done: # 종료 상태
            v_s_next = torch.zeros(1, 1).to(device)
        else:
            v_s_next = critic(next_state) # V_w(s_{t+1})
    
    td_target = reward + GAMMA * v_s_next # R_{t+1} + γ·V_w(s_{t+1})
    advantage = td_target - v_s

    return advantage, v_s, td_target
        
def update(log_prob, advatage, v_s, td_target):
    """
    Critic 업데이트: Δw = β · A · ∇_w V_w(s_t)
      → MSE(V_w(s_t), td_target) 최소화와 동치

    Actor  업데이트: Δθ = α · A · ∇_θ ln π_θ(a_t|s_t)
      → -log π · A 최소화와 동치
    """

    # ----- critic update -----
    critic_loss = F.mse_loss(v_s, td_target.detach())

    critic_optimizer.zero_grad()
    critic_loss.backward()       # ∇_w V_w(s_t) 계산
    critic_optimizer.step()      # β 스케일로 업데이트

    # ----- Actor update -----
    actor_loss = -log_prob * advatage.detach()

    actor_optimizer.zero_grad()
    actor_loss.backward()       # ∇_θ ln π_θ(a_t|s_t) 계산
    actor_optimizer.step()      # α 스케일로 업데이트

    return actor_loss.item(), critic_loss.item()


# ──────────────────────────────────────────────
# 8. 시각화
# ──────────────────────────────────────────────
episode_durations = []

def plot_durations():
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('A2C Training')
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
# 9. 메인 학습 루프
# ──────────────────────────────────────────────
for i_episode in range(NUM_EPISODES):
    env.reset()
    state = get_screen() - get_screen()

    for t in count():
        # --- 행동 선택 ---
        action, log_prob = select_action(state)

        # --- 환경 진행 ---
        _, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        reward = torch.tensor([[reward]], dtype=torch.float32).to(device)

        # --- 다음 상태 ---
        next_state = (get_screen() - get_screen()) if not done else None

        # --- TD Advantage 계산 ---
        # A(s_t, a_t) = R_{t+1} + γ·V_w(s_{t+1}) - V_w(s_t)
        advantage, v_s, td_target = compute_advantage(reward, state, next_state, done)

        # --- 매 스텝 업데이트 ---
        actor_loss, critic_loss = update(log_prob, advantage, v_s, td_target)

        state = next_state

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
env.close()
plt.ioff()
plt.show()