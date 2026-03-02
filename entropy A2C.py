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
plt.title('Example extracted screen')
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
        self.policy_head = nn.Linear(flat_size, n_actions)

    def forward(self, x):
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
        self.value_head = nn.Linear(flat_size, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.value_head(x)


# ──────────────────────────────────────────────
# 5. 하이퍼파라미터 & 모델 초기화
# ──────────────────────────────────────────────
GAMMA         = 0.999   # Discount Factor
ALPHA         = 1e-4    # Actor 학습률 α
BETA          = 1e-3    # Critic 학습률 β
ENTROPY_COEF  = 0.01    # [추가] 엔트로피 보너스 계수 β_H
                        #        클수록 탐험 강화 / 작을수록 수렴 안정
NUM_EPISODES  = 300

# [추가] Advantage 정규화를 위한 running statistics (지수이동평균)
# 단일 스텝 업데이트에서는 배치 정규화 대신 EMA로 분산을 추정
ADV_EMA_MEAN  = 0.0
ADV_EMA_VAR   = 1.0
ADV_EMA_ALPHA = 0.01    # EMA 업데이트 속도 (작을수록 느리게 적응)

env.reset()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n

actor  = ActorNetwork(screen_height, screen_width, n_actions).to(device)
critic = CriticNetwork(screen_height, screen_width).to(device)

actor_optimizer  = optim.RMSprop(actor.parameters(),  lr=ALPHA)
critic_optimizer = optim.RMSprop(critic.parameters(), lr=BETA)


# ──────────────────────────────────────────────
# 6. 행동 선택 — π_θ(a_t|s_t) 에서 샘플링
# ──────────────────────────────────────────────
def select_action(state):
    """
    Actor 확률 분포에서 행동 샘플링
    반환: action, log π_θ(a_t|s_t), H(π) 엔트로피
    """
    probs  = actor(state)
    dist   = Categorical(probs)
    action = dist.sample()
    # [추가] 엔트로피도 함께 반환 — update()에서 손실에 반영
    return action, dist.log_prob(action), dist.entropy()


# ──────────────────────────────────────────────
# 7. TD Advantage 계산
# ──────────────────────────────────────────────
def compute_advantage(reward, state, next_state, done):
    """
    TD(0) Advantage:
      A(s_t, a_t) = R_{t+1} + γ·V_w(s_{t+1}) - V_w(s_t)

    done=True 이면 V(s_{t+1}) = 0
    """
    v_s = critic(state)

    with torch.no_grad():
        v_s_next = torch.zeros(1, 1, device=device) if done \
                   else critic(next_state)

    td_target = reward + GAMMA * v_s_next
    advantage  = td_target - v_s

    return advantage, v_s, td_target


# ──────────────────────────────────────────────
# 8. [추가] Advantage EMA 정규화
# ──────────────────────────────────────────────
def normalize_advantage(advantage):
    """
    단일 스텝 advantage를 지수이동평균(EMA) 기반으로 정규화.

    배치 정규화는 여러 샘플이 있어야 분산 추정이 가능하지만,
    stochastic 업데이트는 매 스텝 advantage가 1개뿐이므로
    EMA로 전역 평균·분산을 추적하여 정규화.

      A_norm = (A - μ_EMA) / (σ_EMA + ε)
    """
    global ADV_EMA_MEAN, ADV_EMA_VAR

    a = advantage.item()

    # EMA 평균 업데이트
    ADV_EMA_MEAN = (1 - ADV_EMA_ALPHA) * ADV_EMA_MEAN + ADV_EMA_ALPHA * a

    # EMA 분산 업데이트 (Welford 근사)
    ADV_EMA_VAR  = (1 - ADV_EMA_ALPHA) * ADV_EMA_VAR  \
                 + ADV_EMA_ALPHA * (a - ADV_EMA_MEAN) ** 2

    adv_norm = (advantage - ADV_EMA_MEAN) / (ADV_EMA_VAR ** 0.5 + 1e-8)
    return adv_norm


# ──────────────────────────────────────────────
# 9. 업데이트 (stochastic, 매 스텝)
# ──────────────────────────────────────────────
def update(log_prob, entropy, advantage, v_s, td_target):
    """
    Critic: MSE(V_w(s_t), td_target) 최소화
    Actor : -log π · A_norm - β_H · H(π) 최소화
              └ policy gradient  └ 엔트로피 보너스 (탐험 장려)
    """
    # [추가] Advantage EMA 정규화
    advantage_norm = normalize_advantage(advantage.detach())

    # ── Critic 업데이트 ──
    critic_loss = F.mse_loss(v_s, td_target.detach())
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # ── Actor 업데이트 ──
    # [추가] 엔트로피 보너스: -β_H · H(π) → 엔트로피 최대화 유도
    actor_loss = -log_prob * advantage_norm - ENTROPY_COEF * entropy
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    return actor_loss.item(), critic_loss.item(), entropy.item()


# ──────────────────────────────────────────────
# 10. 시각화
# ──────────────────────────────────────────────
episode_durations = []
entropy_history   = []  # [추가] 에피소드별 평균 엔트로피

def plot_durations():
    fig = plt.figure(1, figsize=(11, 4))
    plt.clf()

    # ── 왼쪽: Duration ──
    ax1 = fig.add_subplot(1, 2, 1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    ax1.set_title('A2C Training (+ Entropy Bonus)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Duration')
    ax1.plot(durations_t.numpy(), alpha=0.4, label='Duration')
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        ax1.plot(means.numpy(), label='100-ep avg')
    ax1.legend()

    # ── 오른쪽: Entropy — 탐험 정도 모니터링 ──
    # 엔트로피가 너무 빨리 감소 → ENTROPY_COEF 증가 고려
    # 엔트로피가 유지되는데 Duration이 낮음 → ENTROPY_COEF 감소 고려
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Policy Entropy  H(π)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Entropy')
    if entropy_history:
        ax2.plot(entropy_history, alpha=0.6, color='green', label='Entropy')
        max_entropy = np.log(n_actions)   # 균등분포일 때 최대 엔트로피
        ax2.axhline(y=max_entropy, color='red', linestyle='--',
                    label=f'Max H = ln({n_actions}) ≈ {max_entropy:.2f}')
        ax2.legend()

    plt.tight_layout()
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# ──────────────────────────────────────────────
# 11. 메인 학습 루프
# ──────────────────────────────────────────────
for i_episode in range(NUM_EPISODES):
    env.reset()
    state = get_screen() - get_screen()
    episode_entropy = []

    for t in count():
        # --- 행동 선택 (엔트로피 포함) ---
        action, log_prob, entropy = select_action(state)

        # --- 환경 진행 ---
        _, reward, terminated, truncated, _ = env.step(action.item())
        done   = terminated or truncated
        reward = torch.tensor([[reward]], dtype=torch.float32, device=device)

        # --- 다음 상태 ---
        next_state = (get_screen() - get_screen()) if not done else None

        # --- TD Advantage 계산 ---
        advantage, v_s, td_target = compute_advantage(reward, state, next_state, done)

        # --- 매 스텝 업데이트 (stochastic) ---
        _, _, ent_val = update(log_prob, entropy, advantage, v_s, td_target)
        episode_entropy.append(ent_val)

        state = next_state

        if done:
            episode_durations.append(t + 1)
            entropy_history.append(np.mean(episode_entropy))
            plot_durations()
            break

print('Complete')
env.close()
plt.ioff()
plt.show()