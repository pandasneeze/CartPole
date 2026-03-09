import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# ===== hyperparameter =====
learning_rate = 0.0002
gamma         = 0.99   # discount factor
n_rollout     = 10     # batch size
entropy_coef  = 0.01   

# ===== A2C =====
class ActorCritic(nn.Module):  # DNN(MLP)
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 256)   # input: 카트의 위치, 속도, 막대의 각도, 각속도
        self.fc_pi = nn.Linear(256, 2) # actor output: 카트를 왼쪽 또는 오른쪽으로 밀 확률
        self.fc_v = nn.Linear(256, 1)  # critic output: state value
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate) # optimizer
    
    # actor network
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)  # 행동 확률
        return prob

    # critic network   
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        s_batch = torch.tensor(np.array(s_lst), dtype=torch.float)
        a_batch = torch.tensor(a_lst)
        r_batch = torch.tensor(r_lst, dtype=torch.float)
        s_prime_batch = torch.tensor(np.array(s_prime_lst), dtype=torch.float)
        done_batch = torch.tensor(done_lst, dtype=torch.float)
        
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch # s, a, r, s', done
    
    # train network
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        
        # Advantage 계산
        td_target = r + gamma * self.v(s_prime) * done
        advantage = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        
        entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=1, keepdim=True)

        # feedback
        # Loss = L_actor + L_critic - βH(π)
        loss = -torch.log(pi_a) * advantage.detach() + \
               F.smooth_l1_loss(self.v(s), td_target.detach()) - \
               entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        return entropy.mean().item()

def main():  
    env = gym.make('CartPole-v1')
    model = ActorCritic()    
    print_interval = 20
    score = 0.0
    
    score_history = []
    entropy_history = [] 

    for n_epi in range(1000): # episodes
        done = False
        s, info = env.reset()
        epi_score = 0.0
        epi_entropy = []
        
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                
                s_prime, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                
                model.put_data((s, a, r, s_prime, done))
                
                s = s_prime
                score += r
                epi_score += r
                
                if done:
                    break                     
            
            if len(model.data) > 0:
                ent = model.train_net()
                epi_entropy.append(ent)
        
        score_history.append(epi_score)
        if epi_entropy:
            entropy_history.append(np.mean(epi_entropy))

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0
            
    env.close()

    # ------ 그래프 ------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Score ──
    ax1.plot(score_history, alpha=0.4, color='steelblue', label='Total Reward')
    window = 100
    if len(score_history) >= window:
        moving_avg = [
            np.mean(score_history[i - window:i])
            for i in range(window, len(score_history) + 1)
        ]
        ax1.plot(range(window - 1, len(score_history)),
                 moving_avg, color='crimson', linewidth=2,
                 label=f'{window}-Episode Moving Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('A2C (with Entropy) CartPole-v1 Training')
    ax1.legend()

    # ── Entropy ──
    ax2.plot(entropy_history, alpha=0.4, color='green', label='Entropy')
    
    window = 100
    if len(entropy_history) >= window:
        ent_moving_avg = [
            np.mean(entropy_history[i - window:i])
            for i in range(window, len(entropy_history) + 1)
        ]
        ax2.plot(range(window - 1, len(entropy_history)),
                 ent_moving_avg, color='orange', linewidth=2,
                 label=f'{window}-Episode Moving Average')
    
    max_entropy = np.log(2)
    ax2.axhline(y=max_entropy, color='red', linestyle='--',
                label=f'Max H = ln(2) ≈ {max_entropy:.3f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Policy Entropy  H(π)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()