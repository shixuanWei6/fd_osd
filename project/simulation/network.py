# fileName: simulation/network.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class NetworkSimulator:
    """Simulates fluctuating mobile network conditions using a Markov Chain."""
    def __init__(self):
        self.states = ['WIFI', '4G', 'WEAK']
        self.current_state = 'WIFI'
        
        # Probability of transitioning from State A to State B
        self.transition_matrix = {
            'WIFI': {'WIFI': 0.80, '4G': 0.15, 'WEAK': 0.05},
            '4G':   {'WIFI': 0.20, '4G': 0.70, 'WEAK': 0.10},
            'WEAK': {'WIFI': 0.10, '4G': 0.30, 'WEAK': 0.60}
        }
        
        # (mean_RTT_ms, std_dev_ms)
        self.rtt_profiles = {
            'WIFI': (20.0, 5.0),
            '4G':   (80.0, 20.0),
            'WEAK': (300.0, 100.0)
        }

    def get_rtt(self):
        """Steps the simulation and returns the current Round Trip Time (RTT)."""
        probs = list(self.transition_matrix[self.current_state].values())
        self.current_state = np.random.choice(self.states, p=probs)
        mean, std = self.rtt_profiles[self.current_state]
        rtt = max(5.0, np.random.normal(mean, std))
        return rtt

class RLHorizonController:
    """A lightweight DQN Agent to dynamically control the speculative step size (K)."""
    def __init__(self, state_dim=3, action_choices=[2, 4, 6, 8], lr=1e-3, gamma=0.99, epsilon=0.15):
        self.action_choices = action_choices
        self.n_actions = len(action_choices)
        self.gamma = gamma       # Discount factor
        self.epsilon = epsilon   # Exploration rate
        
        # Lightweight MLP for edge inference
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, self.n_actions)
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def select_action(self, state, explore=True):
        """Selects K based on epsilon-greedy policy."""
        if explore and random.random() < self.epsilon:
            idx = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                idx = torch.argmax(q_values).item()
        return self.action_choices[idx], idx
        
    def update(self, state, action_idx, reward, next_state):
        """执行单次 Q-learning 更新步骤。"""
        # [修复] 强制启用梯度以覆盖外层的 no_grad 上下文
        with torch.enable_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            
            q_values = self.q_network(state_tensor)
            current_q = q_values[0, action_idx]
            
            with torch.no_grad():
                next_q_values = self.q_network(next_state_tensor)
                max_next_q = torch.max(next_q_values)
                target_q = reward_tensor + self.gamma * max_next_q
                
            loss = self.criterion(current_q, target_q[0])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return loss.item()