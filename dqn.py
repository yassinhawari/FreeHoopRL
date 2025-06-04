import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, 
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64,
                 memory_size: int = 10000,
                 target_update: int = 10,
                 device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 动作空间参数
        self.angle_bins = 20  # 角度分成20份
        self.force_bins = 10  # 力度分成10份
        self.total_actions = self.angle_bins * self.force_bins
        assert action_dim == self.total_actions, "action_dim必须等于angle_bins * force_bins"
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # DQN网络
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 经验回放
        self.memory = ReplayBuffer(memory_size)
        
        # 训练参数
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.train_step = 0
        
    def select_action(self, state: np.ndarray) -> Tuple[float, float]:
        """选择动作，返回(angle, force)元组"""
        # 使用epsilon-greedy策略选择动作索引
        if random.random() < self.epsilon:
            # 随机探索
            action_idx = random.randrange(self.total_actions)
        else:
            # 根据策略选择
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()
        
        # 将动作索引转换为角度和力度
        angle_idx = action_idx // self.force_bins  # 整除得到角度索引
        force_idx = action_idx % self.force_bins   # 取余得到力度索引
        
        # 转换为实际的角度和力度值
        angle = (angle_idx + 0.5) * (np.pi/2 / self.angle_bins)  # 加0.5使用bin的中心值
        force = (force_idx + 0.5) * (20.0 / self.force_bins)     # 加0.5使用bin的中心值
        
        return angle, force
    
    def train(self) -> bool:
        """训练一步
        
        Returns:
            bool: 是否实际进行了训练（当memory中样本不足时返回False）
        """
        if len(self.memory) < self.batch_size:
            return False
        
        # 从经验回放中采样
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # 准备批次数据
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # 计算损失并优化
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return True
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']