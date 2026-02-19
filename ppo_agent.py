"""
PPO (Proximal Policy Optimization) 智能体
用于彩色五子棋游戏
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random


class BoardEncoder(nn.Module):
    """棋盘编码器 - 使用CNN提取棋盘特征"""
    
    def __init__(self, board_size: int = 9, num_colors: int = 7, hidden_dim: int = 256):
        super().__init__()
        
        self.board_size = board_size
        self.num_colors = num_colors
        
        # 将棋盘转换为one-hot编码: (batch, num_colors+1, board_size, board_size)
        # +1是因为有空格
        
        # CNN特征提取
        self.conv = nn.Sequential(
            nn.Conv2d(num_colors + 1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 计算展平后的维度
        self.flat_dim = 128 * board_size * board_size
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """
        输入: board (batch, board_size, board_size)，值为-1到num_colors-1
        输出: 特征向量 (batch, hidden_dim)
        """
        batch_size = board.shape[0]
        
        # 将-1（空格）映射到num_colors
        board_shifted = board.clone()
        board_shifted[board_shifted == -1] = self.num_colors
        
        # One-hot编码
        board_onehot = nn.functional.one_hot(
            board_shifted.long(), 
            num_classes=self.num_colors + 1
        ).float()
        
        # 调整维度: (batch, board_size, board_size, num_colors+1) -> (batch, num_colors+1, board_size, board_size)
        board_onehot = board_onehot.permute(0, 3, 1, 2)
        
        # CNN特征提取
        conv_out = self.conv(board_onehot)
        
        # 展平
        flat = conv_out.reshape(batch_size, -1)
        
        # 全连接
        features = self.fc(flat)
        
        return features


class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, board_size: int = 9, num_colors: int = 7, hidden_dim: int = 256):
        super().__init__()
        
        self.board_size = board_size
        self.num_colors = num_colors
        self.action_dim = board_size ** 4  # 所有可能的动作数
        
        # 棋盘编码器
        self.board_encoder = BoardEncoder(board_size, num_colors, hidden_dim)
        
        # 即将出现的棋子编码器
        self.upcoming_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        
        # 合并特征
        self.combined_dim = hidden_dim + 64
        
        # Actor (策略网络) - 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        
        # Critic (价值网络) - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # 温度参数（用于控制探索程度，>1 增加探索，<1 减少探索）
        self.temperature = 1.0
    
    def forward(self, board: torch.Tensor, upcoming: torch.Tensor, 
                valid_actions_mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            board: (batch, board_size, board_size)
            upcoming: (batch, 3)
            valid_actions_mask: (batch, action_dim) - 有效动作的掩码
            temperature: 温度参数，>1 增加探索，<1 减少探索
        
        Returns:
            action_probs: (batch, action_dim)
            value: (batch, 1)
        """
        batch_size = board.shape[0]
        
        # 编码棋盘
        board_features = self.board_encoder(board)
        
        # 编码即将出现的棋子
        upcoming_features = self.upcoming_encoder(upcoming.float())
        
        # 合并特征
        combined = torch.cat([board_features, upcoming_features], dim=1)
        
        # Actor: 计算动作logits
        action_logits = self.actor(combined)
        
        # 应用温度参数（温度 > 1 会让概率分布更平缓，增加探索）
        if temperature != 1.0:
            action_logits = action_logits / temperature
        
        # 应用有效动作掩码
        if valid_actions_mask is not None:
            # 将无效动作的logits设为很大的负数
            action_logits = action_logits.masked_fill(~valid_actions_mask, float('-inf'))
        
        # Softmax得到概率
        action_probs = nn.functional.softmax(action_logits, dim=1)
        
        # Critic: 计算状态价值
        value = self.critic(combined)
        
        return action_probs, value
    
    def get_action(self, board: torch.Tensor, upcoming: torch.Tensor,
                   valid_actions: List[int], deterministic: bool = False,
                   temperature: float = 1.0, epsilon: float = 0.0) -> Tuple[int, float, float]:
        """
        获取动作
        
        Args:
            board: 棋盘状态
            upcoming: 即将出现的棋子
            valid_actions: 有效动作列表
            deterministic: 是否确定性选择（贪婪）
            temperature: 温度参数，>1 增加探索
            epsilon: ε-贪婪参数，>0 时以 epsilon 概率随机选择
        
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        with torch.no_grad():
            board_tensor = torch.FloatTensor(board).unsqueeze(0)
            upcoming_tensor = torch.FloatTensor(upcoming).unsqueeze(0)
            
            # 创建有效动作掩码
            mask = torch.zeros(1, self.action_dim, dtype=torch.bool)
            if len(valid_actions) > 0:
                mask[0, valid_actions] = True
            else:
                # 如果没有有效动作，允许所有动作（但会收到惩罚）
                mask[0, :] = True
            
            action_probs, value = self.forward(board_tensor, upcoming_tensor, mask, temperature)
            
            # ε-贪婪探索：以 epsilon 概率从有效动作中随机选择
            if not deterministic and epsilon > 0 and len(valid_actions) > 0:
                if np.random.random() < epsilon:
                    action = np.random.choice(valid_actions)
                    log_prob = torch.log(action_probs[0, action] + 1e-10).item()
                    return action, log_prob, value.item()
            
            if deterministic:
                action = torch.argmax(action_probs, dim=1).item()
            else:
                action = torch.multinomial(action_probs, 1).item()
            
            log_prob = torch.log(action_probs[0, action] + 1e-10).item()
            
            return action, log_prob, value.item()


class PPOAgent:
    """PPO智能体"""
    
    def __init__(
        self,
        board_size: int = 9,
        num_colors: int = 7,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.1,  # 提高默认熵系数，从 0.01 改为 0.1
        max_grad_norm: float = 0.5,
        device: str = 'auto',
        temperature: float = 2.0,  # 默认温度参数，>1 增加探索
        epsilon_start: float = 0.3,  # ε-贪婪初始值
        epsilon_end: float = 0.01,  # ε-贪婪最终值
        epsilon_decay: float = 0.995,  # ε-贪婪衰减率
        entropy_decay: float = 0.999,  # 熵系数衰减
        min_entropy_coef: float = 0.01,  # 最小熵系数
    ):
        """
        初始化PPO智能体
        
        Args:
            board_size: 棋盘大小
            num_colors: 颜色数量
            hidden_dim: 隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda
            clip_epsilon: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵奖励系数
            max_grad_norm: 梯度裁剪范数
            device: 计算设备
            temperature: 温度参数，>1 增加探索
            epsilon_start: ε-贪婪初始值
            epsilon_end: ε-贪婪最终值
            epsilon_decay: ε-贪婪衰减率
            entropy_decay: 熵系数衰减率
            min_entropy_coef: 最小熵系数
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.board_size = board_size
        self.num_colors = num_colors
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 探索相关参数
        self.temperature = temperature
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.entropy_decay = entropy_decay
        self.min_entropy_coef = min_entropy_coef
        
        # 创建网络
        self.network = ActorCritic(board_size, num_colors, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 存储经验
        self.states_board = []
        self.states_upcoming = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.valid_actions_list = []
    
    def select_action(self, obs: Dict, valid_actions: List[int], 
                      deterministic: bool = False) -> Tuple[int, float, float]:
        """选择动作"""
        board = obs['board']
        upcoming = obs['upcoming']
        
        action, log_prob, value = self.network.get_action(
            board, upcoming, valid_actions, deterministic,
            temperature=self.temperature if not deterministic else 1.0,
            epsilon=self.epsilon if not deterministic else 0.0
        )
        
        return action, log_prob, value
    
    def store_transition(self, obs: Dict, action: int, log_prob: float, 
                         reward: float, value: float, done: bool, 
                         valid_actions: List[int]) -> None:
        """存储转移"""
        self.states_board.append(obs['board'].copy())
        self.states_upcoming.append(obs['upcoming'].copy())
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.valid_actions_list.append(valid_actions.copy())
    
    def clear_memory(self) -> None:
        """清空经验缓冲区"""
        self.states_board.clear()
        self.states_upcoming.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.valid_actions_list.clear()
    
    def compute_gae(self, next_value: float) -> Tuple[List[float], List[float]]:
        """计算GAE优势函数"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_v = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_v = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_v * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        
        # 计算returns
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def decay_exploration(self):
        """衰减探索参数"""
        # 衰减 ε-贪婪
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # 衰减熵系数
        self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)
    
    def update(self, next_obs: Dict, num_epochs: int = 4, batch_size: int = 64) -> Dict:
        """
        更新网络
        
        Args:
            next_obs: 下一个观察（用于计算下一个价值）
            num_epochs: 更新轮数
            batch_size: 批次大小
        
        Returns:
            训练统计信息
        """
        # 计算下一个价值
        with torch.no_grad():
            board_tensor = torch.FloatTensor(next_obs['board']).unsqueeze(0).to(self.device)
            upcoming_tensor = torch.FloatTensor(next_obs['upcoming']).unsqueeze(0).to(self.device)
            _, next_value = self.network(board_tensor, upcoming_tensor)
            next_value = next_value.item()
        
        # 计算GAE和returns
        advantages, returns = self.compute_gae(next_value)
        
        # 转换为tensor
        states_board = torch.FloatTensor(np.array(self.states_board)).to(self.device)
        states_upcoming = torch.FloatTensor(np.array(self.states_upcoming)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # 创建有效动作掩码
        valid_actions_mask = torch.zeros(len(self.actions), self.network.action_dim, dtype=torch.bool).to(self.device)
        for i, valid_actions in enumerate(self.valid_actions_list):
            if len(valid_actions) > 0:
                valid_actions_mask[i, valid_actions] = True
            else:
                valid_actions_mask[i, :] = True
        
        # 训练统计
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl_div = 0  # KL散度，用于监控策略变化
        num_updates = 0
        
        # 多轮更新
        dataset_size = len(self.actions)
        indices = np.arange(dataset_size)
        
        for epoch in range(num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                # 获取批次数据
                batch_board = states_board[batch_idx]
                batch_upcoming = states_upcoming[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_tensor[batch_idx]
                batch_returns = returns_tensor[batch_idx]
                batch_mask = valid_actions_mask[batch_idx]
                
                # 前向传播（不使用温度，用标准 softmax）
                action_probs, values = self.network(batch_board, batch_upcoming, batch_mask, temperature=1.0)
                
                # 计算新的log概率
                new_log_probs = torch.log(action_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1) + 1e-10)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 计算裁剪的目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = nn.functional.mse_loss(values.squeeze(1), batch_returns)
                
                # 计算熵（只在有效动作上计算）
                # 对每行进行归一化，只在有效动作上计算
                masked_probs = action_probs * batch_mask.float()
                row_sums = masked_probs.sum(dim=1, keepdim=True) + 1e-10
                normalized_probs = masked_probs / row_sums
                entropy = -(normalized_probs * torch.log(normalized_probs + 1e-10)).sum(dim=1).mean()
                
                # 计算近似KL散度（用于监控）
                with torch.no_grad():
                    log_ratio = new_log_probs - batch_old_log_probs
                    kl_div = (torch.exp(log_ratio) - 1 - log_ratio).mean().item()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl_div += kl_div
                num_updates += 1
        
        # 衰减探索参数
        self.decay_exploration()
        
        # 清空经验缓冲区
        self.clear_memory()
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'kl_div': total_kl_div / num_updates,
            'epsilon': self.epsilon,
            'entropy_coef': self.entropy_coef,
        }
    
    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    # 测试网络
    agent = PPOAgent()
    
    # 创建随机输入
    board = np.random.randint(-1, 7, size=(9, 9))
    upcoming = np.random.randint(0, 7, size=(3,))
    
    obs = {'board': board, 'upcoming': upcoming}
    valid_actions = [0, 1, 2, 3, 4]
    
    action, log_prob, value = agent.select_action(obs, valid_actions)
    print(f"Action: {action}, Log Prob: {log_prob:.4f}, Value: {value:.4f}")
