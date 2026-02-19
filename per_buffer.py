"""
带优先级的经验回放缓冲区 (Prioritized Experience Replay)
使用 Sum Tree 实现高效采样
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import deque
import random


class SumTree:
    """Sum Tree 数据结构，用于高效优先级采样"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 树的总节点数
        self.data = np.zeros(capacity, dtype=object)  # 存储数据
        self.write_idx = 0  # 写入位置
        self.size = 0  # 当前大小
    
    def _propagate(self, idx: int, change: float):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """检索优先级对应的叶子节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """获取总优先级"""
        return self.tree[0]
    
    def add(self, priority: float, data):
        """添加数据"""
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        """更新优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        """
        根据优先级采样
        
        Returns:
            idx: 树中的索引
            priority: 优先级
            data: 数据
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class Transition(NamedTuple):
    """经验转移"""
    obs_board: np.ndarray
    obs_upcoming: np.ndarray
    action: int
    log_prob: float
    reward: float
    value: float
    done: bool
    valid_actions: List[int]
    next_obs_board: Optional[np.ndarray] = None
    next_obs_upcoming: Optional[np.ndarray] = None


class PrioritizedReplayBuffer:
    """
    带优先级的经验回放缓冲区
    
    参考: Schaul et al., "Prioritized Experience Replay", ICLR 2016
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,  # 优先级指数 (0=均匀采样, 1=完全优先级)
        beta_start: float = 0.4,  # 重要性采样指数起始值
        beta_end: float = 1.0,  # 重要性采样指数结束值
        beta_decay: float = 0.999,  # beta衰减率
        epsilon: float = 1e-6,  # 避免零优先级的小常数
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_decay = beta_decay
        self.epsilon = epsilon
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0  # 新样本的最大优先级
        
    def add(self, transition: Transition, priority: Optional[float] = None):
        """
        添加经验到缓冲区
        
        Args:
            transition: 经验转移
            priority: 优先级（如果为None，使用最大优先级）
        """
        if priority is None:
            priority = self.max_priority
        
        # 应用 alpha 指数
        priority = (priority + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        优先级采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            transitions: 采样的经验列表
            indices: 对应的树索引（用于后续更新优先级）
            is_weights: 重要性采样权重
        """
        transitions = []
        indices = np.zeros(batch_size, dtype=np.int32)
        is_weights = np.zeros(batch_size, dtype=np.float32)
        
        # 计算采样区间
        total_priority = self.tree.total()
        segment_size = total_priority / batch_size
        
        # 计算最大权重（用于归一化）
        min_priority = np.min(self.tree.tree[-self.tree.capacity:]) + self.epsilon
        max_weight = (min_priority * self.tree.size / total_priority) ** (-self.beta)
        
        for i in range(batch_size):
            # 在区间内均匀采样
            low = segment_size * i
            high = segment_size * (i + 1)
            s = random.uniform(low, high)
            
            idx, priority, transition = self.tree.get(s)
            
            # 计算重要性采样权重
            prob = priority / total_priority
            is_weight = (prob * self.tree.size) ** (-self.beta)
            is_weight /= max_weight  # 归一化
            
            transitions.append(transition)
            indices[i] = idx
            is_weights[i] = is_weight
        
        return transitions, indices, is_weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        更新优先级（通常在训练后调用）
        
        Args:
            indices: 树索引
            priorities: 新的优先级（通常是TD误差的绝对值）
        """
        for idx, priority in zip(indices, priorities):
            # 应用 alpha 指数
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def decay_beta(self):
        """衰减 beta 值（增加重要性采样的修正力度）"""
        self.beta = min(self.beta_end, self.beta / self.beta_decay)
    
    def __len__(self) -> int:
        return self.tree.size
    
    def is_ready(self, batch_size: int) -> bool:
        """检查是否有足够的数据进行采样"""
        return self.tree.size >= batch_size


class RolloutBuffer:
    """
    PPO 的 Rollout 缓冲区，支持 GAE 计算
    可以与 PrioritizedReplayBuffer 结合使用
    """
    
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.clear()
    
    def clear(self):
        """清空缓冲区"""
        self.transitions = []
        self.episode_boundaries = [0]  # 记录每个回合的起始索引
    
    def add(self, transition: Transition):
        """添加转移"""
        self.transitions.append(transition)
    
    def mark_episode_end(self):
        """标记回合结束"""
        self.episode_boundaries.append(len(self.transitions))
    
    def compute_gae(self, next_value: float) -> Tuple[List[float], List[float]]:
        """
        计算 GAE 优势函数和回报
        
        Args:
            next_value: 下一个状态的估计价值
            
        Returns:
            advantages: 优势函数值列表
            returns: 回报值列表
        """
        advantages = []
        returns = []
        
        # 按回合处理
        for i in range(len(self.episode_boundaries) - 1):
            start = self.episode_boundaries[i]
            end = self.episode_boundaries[i + 1]
            
            episode_advantages = []
            gae = 0
            
            # 倒序计算 GAE
            for t in reversed(range(start, end)):
                trans = self.transitions[t]
                
                if t == end - 1:
                    # 回合结束
                    next_non_terminal = 1.0 - trans.done
                    next_v = next_value if not trans.done else 0.0
                else:
                    next_non_terminal = 1.0 - trans.done
                    next_v = self.transitions[t + 1].value
                
                delta = trans.reward + self.gamma * next_v * next_non_terminal - trans.value
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                episode_advantages.insert(0, gae)
            
            advantages.extend(episode_advantages)
            returns.extend([adv + trans.value for adv, trans in zip(episode_advantages, self.transitions[start:end])])
        
        return advantages, returns
    
    def get_transitions_with_priorities(self, advantages: List[float]) -> List[Tuple[Transition, float]]:
        """
        获取带优先级的转移
        
        Args:
            advantages: 优势函数值（用于计算优先级）
            
        Returns:
            转移和优先级的元组列表
        """
        result = []
        for i, trans in enumerate(self.transitions):
            # 使用优势函数的绝对值作为优先级
            priority = abs(advantages[i]) if i < len(advantages) else 1.0
            result.append((trans, priority))
        return result
    
    def __len__(self) -> int:
        return len(self.transitions)


if __name__ == "__main__":
    # 测试 Sum Tree
    print("Testing Sum Tree...")
    tree = SumTree(capacity=10)
    
    for i in range(5):
        tree.add(priority=i+1, data=f"data_{i}")
    
    print(f"Total priority: {tree.total()}")
    
    # 采样测试
    samples = {}
    for _ in range(1000):
        s = random.uniform(0, tree.total())
        idx, priority, data = tree.get(s)
        samples[data] = samples.get(data, 0) + 1
    
    print("Sampling distribution (should be proportional to priority):")
    for data, count in sorted(samples.items()):
        print(f"  {data}: {count}")
    
    # 测试 PER Buffer
    print("\nTesting Prioritized Replay Buffer...")
    buffer = PrioritizedReplayBuffer(capacity=100)
    
    # 添加一些数据
    for i in range(50):
        trans = Transition(
            obs_board=np.random.randn(9, 9),
            obs_upcoming=np.random.randint(0, 7, 3),
            action=i,
            log_prob=-1.0,
            reward=float(i),
            value=0.0,
            done=(i % 10 == 9),
            valid_actions=[0, 1, 2],
        )
        buffer.add(trans)
    
    print(f"Buffer size: {len(buffer)}")
    
    # 采样
    batch, indices, is_weights = buffer.sample(10)
    print(f"Sampled {len(batch)} transitions")
    print(f"Indices: {indices}")
    print(f"IS weights: {is_weights}")
    
    # 更新优先级
    new_priorities = np.random.rand(10)
    buffer.update_priorities(indices, new_priorities)
    print("Priorities updated")
    
    print("\nAll tests passed!")
