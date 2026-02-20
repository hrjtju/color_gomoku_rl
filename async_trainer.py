"""
基于 Ray 的真正异步 PPO 训练器
Collector 和 Trainer 完全解耦，通过共享缓冲区通信
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import threading
import queue

try:
    import ray
    from ray.util.queue import Queue as RayQueue
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: ray not installed. Async training will not be available.")

from ppo_agent import ActorCritic
from per_buffer import PrioritizedReplayBuffer, RolloutBuffer, Transition


def to_cpu_state_dict(state_dict: Dict) -> Dict:
    """将 state_dict 中的所有张量移到 CPU，用于 Ray 传输"""
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v 
            for k, v in state_dict.items()}


@dataclass
class ExperienceBatch:
    """经验批次"""
    states_board: torch.Tensor
    states_upcoming: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    valid_actions_mask: torch.Tensor
    is_weights: torch.Tensor
    indices: np.ndarray


@ray.remote
class SharedReplayBuffer:
    """
    Ray Actor: 共享的回放缓冲区
    所有 Collector 将经验推送到这里，Trainer 从这里采样
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_decay: float = 0.999,
        epsilon: float = 1e-6,
    ):
        self.per_buffer = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_decay=beta_decay,
            epsilon=epsilon,
        )
        self.stats = {
            'total_added': 0,
            'total_sampled': 0,
            'episodes_added': 0,
        }
    
    def add_transitions(self, transitions_with_priorities: List[Tuple]):
        """
        批量添加带优先级的转移
        
        Args:
            transitions_with_priorities: List of (transition_dict, priority)
        """
        for trans_dict, priority in transitions_with_priorities:
            trans = Transition(**trans_dict)
            self.per_buffer.add(trans, priority)
            self.stats['total_added'] += 1
        self.stats['episodes_added'] += 1
        return len(transitions_with_priorities)
    
    def sample(self, batch_size: int) -> Optional[Tuple]:
        """
        采样一批经验
        
        Returns:
            (transitions_dict_list, indices, is_weights) or None if not ready
        """
        if not self.per_buffer.is_ready(batch_size):
            return None
        
        transitions, indices, is_weights = self.per_buffer.sample(batch_size)
        
        # 将 Transition 对象转换为可序列化的字典
        trans_dicts = []
        for t in transitions:
            trans_dicts.append({
                'obs_board': t.obs_board,
                'obs_upcoming': t.obs_upcoming,
                'action': t.action,
                'log_prob': t.log_prob,
                'reward': t.reward,
                'value': t.value,
                'done': t.done,
                'valid_actions': t.valid_actions,
                'next_obs_board': t.next_obs_board,
                'next_obs_upcoming': t.next_obs_upcoming,
            })
        
        self.stats['total_sampled'] += batch_size
        return trans_dicts, indices, is_weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        self.per_buffer.update_priorities(indices, priorities)
        return True
    
    def decay_beta(self):
        """衰减 beta"""
        self.per_buffer.decay_beta()
        return self.per_buffer.beta
    
    def get_stats(self) -> Dict:
        """获取缓冲区统计信息"""
        stats = self.stats.copy()
        stats['buffer_size'] = len(self.per_buffer)
        stats['beta'] = self.per_buffer.beta
        return stats
    
    def is_ready(self, batch_size: int) -> bool:
        """检查是否可以采样"""
        return self.per_buffer.is_ready(batch_size)


@ray.remote
class AsyncExperienceCollector:
    """
    Ray Actor: 异步经验收集器
    批量收集经验，推送到共享缓冲区
    （注意：Ray Actor 方法是顺序执行的，因此不使用无限循环）
    """
    
    def __init__(
        self,
        shared_buffer,
        board_size: int = 9,
        num_colors: int = 7,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        temperature: float = 2.0,
        epsilon: float = 0.1,
        worker_id: int = 0,
        device: str = 'cpu',
    ):
        self.worker_id = worker_id
        self.shared_buffer = shared_buffer
        self.board_size = board_size
        self.num_colors = num_colors
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.temperature = temperature
        self.epsilon = epsilon
        self.device = torch.device(device)
        
        # 创建本地网络（用于推理）
        self.network = ActorCritic(board_size, num_colors, hidden_dim).to(self.device)
        self.network.eval()
        
        # 创建本地环境
        from game_env import ColorGomokuEnv
        self.env = ColorGomokuEnv(use_shaped_reward=True)
        
        self.collected_episodes = 0
        self.collected_transitions = 0
    
    def set_weights(self, state_dict: Dict):
        """同步网络权重 - 使用 map_location 处理 CPU/GPU 差异"""
        # 将所有张量移到正确的设备
        state_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in state_dict.items()}
        self.network.load_state_dict(state_dict)
        return self.worker_id
    
    def collect_batch(self, num_episodes: int = 1) -> Dict:
        """
        批量收集经验（这是原子操作，可以被调用多次）
        
        Args:
            num_episodes: 要收集的回合数
            
        Returns:
            收集统计信息
        """
        batch_trans_with_priorities = []
        batch_stats = []
        
        for _ in range(num_episodes):
            # 收集一个回合
            transitions, reward, length, score = self._collect_single_episode()
            
            # 转换为可序列化的格式
            for trans in transitions:
                trans_dict = {
                    'obs_board': trans.obs_board,
                    'obs_upcoming': trans.obs_upcoming,
                    'action': trans.action,
                    'log_prob': trans.log_prob,
                    'reward': trans.reward,
                    'value': trans.value,
                    'done': trans.done,
                    'valid_actions': trans.valid_actions,
                    'next_obs_board': trans.next_obs_board,
                    'next_obs_upcoming': trans.next_obs_upcoming,
                }
                priority = abs(trans.reward) + 1.0
                batch_trans_with_priorities.append((trans_dict, priority))
            
            self.collected_episodes += 1
            self.collected_transitions += len(transitions)
            batch_stats.append({
                'reward': reward,
                'length': length,
                'score': score,
                'transitions': len(transitions),
            })
        
        # 推送到共享缓冲区
        if batch_trans_with_priorities:
            ray.get(self.shared_buffer.add_transitions.remote(batch_trans_with_priorities))
        
        return {
            'worker_id': self.worker_id,
            'episodes_collected': len(batch_stats),
            'transitions_collected': sum(s['transitions'] for s in batch_stats),
            'avg_reward': sum(s['reward'] for s in batch_stats) / len(batch_stats) if batch_stats else 0,
            'avg_score': sum(s['score'] for s in batch_stats) / len(batch_stats) if batch_stats else 0,
        }
    
    def get_stats(self) -> Dict:
        """获取收集统计信息"""
        return {
            'worker_id': self.worker_id,
            'episodes': self.collected_episodes,
            'transitions': self.collected_transitions,
        }
    
    def _collect_single_episode(self) -> Tuple[List[Transition], float, int, int]:
        """收集单个回合"""
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        rollout_buffer = RolloutBuffer(self.gamma, self.gae_lambda)
        
        done = False
        truncated = False
        
        while not done and not truncated:
            valid_actions = self.env.get_valid_actions()
            
            # 选择动作
            with torch.no_grad():
                board = torch.FloatTensor(obs['board']).unsqueeze(0).to(self.device)
                upcoming = torch.FloatTensor(obs['upcoming']).unsqueeze(0).to(self.device)
                
                # 创建掩码
                mask = torch.zeros(1, self.network.action_dim, dtype=torch.bool).to(self.device)
                if len(valid_actions) > 0:
                    mask[0, valid_actions] = True
                else:
                    mask[0, :] = True
                
                action_probs, value = self.network(board, upcoming, mask, self.temperature)
                
                # ε-贪婪
                if np.random.random() < self.epsilon and len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    action = torch.multinomial(action_probs, 1).item()
                
                log_prob = torch.log(action_probs[0, action] + 1e-10).item()
                value = value.item()
            
            # 执行动作
            next_obs, reward, done, truncated, info = self.env.step(action)
            reward_to_use = info.get('shaped_reward', reward)
            
            transition = Transition(
                obs_board=obs['board'].copy(),
                obs_upcoming=obs['upcoming'].copy(),
                action=action,
                log_prob=log_prob,
                reward=reward_to_use,
                value=value,
                done=done or truncated,
                valid_actions=valid_actions.copy(),
                next_obs_board=next_obs['board'].copy(),
                next_obs_upcoming=next_obs['upcoming'].copy(),
            )
            
            rollout_buffer.add(transition)
            episode_reward += reward_to_use
            episode_length += 1
            obs = next_obs
        
        rollout_buffer.mark_episode_end()
        
        # 计算 GAE
        with torch.no_grad():
            board_tensor = torch.FloatTensor(obs['board']).unsqueeze(0).to(self.device)
            upcoming_tensor = torch.FloatTensor(obs['upcoming']).unsqueeze(0).to(self.device)
            _, next_value = self.network(board_tensor, upcoming_tensor)
            next_value = next_value.item()
        
        advantages, returns = rollout_buffer.compute_gae(next_value)
        transitions_with_priority = rollout_buffer.get_transitions_with_priorities(advantages)
        transitions = [t for t, _ in transitions_with_priority]
        
        return transitions, episode_reward, episode_length, info.get('score', 0)


class AsyncPPOTrainer:
    """
    真正异步的 PPO 训练器
    
    架构:
    - 1 个 SharedReplayBuffer Actor (共享经验池)
    - N 个 AsyncExperienceCollector Actor (持续收集经验)
    - 主进程持续从缓冲区采样并训练
    
    Collector 和 Trainer 完全解耦:
    - Collector 持续运行，不断收集经验到共享缓冲区
    - Trainer 只要有足够数据就开始训练
    - 定期同步网络权重到所有 Collector
    """
    
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
        entropy_coef: float = 0.1,
        max_grad_norm: float = 0.5,
        device: str = 'auto',
        # 经验收集参数
        num_workers: int = 4,
        temperature: float = 2.0,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        entropy_decay: float = 0.999,
        min_entropy_coef: float = 0.01,
        # PER 参数
        buffer_capacity: int = 100000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        # 训练参数
        batch_size: int = 64,
        num_epochs: int = 4,
        min_buffer_size: int = 1000,
        weight_sync_interval: int = 10,  # 每 N 次训练更新同步权重
    ):
        # 设备设置
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
        
        # 探索参数
        self.temperature = temperature
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.entropy_decay = entropy_decay
        self.min_entropy_coef = min_entropy_coef
        
        # 训练参数
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.min_buffer_size = min_buffer_size
        self.weight_sync_interval = weight_sync_interval
        self.num_workers = num_workers
        
        # 创建主网络
        self.network = ActorCritic(board_size, num_colors, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # 统计信息
        self.total_steps = 0
        self.total_episodes = 0
        self.train_updates = 0
        
        # Ray actors
        self.shared_buffer = None
        self.workers = None
        self.worker_futures = None
        self.using_ray = RAY_AVAILABLE and num_workers > 0
        
        if self.using_ray:
            self._init_ray_actors(
                buffer_capacity=buffer_capacity,
                per_alpha=per_alpha,
                per_beta_start=per_beta_start,
            )
    
    def _init_ray_actors(self, buffer_capacity: int, per_alpha: float, per_beta_start: float):
        """初始化 Ray Actors"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # 创建共享缓冲区
        self.shared_buffer = SharedReplayBuffer.remote(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
        )
        
        # 获取 collector 设备 (CPU 即可，因为只是推理)
        collector_device = 'cpu'
        
        # 创建 worker actors
        self.workers = [
            AsyncExperienceCollector.remote(
                shared_buffer=self.shared_buffer,
                board_size=self.board_size,
                num_colors=self.num_colors,
                hidden_dim=self.network.board_encoder.fc[0].out_features,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                temperature=self.temperature,
                epsilon=self.epsilon,
                worker_id=i,
                device=collector_device,
            )
            for i in range(self.num_workers)
        ]
        
        # 同步初始权重
        self._sync_weights_to_workers()
        
        print(f"Initialized {self.num_workers} Ray workers (device: {collector_device})")
        print(f"Main trainer device: {self.device}")
    
    def _sync_weights_to_workers(self):
        """同步权重到所有 worker - 确保在 CPU 上传输"""
        cpu_state_dict = to_cpu_state_dict(self.network.state_dict())
        state_dict_ref = ray.put(cpu_state_dict)
        ray.get([w.set_weights.remote(state_dict_ref) for w in self.workers])
    
    def collect_experience_async(self, episodes_per_worker: int = 2) -> Dict:
        """
        异步收集一批经验（真正的异步：所有 worker 并行执行）
        
        Args:
            episodes_per_worker: 每个 worker 收集的回合数
            
        Returns:
            收集统计信息
        """
        if not self.using_ray:
            return {'total_transitions': 0, 'avg_reward': 0, 'avg_score': 0}
        
        # 并行分发收集任务给所有 workers
        futures = [w.collect_batch.remote(episodes_per_worker) for w in self.workers]
        results = ray.get(futures)
        
        # 合并统计
        total_episodes = sum(r['episodes_collected'] for r in results)
        total_transitions = sum(r['transitions_collected'] for r in results)
        avg_reward = sum(r['avg_reward'] for r in results) / len(results)
        avg_score = sum(r['avg_score'] for r in results) / len(results)
        
        return {
            'total_episodes': total_episodes,
            'total_transitions': total_transitions,
            'avg_reward': avg_reward,
            'avg_score': avg_score,
        }
    
    def start_collectors(self):
        """（已弃用，保留兼容性）"""
        pass
    
    def stop_collectors(self, timeout: float = 10.0):
        """（已弃用，保留兼容性）"""
        pass
    
    def train_step(self) -> Optional[Dict]:
        """
        执行一步训练
        从共享缓冲区采样（如果可用）
        
        Returns:
            训练统计信息，如果缓冲区数据不足则返回 None
        """
        if not self.using_ray:
            return None
        
        # 从共享缓冲区采样
        sample_result = ray.get(self.shared_buffer.sample.remote(self.batch_size))
        if sample_result is None:
            return None
        
        trans_dicts, indices, is_weights = sample_result
        
        # 转换回 Transition 对象
        transitions = []
        for d in trans_dicts:
            transitions.append(Transition(
                obs_board=d['obs_board'],
                obs_upcoming=d['obs_upcoming'],
                action=d['action'],
                log_prob=d['log_prob'],
                reward=d['reward'],
                value=d['value'],
                done=d['done'],
                valid_actions=d['valid_actions'],
                next_obs_board=d.get('next_obs_board'),
                next_obs_upcoming=d.get('next_obs_upcoming'),
            ))
        
        # 准备批次数据
        batch = self._prepare_batch(transitions, is_weights, indices)
        
        # 训练
        stats = self._update_network(batch)
        
        # 更新优先级
        new_priorities = self._compute_priorities(batch, stats)
        ray.get(self.shared_buffer.update_priorities.remote(indices, new_priorities))
        
        # 衰减探索参数
        self._decay_exploration()
        
        # 定期同步权重到 workers
        self.train_updates += 1
        if self.train_updates % self.weight_sync_interval == 0:
            self._sync_weights_to_workers()
            # 衰减 beta
            beta = ray.get(self.shared_buffer.decay_beta.remote())
            stats['beta'] = beta
        
        return stats
    
    def get_buffer_stats(self) -> Dict:
        """获取缓冲区统计信息"""
        if self.using_ray:
            return ray.get(self.shared_buffer.get_stats.remote())
        return {}
    
    def _prepare_batch(
        self,
        transitions: List[Transition],
        is_weights: np.ndarray,
        indices: np.ndarray,
    ) -> ExperienceBatch:
        """准备训练批次"""
        states_board = torch.FloatTensor(np.array([t.obs_board for t in transitions])).to(self.device)
        states_upcoming = torch.FloatTensor(np.array([t.obs_upcoming for t in transitions])).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        old_log_probs = torch.FloatTensor([t.log_prob for t in transitions]).to(self.device)
        
        # 计算优势和回报（简化版本，使用存储的值）
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        values = torch.FloatTensor([t.value for t in transitions]).to(self.device)
        
        # 使用简单的 TD 误差作为优势
        advantages = rewards - values
        returns = rewards
        
        # 创建有效动作掩码
        valid_actions_mask = torch.zeros(len(transitions), self.network.action_dim, dtype=torch.bool).to(self.device)
        for i, t in enumerate(transitions):
            if len(t.valid_actions) > 0:
                valid_actions_mask[i, t.valid_actions] = True
            else:
                valid_actions_mask[i, :] = True
        
        return ExperienceBatch(
            states_board=states_board,
            states_upcoming=states_upcoming,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            valid_actions_mask=valid_actions_mask,
            is_weights=torch.FloatTensor(is_weights).to(self.device),
            indices=indices,
        )
    
    def _update_network(self, batch: ExperienceBatch) -> Dict:
        """更新网络"""
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        # 标准化优势
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for epoch in range(self.num_epochs):
            # 前向传播
            action_probs, values = self.network(
                batch.states_board,
                batch.states_upcoming,
                batch.valid_actions_mask,
                temperature=1.0,
            )
            
            # 计算新的 log 概率
            new_log_probs = torch.log(action_probs.gather(1, batch.actions.unsqueeze(1)).squeeze(1) + 1e-10)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - batch.old_log_probs)
            
            # PPO 损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = nn.functional.mse_loss(values.squeeze(1), batch.returns)
            
            # 熵
            masked_probs = action_probs * batch.valid_actions_mask.float()
            row_sums = masked_probs.sum(dim=1, keepdim=True) + 1e-10
            normalized_probs = masked_probs / row_sums
            entropy = -(normalized_probs * torch.log(normalized_probs + 1e-10)).sum(dim=1).mean()
            
            # 总损失（应用重要性采样权重）
            loss = (policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy) * batch.is_weights.mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            num_updates += 1
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'epsilon': self.epsilon,
            'entropy_coef': self.entropy_coef,
        }
    
    def _compute_priorities(self, batch: ExperienceBatch, stats: Dict) -> np.ndarray:
        """计算新的优先级（基于 TD 误差）"""
        with torch.no_grad():
            action_probs, values = self.network(
                batch.states_board,
                batch.states_upcoming,
                batch.valid_actions_mask,
                temperature=1.0,
            )
            td_errors = torch.abs(batch.returns - values.squeeze(1)).cpu().numpy()
        
        return td_errors
    
    def _decay_exploration(self):
        """衰减探索参数"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'epsilon': self.epsilon,
            'entropy_coef': self.entropy_coef,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_start)
        self.entropy_coef = checkpoint.get('entropy_coef', self.entropy_coef)


def shutdown_ray():
    """关闭 Ray"""
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    print("Testing Async PPO Trainer (True Async)...")
    
    # 创建训练器
    trainer = AsyncPPOTrainer(
        num_workers=2,
        batch_size=32,
        min_buffer_size=100,
        weight_sync_interval=5,
    )
    
    print(f"Using Ray: {trainer.using_ray}")
    print(f"Trainer device: {trainer.device}")
    
    if not trainer.using_ray:
        print("Ray not available, skipping test")
        import sys
        sys.exit(0)
    
    # 批量收集经验
    print("\nCollecting experience (async parallel)...")
    collect_stats = trainer.collect_experience_async(episodes_per_worker=5)
    print(f"Collected: {collect_stats}")
    
    # 检查缓冲区状态
    buffer_stats = trainer.get_buffer_stats()
    print(f"Buffer stats: {buffer_stats}")
    
    # 训练几步
    print("\nTraining...")
    for i in range(10):
        stats = trainer.train_step()
        if stats:
            print(f"Step {i+1}: policy_loss={stats['policy_loss']:.4f}, "
                  f"value_loss={stats['value_loss']:.4f}, entropy={stats['entropy']:.4f}")
            # 每3步收集一次新经验
            if i % 3 == 0:
                trainer.collect_experience_async(episodes_per_worker=1)
        else:
            print(f"Step {i+1}: Buffer not ready yet")
    
    # 获取最终结果
    buffer_stats = trainer.get_buffer_stats()
    print(f"\nFinal buffer stats: {buffer_stats}")
    
    shutdown_ray()
    print("\nTest completed!")
