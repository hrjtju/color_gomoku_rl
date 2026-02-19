"""
基于 Ray 的异步并行 PPO 训练器
支持优先级经验回放和并行经验收集
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
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: ray not installed. Async training will not be available.")
    print("Install with: pip install ray")

from ppo_agent import ActorCritic
from per_buffer import PrioritizedReplayBuffer, RolloutBuffer, Transition


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


class ExperienceCollector:
    """
    经验收集器（非 Ray 版本，用于单线程或线程池）
    """
    
    def __init__(
        self,
        board_size: int = 9,
        num_colors: int = 7,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        temperature: float = 2.0,
        epsilon: float = 0.1,
        device: str = 'cpu',
    ):
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
    
    def set_weights(self, state_dict: Dict):
        """从主网络同步权重"""
        self.network.load_state_dict(state_dict)
    
    def collect_episode(self, env, global_step: int = 0) -> Tuple[List[Transition], float, int, int]:
        """
        收集一个回合的经验
        
        Returns:
            transitions: 经验列表
            episode_reward: 回合奖励
            episode_length: 回合长度
            episode_score: 最终得分
        """
        # 导入环境
        from game_env import ColorGomokuEnv
        
        if env is None:
            env = ColorGomokuEnv(use_shaped_reward=True)
        
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        rollout_buffer = RolloutBuffer(self.gamma, self.gae_lambda)
        
        done = False
        truncated = False
        
        while not done and not truncated:
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            # 选择动作
            action, log_prob, value = self._select_action(obs, valid_actions)
            
            # 执行动作
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 使用塑形奖励
            reward_to_use = info.get('shaped_reward', reward)
            
            # 创建转移
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
        
        # 将优势函数值添加到转移中
        transitions_with_priority = rollout_buffer.get_transitions_with_priorities(advantages)
        transitions = [t for t, _ in transitions_with_priority]
        
        return transitions, episode_reward, episode_length, info.get('score', 0)
    
    def _select_action(self, obs: Dict, valid_actions: List[int]) -> Tuple[int, float, float]:
        """选择动作"""
        with torch.no_grad():
            board = torch.FloatTensor(obs['board']).unsqueeze(0).to(self.device)
            upcoming = torch.FloatTensor(obs['upcoming']).unsqueeze(0).to(self.device)
            
            action, log_prob, value = self.network.get_action(
                board[0].cpu().numpy(),
                upcoming[0].cpu().numpy(),
                valid_actions,
                deterministic=False,
                temperature=self.temperature,
                epsilon=self.epsilon
            )
            
            return action, log_prob, value


if RAY_AVAILABLE:
    @ray.remote
    class RayExperienceCollector:
        """
        Ray Actor 用于并行经验收集
        """
        
        def __init__(
            self,
            board_size: int = 9,
            num_colors: int = 7,
            hidden_dim: int = 256,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            temperature: float = 2.0,
            epsilon: float = 0.1,
            worker_id: int = 0,
        ):
            self.worker_id = worker_id
            self.board_size = board_size
            self.num_colors = num_colors
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.temperature = temperature
            self.epsilon = epsilon
            
            # 设置设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 创建本地网络
            self.network = ActorCritic(board_size, num_colors, hidden_dim).to(self.device)
            self.network.eval()
            
            # 创建本地环境
            from game_env import ColorGomokuEnv
            self.env = ColorGomokuEnv(use_shaped_reward=True)
        
        def set_weights(self, state_dict: Dict):
            """同步网络权重"""
            self.network.load_state_dict(state_dict)
            return self.worker_id
        
        def collect_episodes(self, num_episodes: int, global_step: int = 0) -> List[Tuple]:
            """
            收集多个回合的经验
            
            Returns:
                每个回合的 (transitions, episode_reward, episode_length, episode_score)
            """
            results = []
            for _ in range(num_episodes):
                result = self._collect_single_episode()
                results.append(result)
            return results
        
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
    异步并行 PPO 训练器
    
    特性:
    1. 多个 Ray worker 并行收集经验
    2. 主进程从 PER 缓冲区采样并训练
    3. 支持优先级经验回放
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
        episodes_per_worker: int = 2,
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
        update_interval: int = 2048,
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
        self.update_interval = update_interval
        self.num_workers = num_workers
        self.episodes_per_worker = episodes_per_worker
        
        # 创建主网络
        self.network = ActorCritic(board_size, num_colors, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # 创建 PER 缓冲区
        self.per_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
        )
        
        # 统计信息
        self.total_steps = 0
        self.total_episodes = 0
        
        # Ray workers
        self.workers = None
        self.using_ray = RAY_AVAILABLE and num_workers > 0
        
        if self.using_ray:
            self._init_ray_workers()
    
    def _init_ray_workers(self):
        """初始化 Ray workers"""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # 创建 workers
        state_dict = self.network.state_dict()
        state_dict_ref = ray.put(state_dict)
        
        self.workers = [
            RayExperienceCollector.remote(
                board_size=self.board_size,
                num_colors=self.num_colors,
                hidden_dim=self.network.board_encoder.fc[0].out_features,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                temperature=self.temperature,
                epsilon=self.epsilon,
                worker_id=i,
            )
            for i in range(self.num_workers)
        ]
        
        # 同步初始权重
        ray.get([w.set_weights.remote(state_dict_ref) for w in self.workers])
        print(f"Initialized {self.num_workers} Ray workers")
    
    def collect_experience_async(self) -> Tuple[int, float, float, float]:
        """
        异步收集经验
        
        Returns:
            num_transitions: 收集的转移数量
            avg_reward: 平均回合奖励
            avg_length: 平均回合长度
            avg_score: 平均得分
        """
        if self.using_ray:
            return self._collect_with_ray()
        else:
            return self._collect_single_thread()
    
    def _collect_with_ray(self) -> Tuple[int, float, float, float]:
        """使用 Ray 并行收集"""
        # 获取当前网络权重
        state_dict = self.network.state_dict()
        state_dict_ref = ray.put(state_dict)
        
        # 异步更新所有 worker 的权重
        weight_futures = [w.set_weights.remote(state_dict_ref) for w in self.workers]
        ray.get(weight_futures)  # 等待权重更新完成
        
        # 异步收集经验
        collect_futures = [
            w.collect_episodes.remote(self.episodes_per_worker, self.total_steps)
            for w in self.workers
        ]
        
        # 收集结果
        all_episodes = ray.get(collect_futures)
        
        total_transitions = 0
        total_reward = 0.0
        total_length = 0
        total_score = 0.0
        num_episodes = 0
        
        for worker_episodes in all_episodes:
            for transitions, reward, length, score in worker_episodes:
                # 添加到 PER 缓冲区
                for trans in transitions:
                    # 使用 TD 误差作为初始优先级（简化版本）
                    priority = abs(trans.reward) + 1.0  # 简单启发式
                    self.per_buffer.add(trans, priority)
                
                total_transitions += len(transitions)
                total_reward += reward
                total_length += length
                total_score += score
                num_episodes += 1
        
        self.total_episodes += num_episodes
        
        avg_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
        avg_length = total_length / num_episodes if num_episodes > 0 else 0.0
        avg_score = total_score / num_episodes if num_episodes > 0 else 0.0
        
        return total_transitions, avg_reward, avg_length, avg_score
    
    def _collect_single_thread(self) -> Tuple[int, float, float, float]:
        """单线程收集（Ray 不可用时的备用方案）"""
        collector = ExperienceCollector(
            board_size=self.board_size,
            num_colors=self.num_colors,
            hidden_dim=self.network.board_encoder.fc[0].out_features,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            temperature=self.temperature,
            epsilon=self.epsilon,
            device=str(self.device),
        )
        collector.set_weights(self.network.state_dict())
        
        total_transitions = 0
        total_reward = 0.0
        total_length = 0
        total_score = 0.0
        num_episodes = 0
        
        from game_env import ColorGomokuEnv
        env = ColorGomokuEnv(use_shaped_reward=True)
        
        for _ in range(self.episodes_per_worker * max(1, self.num_workers)):
            transitions, reward, length, score = collector.collect_episode(env)
            
            for trans in transitions:
                priority = abs(trans.reward) + 1.0
                self.per_buffer.add(trans, priority)
            
            total_transitions += len(transitions)
            total_reward += reward
            total_length += length
            total_score += score
            num_episodes += 1
        
        self.total_episodes += num_episodes
        
        avg_reward = total_reward / num_episodes if num_episodes > 0 else 0.0
        avg_length = total_length / num_episodes if num_episodes > 0 else 0.0
        avg_score = total_score / num_episodes if num_episodes > 0 else 0.0
        
        return total_transitions, avg_reward, avg_length, avg_score
    
    def train_step(self) -> Optional[Dict]:
        """
        执行一步训练
        
        Returns:
            训练统计信息，如果缓冲区数据不足则返回 None
        """
        if not self.per_buffer.is_ready(self.batch_size):
            return None
        
        # 采样
        transitions, indices, is_weights = self.per_buffer.sample(self.batch_size)
        
        # 准备批次数据
        batch = self._prepare_batch(transitions, is_weights, indices)
        
        # 训练
        stats = self._update_network(batch)
        
        # 更新优先级
        new_priorities = self._compute_priorities(batch, stats)
        self.per_buffer.update_priorities(indices, new_priorities)
        
        # 衰减探索参数
        self._decay_exploration()
        
        # 衰减 beta
        self.per_buffer.decay_beta()
        
        return stats
    
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
        dones = torch.FloatTensor([t.done for t in transitions]).to(self.device)
        
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
            'beta': self.per_buffer.beta,
        }
    
    def _compute_priorities(self, batch: ExperienceBatch, stats: Dict) -> np.ndarray:
        """
        计算新的优先级（基于 TD 误差）
        
        这里我们使用简单的启发式：TD 误差的绝对值
        """
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
    print("Testing Async PPO Trainer...")
    
    # 创建训练器
    trainer = AsyncPPOTrainer(
        num_workers=2,
        episodes_per_worker=1,
        batch_size=32,
    )
    
    print(f"Using Ray: {trainer.using_ray}")
    print(f"Device: {trainer.device}")
    
    # 收集一些经验
    print("\nCollecting experience...")
    num_trans, avg_reward, avg_length, avg_score = trainer.collect_experience_async()
    print(f"Collected {num_trans} transitions")
    print(f"Avg reward: {avg_reward:.2f}, Avg length: {avg_length:.1f}, Avg score: {avg_score:.2f}")
    print(f"Buffer size: {len(trainer.per_buffer)}")
    
    # 训练几步
    print("\nTraining...")
    for i in range(5):
        stats = trainer.train_step()
        if stats:
            print(f"Step {i+1}: policy_loss={stats['policy_loss']:.4f}, "
                  f"value_loss={stats['value_loss']:.4f}, entropy={stats['entropy']:.4f}")
        else:
            print(f"Step {i+1}: Not enough data")
    
    shutdown_ray()
    print("\nTest completed!")
