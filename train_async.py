"""
异步并行 PPO 训练脚本
支持优先级经验回放和 Ray 并行计算
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import torch

from game_env import ColorGomokuEnv
from async_trainer import AsyncPPOTrainer, shutdown_ray
from ppo_agent import PPOAgent  # 用于评估

# 尝试导入 wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str, use_wandb: bool = False, wandb_project: str = "color-gomoku-rl", 
                 wandb_entity: Optional[str] = None, wandb_run_name: Optional[str] = None,
                 wandb_config: Optional[Dict] = None):
        self.log_dir = log_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, 'train_async.log')
        self.metrics_file = os.path.join(log_dir, 'metrics_async.jsonl')
        
        # 清空旧日志
        open(self.log_file, 'w').close()
        open(self.metrics_file, 'w').close()
        
        # 初始化 wandb
        if self.use_wandb:
            if wandb_config is None:
                wandb_config = {}
            
            run_name = wandb_run_name or f"async_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                config=wandb_config,
                dir=log_dir,
            )
            self.log(f"Wandb initialized: project={wandb_project}, run_name={run_name}")
    
    def log(self, message: str) -> None:
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def log_metrics(self, metrics: Dict, step: Optional[int] = None) -> None:
        """记录指标"""
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_model(self, model_path: str, metadata: Optional[Dict] = None) -> None:
        """记录模型到 wandb"""
        if self.use_wandb and os.path.exists(model_path):
            artifact = wandb.Artifact(
                name=f"model-async-{wandb.run.id}",
                type="model",
                metadata=metadata
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
    
    def finish(self):
        """结束 wandb 会话"""
        if self.use_wandb:
            wandb.finish()


def train_async_ppo(
    # 训练流程参数
    num_iterations: int = 1000,
    eval_interval: int = 10,
    save_interval: int = 50,
    num_eval_episodes: int = 10,
    log_dir: str = './logs',
    model_dir: str = './models',
    use_wandb: bool = False,
    wandb_project: str = "color-gomoku-rl",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    # 异步训练器参数
    num_workers: int = 4,
    episodes_per_worker: int = 2,
    batch_size: int = 128,
    num_epochs: int = 4,
    update_interval: int = 1000,
    min_buffer_size: int = 1000,
    # 网络参数
    hidden_dim: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.1,
    # PER 参数
    buffer_capacity: int = 100000,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    # 探索参数
    temperature: float = 2.0,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    entropy_decay: float = 0.999,
    min_entropy_coef: float = 0.01,
) -> None:
    """
    训练异步 PPO 智能体
    
    Args:
        num_iterations: 训练迭代次数（经验收集-训练循环）
        eval_interval: 每隔多少迭代评估一次
        save_interval: 每隔多少迭代保存一次模型
        num_workers: Ray worker 数量
        episodes_per_worker: 每个 worker 每次收集的回合数
        batch_size: 训练批次大小
        num_epochs: 每次更新的 epoch 数
        update_interval: 每隔多少步进行一次训练更新
        min_buffer_size: 开始训练所需的最小缓冲区大小
    """
    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 准备 wandb 配置
    wandb_config = {
        'num_iterations': num_iterations,
        'num_workers': num_workers,
        'episodes_per_worker': episodes_per_worker,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'update_interval': update_interval,
        'min_buffer_size': min_buffer_size,
        'hidden_dim': hidden_dim,
        'lr': lr,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_epsilon': clip_epsilon,
        'value_coef': value_coef,
        'entropy_coef': entropy_coef,
        'per_alpha': per_alpha,
        'per_beta_start': per_beta_start,
        'temperature': temperature,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay': epsilon_decay,
    }
    
    # 初始化日志
    logger = Logger(
        log_dir=log_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        wandb_config=wandb_config
    )
    logger.log("Starting Async PPO training for Color Gomoku")
    logger.log(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # 创建异步训练器
    trainer = AsyncPPOTrainer(
        board_size=9,
        num_colors=7,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        device='auto',
        num_workers=num_workers,
        episodes_per_worker=episodes_per_worker,
        temperature=temperature,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        entropy_decay=entropy_decay,
        min_entropy_coef=min_entropy_coef,
        buffer_capacity=buffer_capacity,
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        batch_size=batch_size,
        num_epochs=num_epochs,
        min_buffer_size=min_buffer_size,
        update_interval=update_interval,
    )
    
    logger.log(f"AsyncPPOTrainer created with {num_workers} workers")
    logger.log(f"Using Ray: {trainer.using_ray}")
    logger.log(f"Buffer capacity: {buffer_capacity}, PER alpha: {per_alpha}")
    
    # 训练统计
    iteration_rewards = []
    iteration_scores = []
    iteration_lengths = []
    
    total_steps = 0
    best_eval_score = -float('inf')
    
    start_time = time.time()
    
    try:
        for iteration in range(1, num_iterations + 1):
            iter_start_time = time.time()
            
            # 1. 异步收集经验
            num_trans, avg_reward, avg_length, avg_score = trainer.collect_experience_async()
            
            total_steps += int(num_trans)
            iteration_rewards.append(avg_reward)
            iteration_scores.append(avg_score)
            iteration_lengths.append(avg_length)
            
            collect_time = time.time() - iter_start_time
            
            # 2. 训练（多次更新直到满足条件）
            train_start_time = time.time()
            update_count = 0
            total_train_stats = {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
            }
            
            # 计算应该进行多少次更新
            num_updates = max(1, num_trans // update_interval)
            
            for _ in range(num_updates):
                stats = trainer.train_step()
                if stats:
                    for key in total_train_stats:
                        total_train_stats[key] += stats[key]
                    update_count += 1
            
            train_time = time.time() - train_start_time
            
            # 平均统计
            if update_count > 0:
                for key in total_train_stats:
                    total_train_stats[key] /= update_count
            
            # 记录日志
            logger.log(
                f"Iter {iteration}/{num_iterations} | "
                f"Steps: {total_steps} | "
                f"Trans: {num_trans} | "
                f"Buffer: {len(trainer.per_buffer)} | "
                f"Reward: {avg_reward:.2f} | "
                f"Score: {avg_score:.2f} | "
                f"Time: {collect_time:.1f}s/{train_time:.1f}s"
            )
            
            # 记录指标到 wandb
            metrics = {
                'iteration': iteration,
                'total_steps': total_steps,
                'buffer_size': len(trainer.per_buffer),
                'collect/transitions': num_trans,
                'collect/reward': avg_reward,
                'collect/length': avg_length,
                'collect/score': avg_score,
                'train/policy_loss': total_train_stats['policy_loss'],
                'train/value_loss': total_train_stats['value_loss'],
                'train/entropy': total_train_stats['entropy'],
                'train/updates': update_count,
                'train/beta': trainer.per_buffer.beta,
                'explore/epsilon': trainer.epsilon,
                'explore/entropy_coef': trainer.entropy_coef,
                'time/collect': collect_time,
                'time/train': train_time,
            }
            logger.log_metrics(metrics, step=total_steps)
            
            # 3. 定期评估
            if iteration % eval_interval == 0:
                eval_start_time = time.time()
                eval_rewards, eval_scores, eval_lengths = evaluate_agent(trainer, num_eval_episodes)
                eval_time = time.time() - eval_start_time
                
                avg_train_reward = np.mean(iteration_rewards[-eval_interval:])
                avg_train_score = np.mean(iteration_scores[-eval_interval:])
                avg_train_length = np.mean(iteration_lengths[-eval_interval:])
                
                logger.log(
                    f"  [Eval] Train - Reward: {avg_train_reward:.2f}, Score: {avg_train_score:.2f}, Length: {avg_train_length:.1f}"
                )
                logger.log(
                    f"  [Eval] Eval  - Reward: {np.mean(eval_rewards):.2f}, Score: {np.mean(eval_scores):.2f}, "
                    f"Length: {np.mean(eval_lengths):.1f}, Time: {eval_time:.1f}s"
                )
                
                # 记录评估指标
                eval_metrics = {
                    'eval/train_reward': avg_train_reward,
                    'eval/train_score': avg_train_score,
                    'eval/train_length': avg_train_length,
                    'eval/reward': np.mean(eval_rewards),
                    'eval/score': np.mean(eval_scores),
                    'eval/score_std': np.std(eval_scores),
                    'eval/length': np.mean(eval_lengths),
                    'eval/best_score': best_eval_score,
                    'time/eval': eval_time,
                }
                logger.log_metrics(eval_metrics, step=total_steps)
                
                # 保存最佳模型
                current_eval_score = np.mean(eval_scores)
                if current_eval_score > best_eval_score:
                    best_eval_score = current_eval_score
                    best_model_path = os.path.join(model_dir, 'best_model_async.pt')
                    trainer.save(best_model_path)
                    logger.log(f"  [Eval] New best model saved! Score: {best_eval_score:.2f}")
                    logger.log_model(best_model_path, metadata={'score': best_eval_score, 'iteration': iteration})
            
            # 4. 定期保存检查点
            if iteration % save_interval == 0:
                checkpoint_path = os.path.join(model_dir, f'model_async_iter_{iteration}.pt')
                trainer.save(checkpoint_path)
                logger.log(f"Model saved at iteration {iteration}")
                logger.log_model(checkpoint_path, metadata={'iteration': iteration, 'total_steps': total_steps})
        
        # 训练完成
        final_model_path = os.path.join(model_dir, 'final_model_async.pt')
        trainer.save(final_model_path)
        logger.log_model(final_model_path, metadata={'iteration': num_iterations, 'total_steps': total_steps, 'final': True})
        logger.log("Training completed!")
        logger.log(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
        
    except KeyboardInterrupt:
        logger.log("Training interrupted by user")
        # 保存中断时的模型
        interrupt_path = os.path.join(model_dir, 'model_async_interrupted.pt')
        trainer.save(interrupt_path)
        logger.log(f"Model saved to {interrupt_path}")
    
    finally:
        # 清理资源
        shutdown_ray()
        logger.finish()


def evaluate_agent(trainer: AsyncPPOTrainer, num_episodes: int) -> tuple:
    """
    评估智能体（使用确定性策略）
    """
    from game_env import ColorGomokuEnv
    
    env = ColorGomokuEnv(use_shaped_reward=False)  # 评估时使用原始奖励
    
    rewards = []
    scores = []
    lengths = []
    
    trainer.network.eval()
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            truncated = False
            
            while not done and not truncated:
                valid_actions = env.get_valid_actions()
                
                if len(valid_actions) == 0:
                    break
                
                # 使用主网络选择动作（贪婪）
                board_tensor = torch.FloatTensor(obs['board']).unsqueeze(0).to(trainer.device)
                upcoming_tensor = torch.FloatTensor(obs['upcoming']).unsqueeze(0).to(trainer.device)
                
                # 创建掩码
                mask = torch.zeros(1, trainer.network.action_dim, dtype=torch.bool).to(trainer.device)
                mask[0, valid_actions] = True
                
                action_probs, _ = trainer.network(board_tensor, upcoming_tensor, mask, temperature=1.0)
                action = torch.argmax(action_probs, dim=1).item()
                
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
            
            rewards.append(episode_reward)
            scores.append(info['score'])
            lengths.append(episode_length)
    
    trainer.network.train()
    
    return rewards, scores, lengths


def main():
    parser = argparse.ArgumentParser(description='Async PPO training for Color Gomoku')
    
    # 训练流程参数
    parser.add_argument('--num_iterations', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=50, help='Model save interval')
    parser.add_argument('--num_eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    
    # 异步参数
    parser.add_argument('--num_workers', type=int, default=4, help='Number of Ray workers')
    parser.add_argument('--episodes_per_worker', type=int, default=2, help='Episodes per worker per iteration')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of training epochs per update')
    parser.add_argument('--update_interval', type=int, default=1000, help='Update interval (steps)')
    parser.add_argument('--min_buffer_size', type=int, default=1000, help='Minimum buffer size to start training')
    
    # 网络参数
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.1, help='Entropy coefficient')
    
    # PER 参数
    parser.add_argument('--buffer_capacity', type=int, default=100000, help='Replay buffer capacity')
    parser.add_argument('--per_alpha', type=float, default=0.6, help='PER alpha (0=uniform, 1=full prioritized)')
    parser.add_argument('--per_beta_start', type=float, default=0.4, help='PER beta start')
    
    # 探索参数
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for exploration')
    parser.add_argument('--epsilon_start', type=float, default=0.3, help='Epsilon-greedy start')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='Epsilon-greedy end')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--entropy_decay', type=float, default=0.999, help='Entropy coefficient decay')
    parser.add_argument('--min_entropy_coef', type=float, default=0.01, help='Minimum entropy coefficient')
    
    # 目录参数
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--model_dir', type=str, default='./models', help='Model directory')
    
    # Wandb 参数
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='color-gomoku-rl', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    
    args = parser.parse_args()
    
    # 训练
    train_async_ppo(
        num_iterations=args.num_iterations,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        num_eval_episodes=args.num_eval_episodes,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        num_workers=args.num_workers,
        episodes_per_worker=args.episodes_per_worker,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        update_interval=args.update_interval,
        min_buffer_size=args.min_buffer_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        buffer_capacity=args.buffer_capacity,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        temperature=args.temperature,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        entropy_decay=args.entropy_decay,
        min_entropy_coef=args.min_entropy_coef,
    )


if __name__ == "__main__":
    main()
