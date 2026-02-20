"""
真正异步的并行 PPO 训练脚本
支持优先级经验回放和 Ray 并行计算

架构:
- 多个 Collector Actor 持续收集经验
- 1 个 SharedReplayBuffer Actor 存储经验
- 主进程持续从缓冲区采样训练
- Collector 和 Trainer 完全异步运行
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
from ppo_agent import PPOAgent

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
    warmup_seconds: float = 5.0,  # 启动后等待缓冲区填充的时间
    eval_interval: int = 100,  # 训练步数
    save_interval: int = 500,  # 训练步数
    num_eval_episodes: int = 10,
    log_dir: str = './logs',
    model_dir: str = './models',
    use_wandb: bool = False,
    wandb_project: str = "color-gomoku-rl",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    # 异步训练器参数
    num_workers: int = 4,
    batch_size: int = 128,
    num_epochs: int = 4,
    min_buffer_size: int = 1000,
    weight_sync_interval: int = 10,  # 每 N 次训练更新同步权重
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
    训练真正异步的 PPO 智能体
    
    流程:
    1. 启动 Collector Workers（持续收集经验）
    2. 等待缓冲区填充到足够数据
    3. 主循环:
       - 从共享缓冲区采样并训练
       - 定期评估和保存模型
       - 权重自动同步（由 trainer 内部处理）
    4. 停止 Collectors
    """
    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 准备 wandb 配置
    wandb_config = {
        'num_iterations': num_iterations,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'min_buffer_size': min_buffer_size,
        'weight_sync_interval': weight_sync_interval,
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
        'entropy_decay': entropy_decay,
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
    logger.log("Starting TRUE Async PPO training for Color Gomoku")
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
        weight_sync_interval=weight_sync_interval,
    )
    
    logger.log(f"AsyncPPOTrainer created with {num_workers} workers")
    logger.log(f"Using Ray: {trainer.using_ray}")
    logger.log(f"Buffer capacity: {buffer_capacity}, PER alpha: {per_alpha}")
    
    if not trainer.using_ray:
        logger.log("ERROR: Ray not available, cannot run async training")
        return
    
    # 初始收集（预热）
    logger.log("\n=== Initial Experience Collection ===")
    initial_collect = trainer.collect_experience_async(episodes_per_worker=3)
    logger.log(f"Initial: {initial_collect['total_transitions']} transitions, "
               f"avg_reward={initial_collect['avg_reward']:.2f}")
    
    # 等待缓冲区有足够数据
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_seconds:
        buffer_stats = trainer.get_buffer_stats()
        if buffer_stats.get('buffer_size', 0) >= min_buffer_size:
            break
        # 继续收集
        trainer.collect_experience_async(episodes_per_worker=1)
        time.sleep(0.1)
    
    buffer_stats = trainer.get_buffer_stats()
    logger.log(f"Warmup complete. Buffer size: {buffer_stats.get('buffer_size', 0)}")
    
    # 训练统计
    best_eval_score = -float('inf')
    train_stats_history = []
    train_steps = 0
    
    start_time = time.time()
    
    # 收集间隔：每 N 步训练补充一次经验
    collect_interval = 5
    
    try:
        logger.log("\n=== Starting Training Loop ===")
        
        while train_steps < num_iterations:
            loop_start = time.time()
            
            # 训练一步
            stats = trainer.train_step()
            
            if stats is None:
                # 缓冲区不足，异步收集更多经验
                collect_stats = trainer.collect_experience_async(episodes_per_worker=1)
                logger.log(f"  Buffer low, collected {collect_stats['total_transitions']} more transitions")
                continue
            
            train_steps += 1
            train_stats_history.append(stats)
            
            # 定期补充经验（异步并行收集）
            if train_steps % collect_interval == 0:
                collect_stats = trainer.collect_experience_async(episodes_per_worker=1)
            
            # 记录训练指标
            if train_steps % 10 == 0:
                logger.log(
                    f"Step {train_steps}/{num_iterations} | "
                    f"policy_loss={stats['policy_loss']:.4f}, "
                    f"value_loss={stats['value_loss']:.4f}, "
                    f"entropy={stats['entropy']:.4f}, "
                    f"epsilon={stats['epsilon']:.4f}"
                )
                
                # 获取缓冲区状态
                buffer_stats = trainer.get_buffer_stats()
                
                metrics = {
                    'train_step': train_steps,
                    'train/policy_loss': stats['policy_loss'],
                    'train/value_loss': stats['value_loss'],
                    'train/entropy': stats['entropy'],
                    'train/epsilon': stats['epsilon'],
                    'train/entropy_coef': stats['entropy_coef'],
                    'train/beta': stats.get('beta', 0),
                    'buffer/size': buffer_stats.get('buffer_size', 0),
                    'buffer/episodes': buffer_stats.get('episodes_added', 0),
                    'buffer/total_added': buffer_stats.get('total_added', 0),
                }
                logger.log_metrics(metrics, step=train_steps)
            
            # 定期评估
            if train_steps % eval_interval == 0:
                eval_start = time.time()
                eval_rewards, eval_scores, eval_lengths = evaluate_agent(trainer, num_eval_episodes)
                eval_time = time.time() - eval_start
                
                # 计算最近训练指标的平均值
                if len(train_stats_history) >= 10:
                    recent_stats = train_stats_history[-10:]
                    avg_policy_loss = np.mean([s['policy_loss'] for s in recent_stats])
                    avg_value_loss = np.mean([s['value_loss'] for s in recent_stats])
                else:
                    avg_policy_loss = stats['policy_loss']
                    avg_value_loss = stats['value_loss']
                
                logger.log(
                    f"\n  [Eval @ step {train_steps}] "
                    f"Train - policy_loss: {avg_policy_loss:.4f}, value_loss: {avg_value_loss:.4f}"
                )
                logger.log(
                    f"  [Eval @ step {train_steps}] "
                    f"Eval  - reward: {np.mean(eval_rewards):.2f}, "
                    f"score: {np.mean(eval_scores):.2f}, "
                    f"length: {np.mean(eval_lengths):.1f}, "
                    f"time: {eval_time:.1f}s"
                )
                
                # 记录评估指标
                eval_metrics = {
                    'eval/step': train_steps,
                    'eval/reward': np.mean(eval_rewards),
                    'eval/score': np.mean(eval_scores),
                    'eval/score_std': np.std(eval_scores),
                    'eval/length': np.mean(eval_lengths),
                    'eval/best_score': best_eval_score,
                    'time/eval': eval_time,
                }
                logger.log_metrics(eval_metrics, step=train_steps)
                
                # 保存最佳模型
                current_eval_score = np.mean(eval_scores)
                if current_eval_score > best_eval_score:
                    best_eval_score = current_eval_score
                    best_model_path = os.path.join(model_dir, 'best_model_async.pt')
                    trainer.save(best_model_path)
                    logger.log(f"  [Eval] New best model saved! Score: {best_eval_score:.2f}")
                    logger.log_model(best_model_path, metadata={'score': best_eval_score, 'step': train_steps})
            
            # 定期保存检查点
            if train_steps % save_interval == 0:
                checkpoint_path = os.path.join(model_dir, f'model_async_step_{train_steps}.pt')
                trainer.save(checkpoint_path)
                logger.log(f"Checkpoint saved at step {train_steps}")
                logger.log_model(checkpoint_path, metadata={'step': train_steps})
        
        # 训练完成
        logger.log("\n=== Training Completed ===")
        final_model_path = os.path.join(model_dir, 'final_model_async.pt')
        trainer.save(final_model_path)
        logger.log_model(final_model_path, metadata={'step': train_steps, 'final': True})
        logger.log(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
        
    except KeyboardInterrupt:
        logger.log("\nTraining interrupted by user")
        interrupt_path = os.path.join(model_dir, 'model_async_interrupted.pt')
        trainer.save(interrupt_path)
        logger.log(f"Model saved to {interrupt_path}")
    
    finally:
        # 清理资源
        shutdown_ray()
        logger.finish()


def evaluate_agent(trainer, num_episodes: int) -> tuple:
    """
    评估智能体（使用确定性策略）
    """
    from game_env import ColorGomokuEnv
    
    env = ColorGomokuEnv(use_shaped_reward=False)
    
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
    parser = argparse.ArgumentParser(description='True Async PPO training for Color Gomoku')
    
    # 训练流程参数
    parser.add_argument('--num_iterations', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--warmup_seconds', type=float, default=5.0, help='Warmup time for buffer to fill')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval (steps)')
    parser.add_argument('--save_interval', type=int, default=500, help='Model save interval (steps)')
    parser.add_argument('--num_eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    
    # 异步参数
    parser.add_argument('--num_workers', type=int, default=4, help='Number of Ray workers')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=4, help='Number of training epochs per update')
    parser.add_argument('--min_buffer_size', type=int, default=1000, help='Minimum buffer size to start training')
    parser.add_argument('--weight_sync_interval', type=int, default=10, help='Weight sync interval (updates)')
    
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
    parser.add_argument('--per_alpha', type=float, default=0.6, help='PER alpha')
    parser.add_argument('--per_beta_start', type=float, default=0.4, help='PER beta start')
    
    # 探索参数
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for exploration')
    parser.add_argument('--epsilon_start', type=float, default=0.3, help='Epsilon-greedy start')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='Epsilon-greedy end')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay')
    parser.add_argument('--entropy_decay', type=float, default=0.999, help='Entropy coefficient decay')
    parser.add_argument('--min_entropy_coef', type=float, default=0.01, help='Minimum entropy coefficient')
    
    # 目录参数
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--model_dir', type=str, default='./models', help='Model directory')
    
    # Wandb 参数
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='color-gomoku-rl', help='Wandb project')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    
    args = parser.parse_args()
    
    # 训练
    train_async_ppo(
        num_iterations=args.num_iterations,
        warmup_seconds=args.warmup_seconds,
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
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        min_buffer_size=args.min_buffer_size,
        weight_sync_interval=args.weight_sync_interval,
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
