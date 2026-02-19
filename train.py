"""
PPO训练脚本
用于训练彩色五子棋智能体
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import torch

from game_env import ColorGomokuEnv
from ppo_agent import PPOAgent

# 尝试导入 wandb，如果失败则给出友好提示
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' to enable experiment tracking.")


class Logger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str, use_wandb: bool = False, wandb_project: str = "color-gomoku-rl", 
                 wandb_entity: Optional[str] = None, wandb_run_name: Optional[str] = None,
                 wandb_config: Optional[Dict] = None):
        self.log_dir = log_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, 'train.log')
        self.metrics_file = os.path.join(log_dir, 'metrics.jsonl')
        
        # 清空旧日志
        open(self.log_file, 'w').close()
        open(self.metrics_file, 'w').close()
        
        # 初始化 wandb
        if self.use_wandb:
            if wandb_config is None:
                wandb_config = {}
            
            run_name = wandb_run_name or f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
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
        
        # 同时记录到 wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def log_model(self, model_path: str, metadata: Optional[Dict] = None) -> None:
        """记录模型到 wandb"""
        if self.use_wandb and os.path.exists(model_path):
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                metadata=metadata
            )
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
    
    def finish(self):
        """结束 wandb 会话"""
        if self.use_wandb:
            wandb.finish()


def train_ppo(
    num_episodes: int = 10000,
    update_interval: int = 2048,
    eval_interval: int = 100,
    save_interval: int = 500,
    num_eval_episodes: int = 10,
    log_dir: str = './logs',
    model_dir: str = './models',
    use_shaped_reward: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "color-gomoku-rl",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    **agent_kwargs
) -> None:
    """
    训练PPO智能体
    
    Args:
        num_episodes: 训练回合数
        update_interval: 每隔多少步更新一次
        eval_interval: 每隔多少回合评估一次
        save_interval: 每隔多少回合保存一次模型
        num_eval_episodes: 评估时的回合数
        log_dir: 日志目录
        model_dir: 模型保存目录
        **agent_kwargs: 智能体参数
    """
    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 准备 wandb 配置
    wandb_config = {
        'num_episodes': num_episodes,
        'update_interval': update_interval,
        'eval_interval': eval_interval,
        'use_shaped_reward': use_shaped_reward,
        **agent_kwargs
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
    logger.log("Starting PPO training for Color Gomoku")
    logger.log(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # 创建环境（启用奖励塑形）
    env = ColorGomokuEnv(use_shaped_reward=True)
    
    # 创建智能体
    agent = PPOAgent(
        board_size=env.board_size,
        num_colors=env.num_colors,
        **agent_kwargs
    )
    
    logger.log(f"Agent created with parameters: {agent_kwargs}")
    logger.log(f"Use shaped reward: {use_shaped_reward}")
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    total_steps = 0
    best_eval_score = -float('inf')
    
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        truncated = False
        
        while not done and not truncated:
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            # 选择动作
            action, log_prob, value = agent.select_action(obs, valid_actions, deterministic=False)
            
            # 执行动作
            next_obs, raw_reward, done, truncated, info = env.step(action)
            
            # 使用原始奖励或塑形奖励
            if use_shaped_reward and 'shaped_reward' in info:
                reward = info['shaped_reward']
            else:
                reward = raw_reward
            
            # 存储转移
            agent.store_transition(obs, action, log_prob, reward, value, done or truncated, valid_actions)
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            obs = next_obs
            
            # 更新网络
            if total_steps % update_interval == 0:
                update_info = agent.update(next_obs)
                logger.log(f"Update at step {total_steps}: "
                      f"policy_loss={update_info['policy_loss']:.4f}, "
                      f"value_loss={update_info['value_loss']:.4f}, "
                      f"entropy={update_info['entropy']:.4f}, "
                      f"epsilon={update_info['epsilon']:.4f}, "
                      f"entropy_coef={update_info['entropy_coef']:.4f}")
                
                # 记录训练更新指标到 wandb
                update_metrics = {
                    'train/policy_loss': update_info['policy_loss'],
                    'train/value_loss': update_info['value_loss'],
                    'train/entropy': update_info['entropy'],
                    'train/kl_div': update_info.get('kl_div', 0),
                    'train/epsilon': update_info['epsilon'],
                    'train/entropy_coef': update_info['entropy_coef'],
                    'train/learning_rate': agent.optimizer.param_groups[0]['lr'],
                }
                logger.log_metrics(update_metrics, step=total_steps)
        
        # 记录回合统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(info['score'])
        
        # 定期评估
        if episode % eval_interval == 0:
            eval_rewards, eval_scores, eval_lengths = evaluate_agent(agent, env, num_eval_episodes)
            
            avg_train_reward = np.mean(episode_rewards[-eval_interval:])
            avg_train_score = np.mean(episode_scores[-eval_interval:])
            avg_train_length = np.mean(episode_lengths[-eval_interval:])
            
            logger.log(f"Episode {episode}/{num_episodes}")
            logger.log(f"  Train - Reward: {avg_train_reward:.2f}, Score: {avg_train_score:.2f}, Length: {avg_train_length:.1f}")
            logger.log(f"  Eval  - Reward: {np.mean(eval_rewards):.2f}, Score: {np.mean(eval_scores):.2f}, Length: {np.mean(eval_lengths):.1f}")
            logger.log(f"  Total Steps: {total_steps}, Time: {(time.time() - start_time) / 60:.1f}min")
            
            # 记录指标
            metrics = {
                'episode': episode,
                'total_steps': total_steps,
                'train/reward': avg_train_reward,
                'train/score': avg_train_score,
                'train/length': avg_train_length,
                'eval/reward': np.mean(eval_rewards),
                'eval/score': np.mean(eval_scores),
                'eval/length': np.mean(eval_lengths),
                'eval/score_std': np.std(eval_scores),
                'eval/reward_std': np.std(eval_rewards),
                'best_score': best_eval_score,
            }
            logger.log_metrics(metrics, step=total_steps)
            
            # 保存最佳模型
            if np.mean(eval_scores) > best_eval_score:
                best_eval_score = np.mean(eval_scores)
                best_model_path = os.path.join(model_dir, 'best_model.pt')
                agent.save(best_model_path)
                logger.log(f"  New best model saved! Score: {best_eval_score:.2f}")
                
                # 保存最佳模型到 wandb
                logger.log_model(best_model_path, metadata={'score': best_eval_score, 'episode': episode})
        
        # 定期保存模型
        if episode % save_interval == 0:
            checkpoint_path = os.path.join(model_dir, f'model_episode_{episode}.pt')
            agent.save(checkpoint_path)
            logger.log(f"Model saved at episode {episode}")
            
            # 保存检查点到 wandb
            logger.log_model(checkpoint_path, metadata={'episode': episode, 'total_steps': total_steps})
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'final_model.pt')
    agent.save(final_model_path)
    logger.log_model(final_model_path, metadata={'episode': num_episodes, 'total_steps': total_steps, 'final': True})
    logger.log("Training completed!")
    logger.log(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    
    # 结束 wandb
    logger.finish()


def evaluate_agent(agent: PPOAgent, env: ColorGomokuEnv, num_episodes: int) -> tuple:
    """
    评估智能体
    
    Returns:
        rewards: 各回合奖励列表
        scores: 各回合得分列表
        lengths: 各回合长度列表
    """
    rewards = []
    scores = []
    lengths = []
    
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
            
            action, _, _ = agent.select_action(obs, valid_actions, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
        
        rewards.append(episode_reward)
        scores.append(info['score'])
        lengths.append(episode_length)
    
    return rewards, scores, lengths


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for Color Gomoku')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--update_interval', type=int, default=2048, help='Update interval')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--save_interval', type=int, default=500, help='Model save interval')
    parser.add_argument('--num_eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    
    # 智能体参数
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.1, help='Entropy coefficient')
    parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for exploration (>1 increases exploration)')
    parser.add_argument('--epsilon_start', type=float, default=0.3, help='Epsilon-greedy start value')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='Epsilon-greedy end value')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon-greedy decay rate')
    parser.add_argument('--entropy_decay', type=float, default=0.999, help='Entropy coefficient decay rate')
    parser.add_argument('--min_entropy_coef', type=float, default=0.01, help='Minimum entropy coefficient')
    
    # 目录参数
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--model_dir', type=str, default='./models', help='Model directory')
    
    # Wandb 参数
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='color-gomoku-rl', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity (username or team)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name (default: auto-generated)')
    
    args = parser.parse_args()
    
    # 训练
    train_ppo(
        num_episodes=args.num_episodes,
        update_interval=args.update_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        num_eval_episodes=args.num_eval_episodes,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        temperature=args.temperature,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        entropy_decay=args.entropy_decay,
        min_entropy_coef=args.min_entropy_coef,
    )


if __name__ == "__main__":
    main()
