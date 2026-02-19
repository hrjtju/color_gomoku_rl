"""
模型评估脚本
评估训练好的PPO智能体
"""

import os
import argparse
import numpy as np
from typing import List, Dict
import json

from game_env import ColorGomokuEnv
from ppo_agent import PPOAgent


def evaluate_agent(
    agent: PPOAgent,
    env: ColorGomokuEnv,
    num_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    verbose: bool = False
) -> Dict:
    """
    评估智能体
    
    Args:
        agent: PPO智能体
        env: 游戏环境
        num_episodes: 评估回合数
        deterministic: 是否使用确定性策略
        render: 是否渲染
        verbose: 是否打印详细信息
    
    Returns:
        评估统计信息
    """
    rewards = []
    scores = []
    lengths = []
    matches_counts = []
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=episode * 1000)
        episode_reward = 0
        episode_length = 0
        episode_matches = 0
        
        done = False
        truncated = False
        
        while not done and not truncated:
            valid_actions = env.get_valid_actions()
            
            if len(valid_actions) == 0:
                break
            
            action, _, _ = agent.select_action(obs, valid_actions, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if reward > 0:
                episode_matches += 1
            
            if render:
                env.render()
        
        rewards.append(episode_reward)
        scores.append(info['score'])
        lengths.append(episode_length)
        matches_counts.append(episode_matches)
        
        if verbose and episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}: "
                  f"Reward={episode_reward:.2f}, Score={info['score']}, Length={episode_length}")
    
    stats = {
        'num_episodes': num_episodes,
        'deterministic': deterministic,
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'reward_min': np.min(rewards),
        'reward_max': np.max(rewards),
        'score_mean': np.mean(scores),
        'score_std': np.std(scores),
        'score_min': np.min(scores),
        'score_max': np.max(scores),
        'length_mean': np.mean(lengths),
        'length_std': np.std(lengths),
        'matches_mean': np.mean(matches_counts),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO agent')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file')
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy')
    parser.add_argument('--render', action='store_true', help='Render gameplay')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        print(f"错误: 找不到模型文件 {args.model_path}")
        return
    
    # 创建环境
    env = ColorGomokuEnv()
    
    # 加载智能体
    agent = PPOAgent(board_size=env.board_size, num_colors=env.num_colors)
    agent.load(args.model_path)
    
    print(f"已加载模型: {args.model_path}")
    print(f"评估回合数: {args.num_episodes}")
    print(f"确定性策略: {args.deterministic}")
    print("-" * 50)
    
    # 评估
    stats = evaluate_agent(
        agent, env, 
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        render=args.render,
        verbose=args.verbose
    )
    
    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"回合数: {stats['num_episodes']}")
    print()
    print("奖励统计:")
    print(f"  平均: {stats['reward_mean']:.2f}")
    print(f"  标准差: {stats['reward_std']:.2f}")
    print(f"  最小: {stats['reward_min']:.2f}")
    print(f"  最大: {stats['reward_max']:.2f}")
    print()
    print("得分统计:")
    print(f"  平均: {stats['score_mean']:.2f}")
    print(f"  标准差: {stats['score_std']:.2f}")
    print(f"  最小: {stats['score_min']:.2f}")
    print(f"  最大: {stats['score_max']:.2f}")
    print()
    print("回合长度统计:")
    print(f"  平均: {stats['length_mean']:.2f}")
    print(f"  标准差: {stats['length_std']:.2f}")
    print()
    print(f"平均消除次数: {stats['matches_mean']:.2f}")
    print("=" * 50)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
