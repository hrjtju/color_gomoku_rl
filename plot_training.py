"""
训练进度可视化脚本
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(log_dir='./logs'):
    """加载训练指标"""
    metrics_file = os.path.join(log_dir, 'metrics.jsonl')
    
    if not os.path.exists(metrics_file):
        print(f"找不到指标文件: {metrics_file}")
        return None
    
    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    
    return metrics


def plot_training_progress(metrics, save_path=None):
    """绘制训练进度"""
    if not metrics:
        print("没有数据可绘制")
        return
    
    episodes = [m['episode'] for m in metrics]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PPO Training Progress', fontsize=16)
    
    # 1. 奖励
    ax = axes[0, 0]
    ax.plot(episodes, [m['train_reward'] for m in metrics], 'b-', label='Train', alpha=0.7)
    ax.plot(episodes, [m['eval_reward'] for m in metrics], 'r-', label='Eval', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 得分
    ax = axes[0, 1]
    ax.plot(episodes, [m['train_score'] for m in metrics], 'b-', label='Train', alpha=0.7)
    ax.plot(episodes, [m['eval_score'] for m in metrics], 'r-', label='Eval', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Game Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 回合长度
    ax = axes[1, 0]
    ax.plot(episodes, [m['train_length'] for m in metrics], 'b-', label='Train', alpha=0.7)
    ax.plot(episodes, [m['eval_length'] for m in metrics], 'r-', label='Eval', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 总步数
    ax = axes[1, 1]
    ax.plot(episodes, [m['total_steps'] for m in metrics], 'g-')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Steps')
    ax.set_title('Total Steps')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()


def print_summary(metrics):
    """打印训练摘要"""
    if not metrics:
        return
    
    latest = metrics[-1]
    best_eval_score = max(m['eval_score'] for m in metrics)
    best_eval_episode = max(metrics, key=lambda m: m['eval_score'])['episode']
    
    print("\n" + "=" * 50)
    print("训练摘要")
    print("=" * 50)
    print(f"总回合数: {latest['episode']}")
    print(f"总步数: {latest['total_steps']}")
    print(f"最新训练奖励: {latest['train_reward']:.2f}")
    print(f"最新评估奖励: {latest['eval_reward']:.2f}")
    print(f"最新训练得分: {latest['train_score']:.2f}")
    print(f"最新评估得分: {latest['eval_score']:.2f}")
    print(f"最佳评估得分: {best_eval_score:.2f} (回合 {best_eval_episode})")
    print(f"最新训练长度: {latest['train_length']:.1f}")
    print(f"最新评估长度: {latest['eval_length']:.1f}")
    print("=" * 50)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot training progress')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--save', type=str, default=None, help='Save plot to file')
    parser.add_argument('--summary', action='store_true', help='Print summary only')
    
    args = parser.parse_args()
    
    metrics = load_metrics(args.log_dir)
    
    if metrics is None:
        sys.exit(1)
    
    print(f"加载了 {len(metrics)} 条记录")
    
    print_summary(metrics)
    
    if not args.summary:
        plot_training_progress(metrics, args.save)


if __name__ == "__main__":
    main()
