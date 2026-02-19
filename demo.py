"""
演示脚本 - 展示AI随机玩游戏
"""

import numpy as np
from game_env import ColorGomokuEnv

def demo_random_play(num_episodes=3, max_steps=100):
    """演示随机玩"""
    env = ColorGomokuEnv()
    
    print("=" * 60)
    print("彩色五子棋 - 随机AI演示")
    print("=" * 60)
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset(seed=episode * 100)
        episode_reward = 0
        step_count = 0
        
        print(f"\n回合 {episode}/{num_episodes}")
        print("-" * 40)
        
        # 显示初始状态
        print("初始棋盘:")
        env.render()
        print(f"即将出现的棋子: {[env.COLORS[c] for c in env.upcoming_pieces]}")
        
        done = False
        truncated = False
        
        while not done and not truncated and step_count < max_steps:
            valid_actions = env.get_valid_actions()
            
            if len(valid_actions) == 0:
                print("没有有效动作！")
                break
            
            # 随机选择动作
            action = np.random.choice(valid_actions)
            from_pos, to_pos = env.decode_action(action)
            
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # 只显示重要的步骤（有奖励或每20步）
            if reward > 0 or step_count % 20 == 0 or done:
                print(f"\n步骤 {step_count}: 从 {from_pos} 移动到 {to_pos}")
                if reward > 0:
                    print(f"  *** 获得奖励: {reward} ***")
                env.render()
                print(f"即将出现的棋子: {[env.COLORS[c] for c in env.upcoming_pieces]}")
        
        print(f"\n回合结束!")
        print(f"  总步数: {step_count}")
        print(f"  总奖励: {episode_reward}")
        print(f"  最终得分: {info['score']}")
        print(f"  剩余空格: {info['empty_cells']}")
    
    print("\n" + "=" * 60)
    print("演示结束!")
    print("=" * 60)


def demo_match_detection():
    """演示消除检测"""
    env = ColorGomokuEnv()
    
    print("\n" + "=" * 60)
    print("消除检测演示")
    print("=" * 60)
    
    # 创建一个包含五子连珠的棋盘
    obs, info = env.reset(seed=42)
    env.board = np.full((9, 9), env.EMPTY, dtype=np.int32)
    
    # 横向五子（第0行，0-4列）
    for c in range(5):
        env.board[0, c] = 0  # 深蓝
    
    # 竖向六子（第8列，0-5行）
    for r in range(6):
        env.board[r, 8] = 1  # 浅蓝
    
    # 对角线五子（第2-6行，0-4列）
    for i in range(5):
        env.board[2+i, i] = 2  # 浅紫
    
    print("\n设置测试棋盘:")
    print("- 第4行0-4列: 深蓝（横向五子）")
    print("- 第7列0-5行: 浅蓝（竖向六子）")
    print("- 对角线0-4: 浅紫（对角线五子）")
    
    env.render()
    
    # 检测匹配
    matches = env._find_matches()
    print(f"\n检测到 {len(matches)} 组匹配:")
    
    for i, match in enumerate(matches):
        print(f"  匹配 {i+1}: {len(match)} 个棋子 - {match}")
    
    # 计算得分
    score = env._calculate_score(matches)
    print(f"\n预计得分: {score}")
    print("  (5个=10分, 6个=14分, 7个=18分...)")
    
    # 执行消除
    env._remove_matches(matches)
    print("\n消除后的棋盘:")
    env.render()


if __name__ == "__main__":
    # 演示随机玩
    demo_random_play(num_episodes=2, max_steps=50)
    
    # 演示消除检测
    demo_match_detection()
