"""
异步训练系统测试脚本
测试优先级经验回放和 Ray 并行训练
"""

import os
import sys
import time
import numpy as np
import torch

def test_sum_tree():
    """测试 Sum Tree"""
    print("=" * 60)
    print("Test 1: Sum Tree")
    print("=" * 60)
    
    from per_buffer import SumTree
    
    tree = SumTree(capacity=10)
    
    # 添加数据
    for i in range(5):
        tree.add(priority=i+1, data=f"data_{i}")
    
    print(f"✓ Added 5 items, total priority: {tree.total():.2f}")
    assert abs(tree.total() - 15.0) < 0.01, "Total priority should be 15"
    
    # 采样测试
    samples = {}
    for _ in range(1000):
        s = np.random.uniform(0, tree.total())
        idx, priority, data = tree.get(s)
        samples[data] = samples.get(data, 0) + 1
    
    # 验证采样分布大致与优先级成比例
    expected_ratios = {f"data_{i}": (i+1)/15.0 for i in range(5)}
    print("✓ Sampling distribution test:")
    for data, count in sorted(samples.items()):
        actual_ratio = count / 1000
        expected_ratio = expected_ratios[data]
        print(f"  {data}: expected={expected_ratio:.2%}, actual={actual_ratio:.2%}")
    
    print("✓ Sum Tree test passed!\n")


def test_per_buffer():
    """测试优先级经验回放缓冲区"""
    print("=" * 60)
    print("Test 2: Prioritized Replay Buffer")
    print("=" * 60)
    
    from per_buffer import PrioritizedReplayBuffer, Transition
    
    buffer = PrioritizedReplayBuffer(
        capacity=1000,
        alpha=0.6,
        beta_start=0.4,
    )
    
    # 添加经验
    for i in range(100):
        trans = Transition(
            obs_board=np.random.randn(9, 9),
            obs_upcoming=np.random.randint(0, 7, 3),
            action=i % 10,
            log_prob=-1.0,
            reward=float(i),
            value=0.0,
            done=(i % 20 == 19),
            valid_actions=[0, 1, 2, 3],
        )
        priority = abs(trans.reward) + 1.0
        buffer.add(trans, priority)
    
    print(f"✓ Added 100 transitions, buffer size: {len(buffer)}")
    assert len(buffer) == 100
    
    # 采样
    batch_size = 32
    transitions, indices, is_weights = buffer.sample(batch_size)
    
    print(f"✓ Sampled {len(transitions)} transitions")
    print(f"  Indices shape: {indices.shape}")
    print(f"  IS weights range: [{is_weights.min():.4f}, {is_weights.max():.4f}]")
    
    # 更新优先级
    new_priorities = np.random.rand(batch_size) * 10
    buffer.update_priorities(indices, new_priorities)
    print("✓ Updated priorities")
    
    # 衰减 beta
    old_beta = buffer.beta
    buffer.decay_beta()
    print(f"✓ Beta decayed: {old_beta:.4f} -> {buffer.beta:.4f}")
    
    print("✓ PER Buffer test passed!\n")


def test_rollout_buffer():
    """测试 Rollout Buffer 和 GAE 计算"""
    print("=" * 60)
    print("Test 3: Rollout Buffer with GAE")
    print("=" * 60)
    
    from per_buffer import RolloutBuffer, Transition
    
    buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
    
    # 模拟一个回合
    for i in range(20):
        trans = Transition(
            obs_board=np.random.randn(9, 9),
            obs_upcoming=np.random.randint(0, 7, 3),
            action=i,
            log_prob=-1.0,
            reward=float(i),
            value=float(i),
            done=(i == 19),
            valid_actions=[0, 1, 2],
        )
        buffer.add(trans)
    
    buffer.mark_episode_end()
    
    print(f"✓ Added {len(buffer)} transitions")
    
    # 计算 GAE
    next_value = 10.0
    advantages, returns = buffer.compute_gae(next_value)
    
    print(f"✓ Computed GAE")
    print(f"  Advantages range: [{min(advantages):.2f}, {max(advantages):.2f}]")
    print(f"  Returns range: [{min(returns):.2f}, {max(returns):.2f}]")
    
    # 验证优势函数值的形状
    assert len(advantages) == len(buffer)
    assert len(returns) == len(buffer)
    
    print("✓ Rollout Buffer test passed!\n")


def test_experience_collector():
    """测试经验收集器"""
    print("=" * 60)
    print("Test 4: Experience Collector")
    print("=" * 60)
    
    from async_trainer import ExperienceCollector
    from game_env import ColorGomokuEnv
    
    collector = ExperienceCollector(
        board_size=9,
        num_colors=7,
        hidden_dim=128,
        gamma=0.99,
        gae_lambda=0.95,
        temperature=1.0,
        epsilon=0.1,
        device='cpu',
    )
    
    env = ColorGomokuEnv(use_shaped_reward=True)
    
    # 收集一个回合
    transitions, reward, length, score = collector.collect_episode(env)
    
    print(f"✓ Collected 1 episode")
    print(f"  Transitions: {len(transitions)}")
    print(f"  Episode reward: {reward:.2f}")
    print(f"  Episode length: {length}")
    print(f"  Episode score: {score}")
    
    assert len(transitions) > 0
    assert length > 0
    
    print("✓ Experience Collector test passed!\n")


def test_ray_worker():
    """测试 Ray Worker"""
    print("=" * 60)
    print("Test 5: Ray Experience Collector")
    print("=" * 60)
    
    try:
        import ray
        from async_trainer import RayExperienceCollector
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # 创建 worker
        worker = RayExperienceCollector.remote(
            board_size=9,
            num_colors=7,
            hidden_dim=128,
            gamma=0.99,
            gae_lambda=0.95,
            temperature=1.0,
            epsilon=0.1,
            worker_id=0,
        )
        
        # 收集经验
        results = ray.get(worker.collect_episodes.remote(2))
        
        print(f"✓ Ray worker collected {len(results)} episodes")
        
        total_trans = 0
        for i, (trans, reward, length, score) in enumerate(results):
            print(f"  Episode {i+1}: {len(trans)} transitions, reward={reward:.2f}, length={length}")
            total_trans += len(trans)
        
        print(f"  Total transitions: {total_trans}")
        
        print("✓ Ray Worker test passed!\n")
        
    except ImportError:
        print("⚠ Ray not available, skipping Ray Worker test\n")


def test_async_trainer():
    """测试异步训练器"""
    print("=" * 60)
    print("Test 6: Async PPO Trainer")
    print("=" * 60)
    
    from async_trainer import AsyncPPOTrainer
    
    trainer = AsyncPPOTrainer(
        board_size=9,
        num_colors=7,
        hidden_dim=128,
        lr=3e-4,
        num_workers=2,
        episodes_per_worker=1,
        batch_size=32,
        num_epochs=2,
        buffer_capacity=10000,
    )
    
    print(f"✓ Created AsyncPPOTrainer")
    print(f"  Using Ray: {trainer.using_ray}")
    print(f"  Device: {trainer.device}")
    
    # 收集经验
    num_trans, avg_reward, avg_length, avg_score = trainer.collect_experience_async()
    
    print(f"✓ Collected {num_trans} transitions")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"  Avg length: {avg_length:.1f}")
    print(f"  Avg score: {avg_score:.2f}")
    print(f"  Buffer size: {len(trainer.per_buffer)}")
    
    # 训练
    print("✓ Training for 3 steps...")
    for i in range(3):
        stats = trainer.train_step()
        if stats:
            print(f"  Step {i+1}: policy_loss={stats['policy_loss']:.4f}, "
                  f"value_loss={stats['value_loss']:.4f}, "
                  f"entropy={stats['entropy']:.4f}")
        else:
            print(f"  Step {i+1}: Not enough data")
    
    # 保存和加载
    temp_path = '/tmp/test_async_model.pt'
    trainer.save(temp_path)
    print(f"✓ Saved model to {temp_path}")
    
    trainer.load(temp_path)
    print(f"✓ Loaded model from {temp_path}")
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    print("✓ Async Trainer test passed!\n")


def test_training_integration():
    """端到端训练集成测试"""
    print("=" * 60)
    print("Test 7: End-to-End Training Integration")
    print("=" * 60)
    
    from async_trainer import AsyncPPOTrainer
    from game_env import ColorGomokuEnv
    
    # 创建训练器（使用较小的配置）
    trainer = AsyncPPOTrainer(
        board_size=9,
        num_colors=7,
        hidden_dim=64,  # 较小的网络
        lr=3e-4,
        num_workers=2,
        episodes_per_worker=1,
        batch_size=32,
        num_epochs=2,
        buffer_capacity=5000,
        per_alpha=0.6,
        temperature=1.5,
    )
    
    print(f"✓ Created trainer (Ray: {trainer.using_ray})")
    
    # 模拟几个训练迭代
    rewards_history = []
    scores_history = []
    
    for iteration in range(3):
        # 收集经验
        num_trans, avg_reward, avg_length, avg_score = trainer.collect_experience_async()
        rewards_history.append(avg_reward)
        scores_history.append(avg_score)
        
        print(f"  Iter {iteration+1}: {num_trans} transitions, "
              f"reward={avg_reward:.2f}, score={avg_score:.2f}")
        
        # 训练
        num_updates = max(1, num_trans // 500)
        for _ in range(num_updates):
            stats = trainer.train_step()
    
    # 评估
    print("✓ Evaluating trained agent...")
    env = ColorGomokuEnv(use_shaped_reward=False)
    eval_rewards = []
    eval_scores = []
    
    trainer.network.eval()
    with torch.no_grad():
        for _ in range(3):
            obs, info = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            
            while not done and not truncated:
                valid_actions = env.get_valid_actions()
                if len(valid_actions) == 0:
                    break
                
                board_tensor = torch.FloatTensor(obs['board']).unsqueeze(0).to(trainer.device)
                upcoming_tensor = torch.FloatTensor(obs['upcoming']).unsqueeze(0).to(trainer.device)
                
                mask = torch.zeros(1, trainer.network.action_dim, dtype=torch.bool).to(trainer.device)
                mask[0, valid_actions] = True
                
                action_probs, _ = trainer.network(board_tensor, upcoming_tensor, mask)
                action = torch.argmax(action_probs, dim=1).item()
                
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_scores.append(info['score'])
    
    print(f"  Eval rewards: {eval_rewards}")
    print(f"  Eval scores: {eval_scores}")
    
    print("✓ Integration test passed!\n")


def cleanup():
    """清理 Ray"""
    print("=" * 60)
    print("Cleanup")
    print("=" * 60)
    
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            print("✓ Ray shutdown")
    except:
        pass
    
    print("✓ Cleanup complete\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("Async PPO Training System Test Suite")
    print("=" * 60 + "\n")
    
    start_time = time.time()
    
    tests = [
        ("Sum Tree", test_sum_tree),
        ("PER Buffer", test_per_buffer),
        ("Rollout Buffer", test_rollout_buffer),
        ("Experience Collector", test_experience_collector),
        ("Ray Worker", test_ray_worker),
        ("Async Trainer", test_async_trainer),
        ("Training Integration", test_training_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {name} test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    cleanup()
    
    elapsed = time.time() - start_time
    
    print("=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed, {elapsed:.1f}s")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
