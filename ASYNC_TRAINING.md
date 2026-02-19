# 异步并行 PPO 训练系统

本系统实现了带优先级经验回放（Prioritized Experience Replay, PER）的异步并行 PPO 训练，使用 Ray 进行分布式经验收集。

## 特性

- **优先级经验回放 (PER)**: 使用 Sum Tree 高效采样，优先学习高 TD 误差的样本
- **异步并行**: 多个 Ray Worker 同时收集经验，主进程专注训练
- **GAE 优势估计**: 支持回合内 GAE 计算
- **重要性采样**: PER 的重要性采样权重修正

## 文件结构

```
color_gomoku_rl/
├── per_buffer.py          # PER 缓冲区和 Sum Tree 实现
├── async_trainer.py       # 异步训练器 (Ray + PER)
├── train_async.py         # 异步训练脚本
├── test_async_training.py # 测试脚本
└── ...
```

## 快速开始

### 1. 运行测试

```bash
# 运行所有测试
python test_async_training.py

# 单独测试 PER 缓冲区
python per_buffer.py

# 单独测试异步训练器
python async_trainer.py
```

### 2. 开始训练

```bash
# 基本用法（4 个 worker）
python train_async.py --num_workers 4 --num_iterations 1000

# 完整参数
python train_async.py \
    --num_workers 4 \
    --episodes_per_worker 2 \
    --num_iterations 1000 \
    --batch_size 128 \
    --buffer_capacity 100000 \
    --per_alpha 0.6 \
    --lr 3e-4 \
    --use_wandb
```

### 3. 参数说明

#### 异步参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_workers` | 4 | Ray Worker 数量 |
| `--episodes_per_worker` | 2 | 每个 Worker 每次收集的回合数 |
| `--batch_size` | 128 | 训练批次大小 |
| `--min_buffer_size` | 1000 | 开始训练所需的最小缓冲区大小 |

#### PER 参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--buffer_capacity` | 100000 | 回放缓冲区容量 |
| `--per_alpha` | 0.6 | 优先级指数 (0=均匀采样, 1=完全优先级) |
| `--per_beta_start` | 0.4 | 重要性采样指数起始值 |

#### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_iterations` | 1000 | 训练迭代次数 |
| `--eval_interval` | 10 | 评估间隔 |
| `--save_interval` | 50 | 模型保存间隔 |
| `--num_epochs` | 4 | 每次更新的 epoch 数 |
| `--update_interval` | 1000 | 更新间隔（步数） |

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     AsyncPPOTrainer                         │
│  ┌─────────────────┐      ┌─────────────────────────────┐  │
│  │   Main Network  │◄─────│    PrioritizedReplayBuffer  │  │
│  │   (Training)    │      │    (Sum Tree + PER)         │  │
│  └────────┬────────┘      └──────────────────────┬──────┘  │
│           │                                       │         │
│           │ Pull gradients                        │ Push    │
│           ▼                                       │         │
│  ┌─────────────────┐                              │         │
│  │  Optimizer      │                              │         │
│  │  (Adam)         │                              │         │
│  └─────────────────┘                              │         │
└───────────────────────────────────────────────────┼─────────┘
                                                    │
                     ┌──────────────────────────────┘
                     │ Async Experience Collection
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     Ray Workers                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐               │
│  │ Worker 0  │  │ Worker 1  │  │ Worker 2  │  ...           │
│  │ (Env 0)   │  │ (Env 1)   │  │ (Env 2)   │                │
│  └───────────┘  └───────────┘  └───────────┘               │
│       │               │               │                     │
│       └───────────────┴───────────────┘                     │
│                   │                                         │
│                   ▼                                         │
│         ┌─────────────────┐                                 │
│         │  Local Network  │                                 │
│         │  (Inference)    │                                 │
│         └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────┘
```

## 工作流程

1. **经验收集**: 
   - 多个 Ray Worker 并行运行环境
   - 每个 Worker 使用本地网络进行推理
   - 收集的经验带 TD 误差优先级存入 PER 缓冲区

2. **训练更新**:
   - 从 PER 缓冲区优先级采样
   - 应用重要性采样权重修正偏差
   - 执行 PPO 更新
   - 更新样本优先级

3. **权重同步**:
   - 定期将主网络权重广播到所有 Worker
   - Worker 更新本地网络继续收集

## 性能对比

异步训练相比串行训练的优势:

- **CPU 利用率**: 多个 Worker 并行，充分利用多核 CPU
- **GPU 利用率**: 主进程专注训练，持续利用 GPU
- **样本效率**: PER 优先学习重要样本
- **扩展性**: 易于增加 Worker 数量提升吞吐量

## 注意事项

1. **Ray 初始化**: 首次使用 Ray 可能需要下载依赖，请耐心等待
2. **内存使用**: 大量 Worker 会增加内存占用，可适当减小 `buffer_capacity`
3. **Worker 数量**: 建议设置为 CPU 核心数 - 1，保留一个核心给训练
4. **GIL 限制**: Python GIL 可能影响 CPU 密集型任务，考虑使用 `num_workers <= cpu_count`

## 故障排除

### Ray 启动失败
```bash
# 尝试重启 Ray
ray stop
python train_async.py
```

### 内存不足
```bash
# 减小缓冲区容量和 Worker 数量
python train_async.py --buffer_capacity 50000 --num_workers 2
```

### 训练不稳定
```bash
# 调整 PER 参数
python train_async.py --per_alpha 0.4 --per_beta_start 0.6
```
