# 异步并行 PPO 训练系统

本系统实现了带优先级经验回放（Prioritized Experience Replay, PER）的异步并行 PPO 训练，使用 Ray 进行分布式经验收集。

## 特性

- **优先级经验回放 (PER)**: 使用 Sum Tree 高效采样，优先学习高 TD 误差的样本
- **真正异步**: Collector 和 Trainer 完全解耦，通过共享缓冲区通信
- **CPU/GPU 兼容**: 修复了 Ray 序列化 CUDA 张量的错误，支持单卡服务器
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

# 单卡服务器（CPU workers + GPU trainer）
python train_async.py \
    --num_workers 4 \
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
| `--batch_size` | 128 | 训练批次大小 |
| `--min_buffer_size` | 1000 | 开始训练所需的最小缓冲区大小 |
| `--weight_sync_interval` | 10 | 每 N 次训练更新同步权重 |

#### PER 参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--buffer_capacity` | 100000 | 回放缓冲区容量 |
| `--per_alpha` | 0.6 | 优先级指数 (0=均匀采样, 1=完全优先级) |
| `--per_beta_start` | 0.4 | 重要性采样指数起始值 |

#### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_iterations` | 1000 | 训练步数 |
| `--eval_interval` | 100 | 评估间隔（训练步数） |
| `--save_interval` | 500 | 模型保存间隔（训练步数） |
| `--num_epochs` | 4 | 每次更新的 epoch 数 |

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Process                            │
│  ┌─────────────────┐      ┌─────────────────────────────┐  │
│  │   Main Network  │◄─────│      Trainer (GPU/CPU)      │  │
│  │   (Training)    │      │                             │  │
│  └────────┬────────┘      └─────────────────────────────┘  │
│           │                                                 │
│           │ 1. Sync weights (CPU)                           │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  collect_experience_async() - Parallel Ray Calls    │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐               │   │
│  │  │Worker 0 │ │Worker 1 │ │Worker 2 │ ...            │   │
│  │  │(batch)  │ │(batch)  │ │(batch)  │                │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘               │   │
│  └───────┼───────────┼───────────┼──────────────────────┘   │
│          │           │           │                          │
│          ▼           ▼           ▼                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           SharedReplayBuffer (Ray Actor)            │   │
│  │              (Prioritized Experience Pool)          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 工作流程

1. **并行经验收集**: 
   - 主进程调用 `collect_experience_async()`，并行分发任务给所有 Workers
   - 每个 Worker 收集指定数量的回合
   - Workers 将经验推送到 SharedReplayBuffer

2. **持续训练**:
   - 主进程从 SharedReplayBuffer 采样
   - 执行 PPO 更新
   - 定期同步网络权重到所有 Workers（自动处理 CPU/GPU 转换）

3. **动态补充**:
   - 当缓冲区不足时，自动触发新的经验收集
   - 训练和数据收集交错进行

## 关键修复：CUDA 序列化错误

原错误：`RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`

**解决方案**:
1. Workers 使用 CPU 设备（仅推理，无需 GPU）
2. 传输前将 state_dict 显式转换为 CPU 张量：
   ```python
   def to_cpu_state_dict(state_dict):
       return {k: v.cpu() if isinstance(v, torch.Tensor) else v 
               for k, v in state_dict.items()}
   ```
3. Worker 加载时使用 `map_location` 处理设备转换

## 性能对比

异步训练相比串行训练的优势:

- **CPU 利用率**: 多个 Worker 并行收集，充分利用多核 CPU
- **GPU 利用率**: 主进程专注训练，GPU 持续处于计算状态
- **吞吐量**: 收集和训练并行，减少等待时间
- **样本效率**: PER 优先学习重要样本，加速收敛

## 注意事项

1. **Ray 初始化**: 首次使用 Ray 可能需要下载依赖，请耐心等待
2. **内存使用**: 大量 Worker 会增加内存占用，可适当减小 `buffer_capacity`
3. **Worker 数量**: 建议设置为 CPU 核心数 - 1，保留一个核心给训练
4. **权重同步**: 默认每 10 次训练更新同步一次，可根据需要调整

## 故障排除

### Ray 启动失败
```bash
# 尝试重启 Ray
ray stop
python train_async.py
```

### CUDA 序列化错误
已修复。如果遇到相关问题，确保 Workers 使用 CPU 设备，Trainer 使用 GPU/CPU 自动检测。

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

### 缓冲区填充太慢
```bash
# 增加初始收集量
python train_async.py --warmup_seconds 10 --num_workers 4
```
