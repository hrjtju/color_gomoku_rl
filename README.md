# 彩色五子棋 + PPO强化学习智能体

这是一个基于PPO（Proximal Policy Optimization）算法的彩色五子棋游戏项目。

## 游戏规则

1. **棋盘大小**: 9x9
2. **棋子颜色**: 深蓝、浅蓝、浅紫、浅红、亮黄色、绿色、咖啡（共7种）
3. **消除规则**: 横向、竖向或对角线有五个及以上连续相同颜色棋子消去
   - 消去5个：+10分
   - 消去5个以上：每多一个+4分
4. **移动规则**: 将棋子从一个空格移动至另一个可达的空格（通过上下左右移动）
5. **添加规则**: 
   - 移动后不触发消除：随机向棋盘空位中放置3个随机颜色的棋子
   - 移动后触发消除：不添加新棋子
   - 这三个棋子的颜色会在玩家做出决策之前呈现
6. **结束条件**: 棋盘已满，游戏结束

## 项目结构

```
color_gomoku_rl/
├── game_env.py       # 游戏环境（Gymnasium）
├── ppo_agent.py      # PPO智能体实现
├── train.py          # 训练脚本
├── play.py           # 游戏可视化界面
├── demo.py           # 演示脚本
├── evaluate.py       # 模型评估脚本
├── plot_training.py  # 训练进度可视化
├── requirements.txt  # 依赖包
├── README.md         # 说明文档
├── QUICKSTART.md     # 快速开始指南
└── logs/             # 训练日志目录
└── models/           # 模型保存目录
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练智能体

```bash
# 基本训练
python train.py

# 自定义参数
python train.py --num_episodes 20000 --lr 1e-4 --hidden_dim 512

# 查看所有参数
python train.py --help
```

训练参数：
- `--num_episodes`: 训练回合数（默认：10000）
- `--update_interval`: 网络更新间隔（默认：2048）
- `--eval_interval`: 评估间隔（默认：100）
- `--hidden_dim`: 隐藏层维度（默认：256）
- `--lr`: 学习率（默认：3e-4）
- `--gamma`: 折扣因子（默认：0.99）
- `--entropy_coef`: 熵系数（默认：0.01）

### 2. 玩游戏

#### 人类玩家
```bash
python play.py --mode human
```

操作说明：
- 点击空格选择起始位置
- 再点击另一个可达的空格进行移动
- 按 R 重新开始
- 按 ESC 退出

#### AI玩家
```bash
python play.py --mode ai --model_path ./models/best_model.pt
```

操作说明：
- 按 R 重新开始
- 按 SPACE 加速/减速
- 按 ESC 退出

### 3. 演示

```bash
# 运行演示（随机AI + 消除检测测试）
python demo.py
```

### 4. 评估模型

```bash
# 评估训练好的模型
python evaluate.py --model_path ./models/best_model.pt --num_episodes 100

# 保存评估结果
python evaluate.py --model_path ./models/best_model.pt --output eval_results.json
```

### 5. 可视化训练进度

```bash
# 显示训练图表
python plot_training.py

# 只显示摘要
python plot_training.py --summary

# 保存图表
python plot_training.py --save training_progress.png
```

### 6. 测试环境

```bash
# 测试游戏环境
python game_env.py

# 测试PPO智能体
python ppo_agent.py
```

## PPO智能体设计

### 状态空间
- **棋盘状态**: 9x9矩阵，-1表示空格，0-6表示不同颜色
- **即将出现的棋子**: 3个整数值，表示接下来要添加的3个棋子的颜色

### 动作空间
- 动作编码: `(from_row, from_col, to_row, to_col)`
- 离散化: `action = ((from_row * 9 + from_col) * 81 + (to_row * 9 + to_col))`
- 总动作数: 9^4 = 6561

### 网络架构
- **棋盘编码器**: CNN (Conv2d layers) 提取棋盘特征
- **即将出现的棋子编码器**: MLP编码
- **Actor（策略网络）**: 输出动作概率分布
- **Critic（价值网络）**: 输出状态价值估计

### 奖励设计
- 直接奖励: 消除得分（5个=10分，每多一个+4分）
- 无效动作惩罚: -10分

## 训练过程

训练过程中会记录以下指标：
- 回合奖励（Reward）
- 回合得分（Score）
- 回合长度（Length）
- 策略损失（Policy Loss）
- 价值损失（Value Loss）
- 熵（Entropy）

模型会自动保存在 `./models/` 目录：
- `best_model.pt`: 最佳模型（基于评估得分）
- `final_model.pt`: 最终模型
- `model_episode_XXX.pt`: 定期保存的模型

训练日志保存在 `./logs/` 目录：
- `train.log`: 训练日志
- `metrics.jsonl`: 训练指标（JSON格式）

## 可视化

使用Pygame实现可视化界面：
- 彩色棋子显示
- 鼠标悬停高亮
- 选中格子高亮
- 实时分数和步数显示
- 即将出现的棋子预览

## 算法特点

1. **PPO算法**: 使用裁剪的目标函数，训练稳定
2. **GAE（Generalized Advantage Estimation）**: 用于优势函数估计
3. **CNN特征提取**: 有效提取棋盘空间特征
4. **有效动作掩码**: 只选择合法的动作
5. **经验回放**: 收集多步经验后批量更新

## 扩展建议

1. **改进奖励函数**: 
   - 添加潜在消除奖励
   - 添加棋盘评估奖励
   - 添加生存时间奖励

2. **网络架构改进**:
   - 使用ResNet块
   - 添加注意力机制
   - 使用Transformer编码器

3. **训练技巧**:
   - 课程学习（逐渐增加难度）
   - 自对弈训练
   - 多智能体训练

4. **游戏变体**:
   - 不同棋盘大小
   - 不同颜色数量
   - 不同消除规则

## 许可证

MIT License
