"""
彩色五子棋游戏环境
基于Gymnasium框架实现
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Dict, Optional
from collections import deque


class ColorGomokuEnv(gym.Env):
    """
    彩色五子棋环境
    
    规则:
    1. 9x9 棋盘
    2. 7种颜色: 深蓝(0)、浅蓝(1)、浅紫(2)、浅红(3)、亮黄色(4)、绿色(5)、咖啡(6)
    3. 五子连珠消除，5个得10分，每多一个加4分
    4. 棋子从一个空格移动到另一个可达的空格（上下左右）
    5. 不触发消除则随机放3个棋子（颜色提前告知）
    6. 棋盘满则游戏结束
    """
    
    # 颜色定义
    COLORS = {
        0: "深蓝",
        1: "浅蓝", 
        2: "浅紫",
        3: "浅红",
        4: "亮黄色",
        5: "绿色",
        6: "咖啡"
    }
    
    # 颜色RGB值（用于可视化）
    COLOR_RGB = {
        0: (0, 0, 139),      # 深蓝
        1: (135, 206, 235),  # 浅蓝
        2: (216, 191, 216),  # 浅紫
        3: (240, 128, 128),  # 浅红
        4: (255, 255, 0),    # 亮黄色
        5: (0, 128, 0),      # 绿色
        6: (139, 69, 19),    # 咖啡
    }
    
    EMPTY = -1  # 空格子
    
    def __init__(self, board_size: int = 9, num_colors: int = 7, use_shaped_reward: bool = True):
        super().__init__()
        
        self.board_size = board_size
        self.num_colors = num_colors
        self.use_shaped_reward = use_shaped_reward
        
        # 动作空间: (from_row, from_col, to_row, to_col)
        # 使用离散动作空间编码: action = ((from_row * 9 + from_col) * 81 + (to_row * 9 + to_col))
        self.action_space = spaces.Discrete(board_size ** 4)
        
        # 观察空间: 棋盘状态 (9x9) + 即将添加的3个棋子颜色 (3,)
        # 棋盘: -1表示空，0-6表示颜色
        self.observation_space = spaces.Dict({
            'board': spaces.Box(
                low=-1, 
                high=num_colors - 1, 
                shape=(board_size, board_size), 
                dtype=np.int32
            ),
            'upcoming': spaces.Box(
                low=0, 
                high=num_colors - 1, 
                shape=(3,), 
                dtype=np.int32
            )
        })
        
        self.board = None
        self.upcoming_pieces = None
        self.score = 0
        self.step_count = 0
        self.max_steps = 1000  # 最大步数限制
        self.last_potential = 0  # 上一步的潜在函数值（用于奖励塑形）
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化空棋盘
        self.board = np.full((self.board_size, self.board_size), self.EMPTY, dtype=np.int32)
        
        # 随机放置一些初始棋子（约30%的格子）
        num_initial = int(self.board_size * self.board_size * 0.3)
        empty_positions = [(r, c) for r in range(self.board_size) for c in range(self.board_size)]
        np.random.shuffle(empty_positions)
        
        for i in range(min(num_initial, len(empty_positions))):
            r, c = empty_positions[i]
            self.board[r, c] = np.random.randint(0, self.num_colors)
        
        # 生成即将添加的3个棋子
        self.upcoming_pieces = self._generate_upcoming()
        
        self.score = 0
        self.step_count = 0
        self.last_potential = 0  # 重置潜在函数
        
        # 计算初始潜在函数（用于奖励塑形）
        if self.use_shaped_reward:
            self.last_potential = self._compute_potential()
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _generate_upcoming(self) -> np.ndarray:
        """生成即将添加的3个随机颜色棋子"""
        return np.random.randint(0, self.num_colors, size=3)
    
    def _get_obs(self) -> Dict:
        """获取当前观察"""
        return {
            'board': self.board.copy(),
            'upcoming': self.upcoming_pieces.copy()
        }
    
    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            'score': self.score,
            'step_count': self.step_count,
            'empty_cells': np.sum(self.board == self.EMPTY)
        }
    
    def decode_action(self, action: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """将离散动作解码为(from_pos, to_pos)"""
        action = int(action)
        from_idx = action // (self.board_size ** 2)
        to_idx = action % (self.board_size ** 2)
        
        from_row = from_idx // self.board_size
        from_col = from_idx % self.board_size
        to_row = to_idx // self.board_size
        to_col = to_idx % self.board_size
        
        return (from_row, from_col), (to_row, to_col)
    
    def encode_action(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """将(from_pos, to_pos)编码为离散动作"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        from_idx = from_row * self.board_size + from_col
        to_idx = to_row * self.board_size + to_col
        return from_idx * (self.board_size ** 2) + to_idx
    
    def _is_valid_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """检查移动是否有效"""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # 检查边界
        if not (0 <= from_row < self.board_size and 0 <= from_col < self.board_size):
            return False
        if not (0 <= to_row < self.board_size and 0 <= to_col < self.board_size):
            return False
        
        # 起始位置必须有球（非空格）
        if self.board[from_row, from_col] == self.EMPTY:
            return False
        
        # 目标位置必须是空格
        if self.board[to_row, to_col] != self.EMPTY:
            return False
        
        # 目标位置必须可达（通过BFS检查连通性）
        if not self._is_reachable(from_pos, to_pos):
            return False
        
        return True
    
    def _is_reachable(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        """使用BFS检查from_pos是否可以通过空格到达to_pos"""
        if from_pos == to_pos:
            return False  # 起始位置和目标位置不能相同
        
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # 目标位置必须是空格
        if self.board[to_row, to_col] != self.EMPTY:
            return False
        
        # BFS - 从起始位置出发，通过空格路径找到目标位置
        visited = np.zeros((self.board_size, self.board_size), dtype=bool)
        queue = deque([from_pos])
        visited[from_row, from_col] = True
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            r, c = queue.popleft()
            
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if not visited[nr, nc]:
                        # 如果是目标位置，返回True
                        if (nr, nc) == (to_row, to_col):
                            return True
                        # 只能通过空格移动
                        if self.board[nr, nc] == self.EMPTY:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
        
        return False
    
    def _find_matches(self) -> List[List[Tuple[int, int]]]:
        """找到所有五子连珠的匹配"""
        matches = []
        visited = set()
        
        directions = [
            (0, 1),   # 横向
            (1, 0),   # 竖向
            (1, 1),   # 对角线
            (1, -1)   # 反对角线
        ]
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == self.EMPTY:
                    continue
                
                color = self.board[r, c]
                
                for dr, dc in directions:
                    # 检查这个方向是否有5个或更多连续同色棋子
                    line = [(r, c)]
                    
                    # 正向检查
                    nr, nc = r + dr, c + dc
                    while (0 <= nr < self.board_size and 
                           0 <= nc < self.board_size and 
                           self.board[nr, nc] == color):
                        line.append((nr, nc))
                        nr += dr
                        nc += dc
                    
                    # 反向检查
                    nr, nc = r - dr, c - dc
                    while (0 <= nr < self.board_size and 
                           0 <= nc < self.board_size and 
                           self.board[nr, nc] == color):
                        line.append((nr, nc))
                        nr -= dr
                        nc -= dc
                    
                    # 如果长度>=5，添加到匹配
                    if len(line) >= 5:
                        line_sorted = tuple(sorted(line))
                        if line_sorted not in visited:
                            visited.add(line_sorted)
                            matches.append(line)
        
        return matches
    
    def _calculate_score(self, matches: List[List[Tuple[int, int]]]) -> int:
        """计算消除得分"""
        total_score = 0
        for match in matches:
            num_pieces = len(match)
            if num_pieces >= 5:
                total_score += num_pieces * 2.0
        return total_score
    
    def _count_aligned_pieces(self) -> Dict[int, int]:
        """
        统计各种长度的连子数（用于奖励塑形）
        返回: {长度: 该长度的连线数量}
        """
        aligned_counts = {2: 0, 3: 0, 4: 0}
        visited = set()
        
        directions = [
            (0, 1),   # 横向
            (1, 0),   # 竖向
            (1, 1),   # 对角线
            (1, -1)   # 反对角线
        ]
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == self.EMPTY:
                    continue
                
                color = self.board[r, c]
                
                for dr, dc in directions:
                    # 只在每个连线的起点检查（避免重复计数）
                    nr, nc = r - dr, c - dc
                    if (0 <= nr < self.board_size and 
                        0 <= nc < self.board_size and 
                        self.board[nr, nc] == color):
                        continue  # 不是起点，跳过
                    
                    # 从起点开始计算连线长度
                    line = [(r, c)]
                    nr, nc = r + dr, c + dc
                    while (0 <= nr < self.board_size and 
                           0 <= nc < self.board_size and 
                           self.board[nr, nc] == color):
                        line.append((nr, nc))
                        nr += dr
                        nc += dc
                    
                    line_len = len(line)
                    if line_len in aligned_counts:
                        aligned_counts[line_len] += 1
        
        return aligned_counts
    
    def _compute_potential(self) -> float:
        """
        计算潜在函数（用于奖励塑形）
        基于当前棋盘状态评估形成五子连珠的潜力
        """
        aligned = self._count_aligned_pieces()
        # 潜在值：2连子权重0.1，3连子权重0.5，4连子权重2.0
        potential = aligned[2] * 0.1 + aligned[3] * 0.5 + aligned[4] * 2.0
        return potential
    
    def _remove_matches(self, matches: List[List[Tuple[int, int]]]) -> None:
        """移除匹配的棋子"""
        for match in matches:
            for r, c in match:
                self.board[r, c] = self.EMPTY
    
    def _add_upcoming_pieces(self) -> None:
        """添加即将出现的棋子到棋盘"""
        empty_positions = [(r, c) for r in range(self.board_size) 
                          for c in range(self.board_size) 
                          if self.board[r, c] == self.EMPTY]
        
        if len(empty_positions) >= 3:
            np.random.shuffle(empty_positions)
            for i, color in enumerate(self.upcoming_pieces):
                r, c = empty_positions[i]
                self.board[r, c] = color
        
        # 生成新的即将出现的棋子
        self.upcoming_pieces = self._generate_upcoming()
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        执行一步动作
        
        Returns:
            observation: 观察
            reward: 奖励
            terminated: 是否结束
            truncated: 是否截断
            info: 额外信息
        """
        self.step_count += 1
        
        from_pos, to_pos = self.decode_action(action)
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # 检查动作是否有效
        if not self._is_valid_move(from_pos, to_pos):
            # 无效动作，给予惩罚
            obs = self._get_obs()
            info = self._get_info()
            info['invalid'] = True
            return obs, -10.0, False, False, info
        
        # 执行移动
        self.board[to_row, to_col] = self.board[from_row, from_col]
        self.board[from_row, from_col] = self.EMPTY
        
        # 检查匹配
        matches = self._find_matches()
        
        reward = 0.0
        
        shaped_reward = 0.0  # 塑形奖励
        
        if matches:
            # 有匹配，计算得分
            score_gain = self._calculate_score(matches)
            self.score += score_gain
            reward = float(score_gain)
            
            # 额外奖励：成功消除是很大的成就
            shaped_reward += score_gain * 0.5  # 额外50%的塑形奖励
            
            # 移除匹配的棋子
            self._remove_matches(matches)
        else:
            # 没有匹配，添加即将出现的棋子
            self._add_upcoming_pieces()
            reward = 0.1  # 小奖励，鼓励继续游戏
        
        # 计算奖励塑形（使用潜在函数）
        if self.use_shaped_reward:
            current_potential = self._compute_potential()
            # 潜在函数变化作为奖励（新状态潜力 - 旧状态潜力）
            potential_diff = current_potential - self.last_potential
            shaped_reward += potential_diff
            self.last_potential = current_potential
        
        # 检查游戏是否结束
        empty_cells = np.sum(self.board == self.EMPTY)
        terminated = (empty_cells == 0)  # 棋盘满
        truncated = (self.step_count >= self.max_steps)  # 达到最大步数
        
        obs = self._get_obs()
        info = self._get_info()
        
        # 添加塑形奖励到 info
        if self.use_shaped_reward:
            # 塑形奖励 = 原始奖励 + 塑形部分
            total_shaped = reward + shaped_reward
            info['shaped_reward'] = total_shaped
            info['raw_reward'] = reward
            info['shaped_component'] = shaped_reward
        
        return obs, reward, terminated, truncated, info
    
    def get_valid_actions(self) -> List[int]:
        """获取所有有效的动作"""
        valid_actions = []
        
        # 起始位置：有球的cell
        piece_positions = [(r, c) for r in range(self.board_size) 
                          for c in range(self.board_size) 
                          if self.board[r, c] != self.EMPTY]
        
        # 目标位置：空格
        empty_positions = [(r, c) for r in range(self.board_size) 
                          for c in range(self.board_size) 
                          if self.board[r, c] == self.EMPTY]
        
        for from_pos in piece_positions:
            for to_pos in empty_positions:
                if self._is_reachable(from_pos, to_pos):
                    action = self.encode_action(from_pos, to_pos)
                    valid_actions.append(action)
        
        return valid_actions
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """渲染环境"""
        if mode == 'human':
            print(f"Score: {self.score}, Step: {self.step_count}")
            print("Upcoming:", [self.COLORS[c] for c in self.upcoming_pieces])
            print("-" * (self.board_size * 3 + 1))
            
            for r in range(self.board_size):
                row_str = "|"
                for c in range(self.board_size):
                    if self.board[r, c] == self.EMPTY:
                        row_str += " . |"
                    else:
                        row_str += f" {self.board[r, c]} |"
                print(row_str)
                print("-" * (self.board_size * 3 + 1))
        
        return None
    
    def close(self) -> None:
        """关闭环境"""
        pass


if __name__ == "__main__":
    # 测试环境
    env = ColorGomokuEnv()
    obs, info = env.reset(seed=42)
    
    print("Initial state:")
    env.render()
    
    # 测试一些随机动作
    for i in range(10):
        valid_actions = env.get_valid_actions()
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
            from_pos, to_pos = env.decode_action(action)
            print(f"\nStep {i+1}: Move from {from_pos} to {to_pos}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            print(f"Reward: {reward}, Terminated: {terminated}")
            
            if terminated or truncated:
                break
        else:
            print("No valid actions!")
            break
