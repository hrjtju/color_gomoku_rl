"""
游戏可视化界面
支持人机对战和AI演示
"""

import os
import sys
import argparse
import numpy as np
import pygame
from pygame.locals import *

from game_env import ColorGomokuEnv
from ppo_agent import PPOAgent


# 颜色定义
COLORS = {
    'background': (240, 240, 240),
    'grid': (100, 100, 100),
    'text': (50, 50, 50),
    'highlight': (255, 255, 0, 128),
    'selected': (0, 255, 0, 128),
}

# 棋子颜色
PIECE_COLORS = {
    0: (0, 0, 139),       # 深蓝
    1: (135, 206, 235),   # 浅蓝
    2: (216, 191, 216),   # 浅紫
    3: (240, 128, 128),   # 浅红
    4: (255, 255, 0),     # 亮黄色
    5: (0, 128, 0),       # 绿色
    6: (139, 69, 19),     # 咖啡
}

COLOR_NAMES = {
    0: "深蓝",
    1: "浅蓝",
    2: "浅紫",
    3: "浅红",
    4: "亮黄色",
    5: "绿色",
    6: "咖啡",
}


class GameVisualizer:
    """游戏可视化器"""
    
    def __init__(self, env: ColorGomokuEnv, cell_size: int = 60):
        self.env = env
        self.cell_size = cell_size
        self.board_size = env.board_size
        
        # 计算窗口大小
        self.margin = 50
        self.info_height = 150
        self.width = self.board_size * cell_size + 2 * self.margin
        self.height = self.board_size * cell_size + 2 * self.margin + self.info_height
        
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("彩色五子棋 - Color Gomoku")
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # 游戏状态
        self.selected_cell = None
        self.hovered_cell = None
        self.running = True
        
    def draw_board(self) -> None:
        """绘制棋盘"""
        self.screen.fill(COLORS['background'])
        
        # 绘制网格
        board_left = self.margin
        board_top = self.margin
        board_width = self.board_size * self.cell_size
        board_height = self.board_size * self.cell_size
        
        # 绘制棋盘背景
        pygame.draw.rect(self.screen, (222, 184, 135), 
                        (board_left, board_top, board_width, board_height))
        
        # 绘制网格线
        for i in range(self.board_size + 1):
            # 横线
            pygame.draw.line(self.screen, COLORS['grid'],
                           (board_left, board_top + i * self.cell_size),
                           (board_left + board_width, board_top + i * self.cell_size), 2)
            # 竖线
            pygame.draw.line(self.screen, COLORS['grid'],
                           (board_left + i * self.cell_size, board_top),
                           (board_left + i * self.cell_size, board_top + board_height), 2)
        
        # 绘制棋子
        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = self.env.board[r, c]
                if piece != self.env.EMPTY:
                    x = board_left + c * self.cell_size + self.cell_size // 2
                    y = board_top + r * self.cell_size + self.cell_size // 2
                    radius = self.cell_size // 2 - 4
                    
                    color = PIECE_COLORS[piece]
                    pygame.draw.circle(self.screen, color, (x, y), radius)
                    pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 2)
        
        # 高亮选中的格子
        if self.selected_cell:
            r, c = self.selected_cell
            x = board_left + c * self.cell_size
            y = board_top + r * self.cell_size
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            s.fill(COLORS['selected'])
            self.screen.blit(s, (x, y))
        
        # 高亮悬停的格子
        if self.hovered_cell and self.hovered_cell != self.selected_cell:
            r, c = self.hovered_cell
            x = board_left + c * self.cell_size
            y = board_top + r * self.cell_size
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            s.fill(COLORS['highlight'])
            self.screen.blit(s, (x, y))
    
    def draw_info(self) -> None:
        """绘制信息面板"""
        info_top = self.margin + self.board_size * self.cell_size + 20
        
        # 分数
        score_text = self.font.render(f"Score: {self.env.score}", True, COLORS['text'])
        self.screen.blit(score_text, (self.margin, info_top))
        
        # 步数
        step_text = self.font.render(f"Steps: {self.env.step_count}", True, COLORS['text'])
        self.screen.blit(step_text, (self.margin + 200, info_top))
        
        # 即将出现的棋子
        upcoming_text = self.font.render("Upcoming:", True, COLORS['text'])
        self.screen.blit(upcoming_text, (self.margin, info_top + 40))
        
        for i, color in enumerate(self.env.upcoming_pieces):
            x = self.margin + 120 + i * 50
            y = info_top + 45
            pygame.draw.circle(self.screen, PIECE_COLORS[color], (x, y), 18)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 18, 2)
        
        # 操作提示
        hint_text = self.small_font.render(
            "Click a cell with ball to select, then click a reachable empty cell to move", 
            True, COLORS['text'])
        self.screen.blit(hint_text, (self.margin, info_top + 80))
        
        # 空格数
        empty_cells = np.sum(self.env.board == self.env.EMPTY)
        empty_text = self.font.render(f"Empty: {empty_cells}", True, COLORS['text'])
        self.screen.blit(empty_text, (self.margin + 350, info_top))
    
    def get_cell_from_pos(self, pos: tuple) -> tuple:
        """从鼠标位置获取格子坐标"""
        x, y = pos
        board_left = self.margin
        board_top = self.margin
        
        if (board_left <= x < board_left + self.board_size * self.cell_size and
            board_top <= y < board_top + self.board_size * self.cell_size):
            col = (x - board_left) // self.cell_size
            row = (y - board_top) // self.cell_size
            return (row, col)
        return None
    
    def handle_click(self, pos: tuple) -> bool:
        """处理点击事件"""
        cell = self.get_cell_from_pos(pos)
        
        if cell is None:
            return False
        
        row, col = cell
        
        # 如果没有选中起始位置
        if self.selected_cell is None:
            # 只能选有球的cell作为起始位置
            if self.env.board[row, col] != self.env.EMPTY:
                self.selected_cell = cell
            return False
        else:
            # 已选中起始位置，现在选择目标位置
            from_pos = self.selected_cell
            to_pos = cell
            
            # 目标位置必须是空格
            if self.env.board[row, col] != self.env.EMPTY:
                # 如果点击的是另一个有球的cell，将其作为新的起始位置
                self.selected_cell = cell
                return False
            
            # 检查是否可达
            if self.env._is_reachable(from_pos, to_pos):
                action = self.env.encode_action(from_pos, to_pos)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.selected_cell = None
                
                if terminated or truncated:
                    return True
            else:
                # 不可达，取消选择
                self.selected_cell = None
        
        return False
    
    def play_human(self) -> None:
        """人类玩家游戏循环"""
        clock = pygame.time.Clock()
        game_over = False
        
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                    return
                
                elif event.type == MOUSEMOTION:
                    self.hovered_cell = self.get_cell_from_pos(event.pos)
                
                elif event.type == MOUSEBUTTONDOWN:
                    if not game_over:
                        game_over = self.handle_click(event.pos)
                        if game_over:
                            print(f"Game Over! Final Score: {self.env.score}")
                
                elif event.type == KEYDOWN:
                    if event.key == K_r:
                        # 重新开始
                        self.env.reset()
                        self.selected_cell = None
                        game_over = False
                    elif event.key == K_ESCAPE:
                        self.running = False
                        return
            
            self.draw_board()
            self.draw_info()
            
            if game_over:
                game_over_text = self.font.render(
                    f"Game Over! Score: {self.env.score} - Press R to restart", 
                    True, (255, 0, 0))
                text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2))
                self.screen.blit(game_over_text, text_rect)
            
            pygame.display.flip()
            clock.tick(60)
    
    def play_ai(self, agent: PPOAgent, delay: float = 0.5) -> None:
        """AI玩家游戏循环"""
        clock = pygame.time.Clock()
        
        obs, info = self.env.reset()
        game_over = False
        
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                    return
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.running = False
                        return
                    elif event.key == K_r:
                        obs, info = self.env.reset()
                        game_over = False
                    elif event.key == K_SPACE:
                        delay = 0.1 if delay > 0.1 else 0.5
            
            if not game_over:
                # AI选择动作
                valid_actions = self.env.get_valid_actions()
                
                if len(valid_actions) > 0:
                    action, _, _ = agent.select_action(obs, valid_actions, deterministic=True)
                    from_pos, to_pos = self.env.decode_action(action)
                    
                    # 高亮AI的移动
                    self.selected_cell = from_pos
                    self.draw_board()
                    self.draw_info()
                    pygame.display.flip()
                    pygame.time.wait(int(delay * 500))
                    
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    
                    if reward > 0:
                        print(f"AI got reward: {reward}")
                    
                    game_over = terminated or truncated
                    
                    if game_over:
                        print(f"AI Game Over! Final Score: {self.env.score}")
                else:
                    print("No valid actions!")
                    game_over = True
            
            self.selected_cell = None
            self.draw_board()
            self.draw_info()
            
            if game_over:
                game_over_text = self.font.render(
                    f"Game Over! Score: {self.env.score} - Press R to restart", 
                    True, (255, 0, 0))
                text_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 2))
                self.screen.blit(game_over_text, text_rect)
            
            pygame.display.flip()
            pygame.time.wait(int(delay * 1000))
            clock.tick(60)
    
    def close(self) -> None:
        """关闭可视化"""
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Play Color Gomoku')
    parser.add_argument('--mode', type=str, default='human', choices=['human', 'ai'],
                       help='Game mode: human or ai')
    parser.add_argument('--model_path', type=str, default='./models/best_model.pt',
                       help='Path to AI model')
    parser.add_argument('--cell_size', type=int, default=60, help='Cell size in pixels')
    parser.add_argument('--delay', type=float, default=0.5, help='AI move delay in seconds')
    
    args = parser.parse_args()
    
    # 创建环境
    env = ColorGomokuEnv()
    env.reset()
    
    # 创建可视化器
    visualizer = GameVisualizer(env, cell_size=args.cell_size)
    
    try:
        if args.mode == 'human':
            print("人类模式")
            print("操作说明：")
            print("- 点击有球的格子选择起始位置")
            print("- 再点击一个可达的空格进行移动")
            print("- 按 R 重新开始")
            print("- 按 ESC 退出")
            visualizer.play_human()
        else:
            print("AI模式")
            if not os.path.exists(args.model_path):
                print(f"错误：找不到模型文件 {args.model_path}")
                print("请先训练模型：python train.py")
                return
            
            # 加载AI
            agent = PPOAgent(board_size=env.board_size, num_colors=env.num_colors)
            agent.load(args.model_path)
            print(f"已加载模型：{args.model_path}")
            print("操作说明：")
            print("- 按 R 重新开始")
            print("- 按 SPACE 加速/减速")
            print("- 按 ESC 退出")
            
            visualizer.play_ai(agent, delay=args.delay)
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
