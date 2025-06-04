import pygame
import numpy as np
from typing import Tuple, Optional

class BasketballEnv:
    def __init__(self):
        # 物理参数
        self.gravity = 10.0  # 重力加速度
        self.dt = 0.02      # 时间步长
        self.max_steps = 200  # 每回合最大步数
        
        # 游戏参数
        self.player_pos = np.array([0.0, 0.0])  # 玩家位置
        self.target_distance = 10.0  # 目标距离
        self.target_radius = 1.0     # 目标半径
        self.max_force = 20.0        # 最大投掷力度
        self.target_pos = None       # 目标位置
        
        # 显示参数
        self.screen_width = 800
        self.screen_height = 600
        self.scale = 30  # 物理单位到像素的转换比例
        self.screen = None
        self.clock = None
        
        # 状态参数
        self.ball_pos = None
        self.ball_vel = None
        self.steps = 0
        
        self.reset()
        
    def _init_pygame(self):
        """初始化Pygame"""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Basketball DQN")
            self.clock = pygame.time.Clock()
    
    def reset(self) -> np.ndarray:
        """重置环境状态"""
        # 在第一象限、距离15以内随机生成目标位置
        while True:
            # 随机生成x和y坐标
            x = np.random.uniform(0, 15)
            y = np.random.uniform(0, 15)
            # 检查是否在15单位距离内
            if np.sqrt(x*x + y*y) <= 15:
                self.target_pos = np.array([x, y])
                break
        
        # 重置球的位置和速度
        self.ball_pos = self.player_pos.copy()
        self.ball_vel = np.zeros(2)
        self.steps = 0
        
        return self._get_state()
    
    def step(self, action: Tuple[float, float]) -> Tuple[np.ndarray, float, bool, dict]:
        """执行一步动作
        
        Args:
            action: (angle, force) 投球角度（弧度）和力度
            
        Returns:
            state: 新的状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        angle, force = action
        force = min(force, self.max_force)  # 限制最大力度
        
        # 计算初始速度
        self.ball_vel = np.array([
            force * np.cos(angle),
            force * np.sin(angle)
        ])
        
        done = False
        reward = 0
        
        # 模拟球的运动
        while True:
            self.steps += 1
            
            # 更新位置和速度
            self.ball_pos = self.ball_pos + self.ball_vel * self.dt
            self.ball_vel[1] = self.ball_vel[1] - self.gravity * self.dt
            
            # 检查是否进球
            dist_to_target = np.linalg.norm(self.ball_pos - self.target_pos)
            if dist_to_target <= self.target_radius:
                reward = 1.0
                done = True
                break
            
            # 检查是否出界或超时
            if (self.ball_pos[1] < 0 or 
                self.ball_pos[0] > 15 * 1.5 or  # 调整出界检查范围
                self.ball_pos[0] < 0 or         # 增加左边界检查
                self.ball_pos[1] > 15 * 1.5 or  # 增加上边界检查
                self.steps >= self.max_steps):
                reward = -1.0
                done = True
                break
                
        return self._get_state(), reward, done, {}
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        return np.array([
            self.target_pos[0],
            self.target_pos[1]
        ])
    
    def render(self):
        """渲染当前状态"""
        self._init_pygame()
        self.screen.fill((255, 255, 255))
        
        # 坐标转换函数
        def to_screen(pos):
            return (
                int(pos[0] * self.scale + self.screen_width // 2),
                int(self.screen_height - (pos[1] * self.scale + 100))
            )
        
        # 绘制玩家
        player_screen_pos = to_screen(self.player_pos)
        pygame.draw.circle(self.screen, (0, 0, 255), player_screen_pos, 10)
        
        # 绘制目标
        target_screen_pos = to_screen(self.target_pos)
        pygame.draw.circle(self.screen, (255, 0, 0), target_screen_pos, 
                         int(self.target_radius * self.scale), 2)
        
        # 绘制球
        if self.ball_pos is not None:
            ball_screen_pos = to_screen(self.ball_pos)
            pygame.draw.circle(self.screen, (0, 255, 0), ball_screen_pos, 5)
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None