import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def calculate_trajectory(start_pos: np.ndarray, angle: float, force: float, 
                        gravity: float = 10.0, dt: float = 0.02, 
                        max_steps: int = 500) -> List[np.ndarray]:
    """计算投球轨迹
    
    Args:
        start_pos: 起始位置 [x, y]
        angle: 投球角度（弧度）
        force: 投球力度
        gravity: 重力加速度
        dt: 时间步长
        max_steps: 最大步数
        
    Returns:
        trajectory: 轨迹点列表
    """
    pos = start_pos.copy()
    vel = np.array([
        force * np.cos(angle),
        force * np.sin(angle)
    ])
    
    trajectory = [pos.copy()]
    
    for _ in range(max_steps):
        # 更新位置
        pos = pos + vel * dt
        
        # 更新速度（只有y方向受重力影响）
        vel[1] = vel[1] - gravity * dt
        
        trajectory.append(pos.copy())
        
        # 如果球落地，停止模拟
        if pos[1] < 0:
            break
    
    return trajectory

def plot_trajectory(trajectory: List[np.ndarray], target_pos: np.ndarray, 
                   target_radius: float, save_path: str = None):
    """绘制轨迹
    
    Args:
        trajectory: 轨迹点列表
        target_pos: 目标位置
        target_radius: 目标半径
        save_path: 保存路径（如果为None则显示图像）
    """
    trajectory = np.array(trajectory)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制轨迹
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Ball Trajectory')
    
    # 绘制起点
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    
    # 绘制目标
    circle = plt.Circle((target_pos[0], target_pos[1]), target_radius, 
                       fill=False, color='r', linewidth=2, label='Target')
    plt.gca().add_patch(circle)
    
    # 绘制地面
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlabel('X Distance')
    plt.ylabel('Height')
    plt.title('Basketball Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_training(rewards: List[float], window_size: int = 100, 
                    save_path: str = 'training_analysis.png'):
    """分析训练过程
    
    Args:
        rewards: 奖励列表
        window_size: 滑动窗口大小
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    
    # 创建子图
    plt.subplot(2, 1, 1)
    plt.plot(rewards, alpha=0.3, label='Raw')
    
    # 计算滑动平均
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
        plt.plot(range(window_size-1, len(rewards)), 
                moving_avg, 
                label=f'Moving Average ({window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True)
    
    # 计算成功率（奖励为正的比例）
    plt.subplot(2, 1, 2)
    success_rate = [sum(1 for r in rewards[:i+1] if r > 0) / (i+1) 
                   for i in range(len(rewards))]
    plt.plot(success_rate, 'g-')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Training Success Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_policy(agent, env, num_samples: int = 10, 
                    save_path: str = 'policy_visualization.png'):
    """可视化策略
    
    Args:
        agent: DQN代理
        env: 游戏环境
        num_samples: 样本数量
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    
    # 保存原始epsilon
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # 关闭探索
    
    for i in range(num_samples):
        state = env.reset()
        target_pos = env.target_pos
        
        # 获取代理选择的动作
        angle, force = agent.select_action(state)
        
        # 计算轨迹
        trajectory = calculate_trajectory(
            env.player_pos, angle, force, env.gravity, env.dt
        )
        trajectory = np.array(trajectory)
        
        # 绘制轨迹
        plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.7, 
               label=f'Sample {i+1}')
        
        # 绘制目标
        circle = plt.Circle((target_pos[0], target_pos[1]), env.target_radius, 
                           fill=False, color='r', linewidth=1)
        plt.gca().add_patch(circle)
    
    # 恢复epsilon
    agent.epsilon = original_epsilon
    
    # 绘制地面
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlabel('X Distance')
    plt.ylabel('Height')
    plt.title('Policy Visualization')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()