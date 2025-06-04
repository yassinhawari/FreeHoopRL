import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
from env import BasketballEnv
from dqn import DQNAgent
from utils import plot_trajectory, analyze_training, visualize_policy
import time
from typing import List, Tuple

def train(env: BasketballEnv, agent: DQNAgent, episodes: int, 
         render_interval: int = 100, save_interval: int = 100,
         save_dir: str = "models") -> List[float]:
    """训练DQN代理
    
    Args:
        env: 游戏环境
        agent: DQN代理
        episodes: 训练回合数
        render_interval: 渲染间隔
        save_interval: 保存模型间隔
        save_dir: 保存模型的目录
        
    Returns:
        rewards: 每个episode的奖励列表
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    rewards = []
    best_reward = float('-inf')
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # 是否渲染这个回合
        should_render = (episode % render_interval == 0)
        
        while not done:
            if should_render:
                env.render()
                time.sleep(1/60)  # 控制帧率为60fps
            
            # 选择动作
            angle, force = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step((angle, force))
            
            # 计算动作索引（用于存储到经验回放）
            # 反向计算：从连续值转回离散索引
            angle_idx = min(int(angle / (np.pi/2) * agent.angle_bins), agent.angle_bins - 1)
            force_idx = min(int(force / 20 * agent.force_bins), agent.force_bins - 1)
            action_idx = angle_idx * agent.force_bins + force_idx
            
            # 存储转换到经验回放
            agent.memory.push(state, action_idx, reward, next_state, done)
            
            # 训练代理
            agent.train()  # 在经验回放中有足够样本时进行训练
            
            state = next_state
            episode_reward += reward
        
        rewards.append(episode_reward)
        
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            print(f"Episode {episode + 1}/{episodes}, "
                  f"Average Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.2f}")
            
            # 保存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(os.path.join(save_dir, "best_model.pth"))
            
            # 定期保存模型和分析结果
            if (episode + 1) % save_interval == 0:
                # 保存检查点
                agent.save(os.path.join(save_dir, f"checkpoint_{episode+1}.pth"))
                # 分析训练过程
                analyze_training(rewards, save_path=os.path.join(save_dir, f"analysis_{episode+1}.png"))
                # 可视化当前策略
                visualize_policy(agent, env, save_path=os.path.join(save_dir, f"policy_{episode+1}.png"))
    
    return rewards

def evaluate(env: BasketballEnv, agent: DQNAgent, episodes: int = 10, 
            render: bool = True) -> float:
    """评估代理的性能
    
    Args:
        env: 游戏环境
        agent: DQN代理
        episodes: 评估回合数
        render: 是否渲染
        
    Returns:
        avg_reward: 平均奖励
    """
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            if render:
                env.render()
                time.sleep(0.02)
            
            # 选择动作（评估时不使用探索）
            epsilon_backup = agent.epsilon
            agent.epsilon = 0
            angle, force = agent.select_action(state)
            agent.epsilon = epsilon_backup
            
            next_state, reward, done, _ = env.step((angle, force))
            state = next_state
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)

def plot_rewards(rewards: List[float], window_size: int = 100):
    """绘制奖励曲线
    
    Args:
        rewards: 奖励列表
        window_size: 滑动窗口大小
    """
    plt.figure(figsize=(10, 5))
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
    plt.grid()
    plt.savefig('training_rewards.png')
    plt.close()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Basketball DQN Training and Evaluation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'both'],
                        help='运行模式: train, test, or both')
    parser.add_argument('--episodes', type=int, default=1000, help='训练回合数')
    parser.add_argument('--eval-episodes', type=int, default=10, help='评估回合数')
    parser.add_argument('--render-interval', type=int, default=100, help='训练时渲染间隔')
    parser.add_argument('--save-interval', type=int, default=100, help='保存模型间隔')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth', help='模型路径')
    parser.add_argument('--save-dir', type=str, default='models', help='保存模型的目录')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='起始探索率')
    parser.add_argument('--epsilon-end', type=float, default=0.01, help='最终探索率')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='探索率衰减')
    
    args = parser.parse_args()
    
    # 创建环境和代理
    env = BasketballEnv()
    state_dim = 2  # 目标的x和y坐标
    action_dim = 200  # 20(角度) * 10(力度)
    agent = DQNAgent(
        state_dim, 
        action_dim, 
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_end,
        epsilon_decay=args.epsilon_decay
    )
    
    # 训练模式
    if args.mode in ['train', 'both']:
        print("Starting training...")
        rewards = train(
            env, 
            agent, 
            args.episodes, 
            render_interval=args.render_interval,
            save_interval=args.save_interval,
            save_dir=args.save_dir
        )
        
        # 保存最终模型和绘制奖励曲线
        final_model_path = os.path.join(args.save_dir, "final_model.pth")
        agent.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # 分析训练结果
        analyze_training(rewards, save_path=os.path.join(args.save_dir, "final_analysis.png"))
        print(f"Training analysis saved to {os.path.join(args.save_dir, 'final_analysis.png')}")
    
    # 评估模式
    if args.mode in ['test', 'both']:
        # 如果只是测试模式，加载预训练模型
        if args.mode == 'test':
            print(f"Loading model from {args.model_path}")
            agent.load(args.model_path)
        
        print("\nStarting evaluation...")
        avg_reward = evaluate(env, agent, episodes=args.eval_episodes, render=True)
        print(f"Average Evaluation Reward: {avg_reward:.2f}")
        
        # 可视化最终策略
        visualize_policy(agent, env, save_path=os.path.join(args.save_dir, "final_policy.png"))
        print(f"Policy visualization saved to {os.path.join(args.save_dir, 'final_policy.png')}")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()