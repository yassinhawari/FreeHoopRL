# 投篮强化学习项目

这个项目使用深度Q网络（DQN）算法来训练一个AI代理，没有考虑空气阻力，仅用于算法理解学习如何在2D环境中投篮。代理需要学习选择合适的投篮角度和力度，使篮球能够准确地进入篮筐。
## 运行结果
![final_policy](https://github.com/user-attachments/assets/90f0fd60-123b-4a60-97b3-809816ef127c)
![final_analysis](https://github.com/user-attachments/assets/8fbf3d5b-4e9d-4ade-adc8-1e962b7a4a0d)

## 项目结构

```
.
├── env.py          # 篮球投篮环境
├── dqn.py          # DQN代理实现
├── main.py         # 训练和评估脚本
├── utils.py        # 辅助函数（轨迹计算、可视化等）
└── models/         # 保存训练模型的目录
```

## 环境说明

篮球投篮环境是一个简单的2D物理环境：

- 玩家位于固定位置
- 目标（篮筐）位置随机生成
- 代理需要选择投篮角度和力度
- 如果球进入篮筐，获得正奖励；否则获得负奖励
- 奖励大小与球和篮筐中心的距离相关

环境参数：
- 重力加速度：10.0
- 时间步长：0.02
- 篮筐半径：1.0

## 安装依赖

```bash
pip install numpy torch matplotlib
```

## 使用方法

### 训练模型

```bash
python main.py --mode train --episodes 1000 --save-dir models
```

### 测试模型

```bash
python main.py --mode test --model-path models/best_model.pth
```

### 训练并测试

```bash
python main.py --mode both --episodes 500
```

## 命令行参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式：train, test, both | train |
| `--episodes` | 训练回合数 | 1000 |
| `--eval-episodes` | 评估回合数 | 10 |
| `--render-interval` | 训练时渲染间隔 | 100 |
| `--save-interval` | 保存模型间隔 | 100 |
| `--model-path` | 测试模式下加载的模型路径 | models/best_model.pth |
| `--save-dir` | 保存模型的目录 | models |
| `--lr` | 学习率 | 0.001 |
| `--gamma` | 折扣因子 | 0.99 |
| `--epsilon-start` | 起始探索率 | 1.0 |
| `--epsilon-end` | 最终探索率 | 0.01 |
| `--epsilon-decay` | 探索率衰减 | 0.995 |

## 训练过程

训练过程中，代理会通过试错学习如何投篮。训练脚本会：

1. 定期保存模型检查点
2. 保存最佳性能的模型
3. 生成训练分析图表
4. 可视化代理的策略

## 可视化

项目提供了多种可视化工具：

1. **训练奖励分析**：显示训练过程中的奖励和成功率
2. **策略可视化**：显示代理在不同目标位置的投篮轨迹
3. **轨迹绘制**：可视化单次投篮的轨迹

## 示例输出

训练完成后，你可以在`models`目录中找到：

- `best_model.pth`：性能最好的模型
- `final_model.pth`：最终训练的模型
- `analysis_*.png`：训练过程分析图
- `policy_*.png`：策略可视化图
