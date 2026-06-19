# 五子棋 AI 训练系统 — 多架构 AlphaZero

一个基于 **AlphaZero + MCTS** 的五子棋强化学习训练系统，支持 **CNN v9.2 / v9.3 / Transformer** 三种神经网络架构的赛马对比。

## 快速开始

### 安装依赖

```bash
pip install torch numpy matplotlib tqdm
```

### 测试已有模型

```bash
# 测试 checkpoint (自动推断架构)
python test.py --model_path checkpoints/az_train/cnn_v3/best_model.pt

# 测试随机权重的某一架构
python test.py --arch transformer
```

### 人机对弈

```bash
python human_vs_ai.py --model checkpoints/az_train/cnn_v3/best_model.pt --color 1
```

---

## 神经网络架构

| 架构 | 主干 | 参数 | 价值头 | 特点 |
|------|------|:---:|------|------|
| **CNN v9.2** `cnn_v2` | 4×ResBlock(128) | ~124万 | Conv→GAP→FC | 经典稳定，需要更多数据 |
| **CNN v9.3** `cnn_v3` | 5×ResBlock(64) | ~41万 | Cross-Attn+MLP | 轻量高效，注意力价值头 |
| **Transformer** `transformer` | 5×Pre-LN Transformer | ~27万 | Cross-Attn+MLP | 全局视野，需要预训练 |

三架构共享相同接口：`(B, 3, 15, 15) → policy_logits(B, 225), value(B,)`，可无缝切换。

---

## 项目结构

```
wuziqi_agent/
├── az_train.py              # ★ AlphaZero 自对弈训练 (--arch 切换架构)
├── pre_train.py             # ★ 联合预训练 (Behavior Cloning + Value Regression)
├── pretrain_vs_agent.py     # ★ 神经网络 vs 规则引擎 对弈训练
├── test.py                  # 模型推理测试 + 战术评估
├── run_arena.py             # ★ AI 竞技场 (--agent1/--agent2 赛马)
├── human_vs_ai.py           # 人机对弈 GUI
├── data_collector.py        # 训练数据采集
│
├── agents/
│   ├── rule_based.py        # 规则引擎 Agent
│   └── neural/
│       ├── registry.py      # ★ 架构注册表 (单一真理源)
│       ├── cnn_v2.py        # CNN v9.2 实现
│       ├── cnn_v3.py        # CNN v9.3 实现
│       ├── transformer.py   # Transformer 实现
│       └── az_agent.py      # AZAgent (MCTS + 任意网络)
│
├── training/
│   ├── config.py            # 训练配置 (统一多架构参数)
│   ├── inference_server.py  # GPU 批量推理服务器
│   └── replay_buffer.py     # 经验回放缓冲区
│
├── search/
│   └── mcts.py              # MCTS 搜索树
│
├── arena/
│   └── visual.py            # GUI 竞技场观战
│
├── core/
│   └── gamerules.py         # 五子棋游戏规则
│
└── utils/                   # 工具函数 (棋盘增强、Zobrist哈希等)
```

---

## Checkpoint 格式规范 (v12+)

所有训练脚本保存的模型文件 **必须** 包含 `model_config` 字段，使 downstream 消费者无需猜测架构类型。

```json
{
    "model_state_dict": OrderedDict(...),   // 模型权重
    "model_config": {                       // ★ 架构元数据（强制）
        "arch_type": "cnn_v3",              // 架构标识 (cnn_v2 | cnn_v3 | transformer)
        "num_res_blocks": 5,                // CNN 残差块数
        "channels": 64,                     // CNN 通道数
        "board_size": 15
    }
}
```

**设计原则：**
- **生产者写入元数据** — 所有 save 点（`best_model.pt`、`new_model_arena.pt`、历史快照）统一写入 `model.get_config()` 产出的 `model_config`
- **消费者单一入口** — 所有加载统一调用 `registry.build_model_from_checkpoint()`，该函数是唯一真理源
- **最后防线** — 如果旧 checkpoint 缺失 `model_config`，`infer_arch_from_state_dict()` 从权重键名推断架构作为 fallback

---

## 完整训练流程

### 1. 收集训练数据

```bash
python data_collector.py
# 生成 collected_data/training_data.npz
```

### 2. 联合预训练

从数据集学习策略和价值：

```bash
# 预训练 CNN v9.3 (默认)
python pre_train.py --arch cnn_v3

# 预训练 Transformer
python pre_train.py --arch transformer

# 续训
python pre_train.py --arch cnn_v3 --resume --max_epochs 100
```

Checkpoint 保存至 `checkpoints/joint_pretrain/{arch}/`。

### 3. AlphaZero 自对弈训练

用预训练权重初始化，启动强化学习循环：

```bash
# CNN v9.3 (默认)
python az_train.py --arch cnn_v3 \
    --initial_model checkpoints/joint_pretrain/cnn_v3/best_model.pt

# CNN v9.2
python az_train.py --arch cnn_v2

# Transformer
python az_train.py --arch transformer

# 续训
python az_train.py --arch cnn_v3 --resume
```

每轮迭代包含三个阶段：自对弈 → 网络训练 → 竞技场评估。

### 4. 模型测试

```bash
python test.py --model_path checkpoints/az_train/cnn_v3/best_model.pt
```

### 5. 竞技场赛马

```bash
# CNN v2 vs CNN v3
python run_arena.py --agent1 cnn_v2 --agent2 cnn_v3

# CNN v3 vs Transformer
python run_arena.py --agent1 cnn_v3 --agent2 transformer

# 规则引擎 vs CNN v3
python run_arena.py --agent1 rule_based --agent2 cnn_v3

# 自定义模型路径
python run_arena.py --agent1 rule_based --agent2 checkpoints/az_train/cnn_v3/best_model.pt

# 指定 MCTS 模拟次数
python run_arena.py --agent1 cnn_v2 --agent2 cnn_v3 --sims 800
```

---

## 命令行参考

### `az_train.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arch` | `cnn_v3` | 网络架构: `cnn_v2` / `cnn_v3` / `transformer` |
| `--initial_model` | (无) | 预训练权重路径 |
| `--resume` | False | 从检查点续训 |

### `pre_train.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arch` | `cnn_v3` | 网络架构 |
| `--resume` | False | 续训 |
| `--max_epochs` | 50 | 最大训练轮数 |

### `run_arena.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--agent1` | `rule_based` | 黑方: `rule_based` / `cnn_v2` / `cnn_v3` / `transformer` / 路径 |
| `--agent2` | `cnn_v2` | 白方(同上) |
| `--sims` | 400 | MCTS 模拟次数 |

### `test.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | (无) | 模型路径(自动推断架构) |
| `--arch` | `cnn_v3` | 无 checkpoint 时使用的随机权重架构 |

---

## 配置参数说明

项目中的超参数分为两大类：

### 类型 1：架构超参数（Architecture Params）

定义网络**结构**，不同架构有不同的参数名和默认值，在各自网络文件的 `@register` 装饰器中声明：

| 架构 | 参数名 | 默认值 | 含义 |
|------|------|--------|------|
| **CNN v9.2** | `num_res_blocks` | 4 | 残差块数量 |
| | `channels` | 128 | 通道数 |
| | `board_size` | 15 | 棋盘大小 |
| **CNN v9.3** | `num_res_blocks` | 5 | 残差块数量 |
| | `channels` | 64 | 通道数 |
| | `board_size` | 15 | 棋盘大小 |
| **Transformer** | `d_model` | 128 | 嵌入维度 |
| | `num_heads` | 4 | 注意力头数 |
| | `num_layers` | 5 | Transformer 块数 |
| | `ff_expand` | 4 | FFN 扩展倍数 |
| | `dropout` | 0.1 | Dropout 比例 |
| | `board_size` | 15 | 棋盘大小 |

### 类型 2：训练/搜索超参数（Training & MCTS Params）

控制**训练过程**和**搜索行为**，三个架构共用相同的参数名（可设不同值）：

#### AlphaZeroConfig (az_train.py) 完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `arch_type` | `cnn_v3` | 网络架构: `cnn_v2` / `cnn_v3` / `transformer` |
| `arch_params` | `None` | 架构参数覆盖字典（覆盖注册表默认值） |
| `num_iterations` | 200 | 总训练迭代数 |
| `games_per_iteration` | 200 | 每轮自对弈局数 |
| `train_steps_per_iteration` | 80 | 每轮训练步数 |
| `baseline_eval_games` | 40 | 基准评估局数 |
| `arena_games` | 50 | 竞技场评估局数 |
| `num_sims` | 400 | MCTS 模拟次数（越大越强但越慢） |
| `c_puct` | 2.5 | PUCT 探索常数（大 = 更多探索） |
| `dirichlet_alpha` | 0.2 | Dirichlet 噪声 alpha（大 = 更均匀的噪声） |
| `dirichlet_epsilon` | 0.25 | Dirichlet 噪声混合比例 |
| `temp_threshold` | 60 | 温度衰减步数阈值 |
| `candidate_radius` | 2 | 候选着法搜索半径 |
| `advantage_clip` | 1.0 | 优势裁剪范围 |
| `arena_win_threshold` | 0.6 | 模型更新胜率阈值（新模型 >= 此值则更新） |
| `arena_num_sims` | 400 | 竞技场 MCTS 模拟次数 |
| `arena_c_puct` | 2.5 | 竞技场 PUCT 探索常数 |
| `arena_dirichlet_alpha` | 0.2 | 竞技场 Dirichlet alpha |
| `arena_dirichlet_epsilon` | 0.0 | 竞技场 Dirichlet 噪声（通常关） |
| `arena_temperature` | 1e-3 | 竞技场温度（接近确定性） |
| `arena_temp_threshold` | 4 | 竞技场温度阈值 |
| `arena_collapse_threshold` | 0.35 | 坍塌检测阈值 |
| `arena_save_image_every_n_games` | 5 | 竞技场图片保存间隔 |
| `arena_data_to_buffer` | True | 竞技场数据是否加入缓冲区 |
| `baseline_num_sims` | 400 | 基准评估 MCTS 模拟次数 |
| `baseline_agent_depth` | 4 | 基准 Agent 搜索深度 |
| `baseline_agent_max_candidates` | 10 | 基准 Agent 候选数 |
| `batch_size` | 128 | 训练批次大小 |
| `learning_rate` | 1e-4 | 学习率 |
| `lr_warmup_iterations` | 5 | LR 预热迭代数 |
| `weight_decay` | 1e-4 | 权重衰减（L2 正则化） |
| `grad_clip` | 1.0 | 梯度裁剪阈值 |
| `policy_loss_weight` | 1.0 | 策略损失权重 |
| `value_loss_weight` | 1.0 | 价值损失权重 |
| `value_loss_delta` | 0.5 | HuberLoss delta |
| `replay_buffer_size` | 500000 | 回放缓冲区容量 |
| `min_replay_size` | 5000 | 开始训练的最小样本数 |
| `num_workers` | 16 | Worker 进程数 |
| `max_batch_size` | 128 | 推理服务器最大批次大小 |
| `checkpoint_dir` | `checkpoints/az_train` | 存档目录（自动追加架构名） |
| `save_interval` | 1 | 存档间隔（迭代） |
| `save_replay_interval` | 1 | 回放缓冲区存档间隔 |
| `save_image_every_n_games` | 50 | 图片保存间隔 |
| `device` | `auto` | 计算设备: `auto` / `cuda` / `cpu` |
| `initial_model` | `checkpoints/joint_pretrain/best_model.pt` | 预训练权重路径 |
| `resume` | False | 从检查点续训 |

#### PretrainConfig (pretrain_vs_agent.py) 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_iterations` | 50 | 预训练轮次 |
| `games_per_iteration` | 100 | 每轮对弈局数 |
| `num_sims` | 400 | MCTS 模拟次数 |
| `c_puct` | 1.5 | PUCT（比 RL 阶段保守） |
| `dirichlet_epsilon` | 0 | 关闭噪声（对手随机） |
| `temp_threshold` | 6 | 温度阈值 |
| `candidate_radius` | 3 | 候选半径 |
| `agent_depth` | 4 | 对手搜索深度 |
| `agent_max_candidates` | 10 | 对手候选数 |
| `agent_vct_depth` | 8 | 对手 VCT 深度 |
| `early_stop_patience` | 15 | 早停耐心 |
| `early_stop_min_delta` | 0.02 | 早停最小改善 |

#### TrainConfig (pre_train.py) 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_path` | `collected_data/training_data.npz` | 训练数据路径 |
| `val_ratio` | 0.1 | 验证集比例 |
| `max_samples` | 0 | 最大样本数 (0=全部) |
| `max_epochs` | 50 | 最大训练轮数 |
| `warmup_epochs` | 5 | 学习率预热轮数 |
| `patience` | 15 | 早停耐心 |
| `min_delta` | 1e-5 | 早停最小改善 |
| `scheduler_type` | `cosine` | 学习率调度: `cosine` / `plateau` |
| `actor_loss_weight` | 1.0 | 策略损失权重 |
| `critic_loss_weight` | 1.0 | 价值损失权重 |
| `loss_type` | `huber` | 损失类型: `huber` / `mse` |

---

## 多架构超参数管理原理

### 数据流：从注册表到训推全链路

```
┌──────────────────────────────────────────────────────────────┐
│  @register('cnn_v3', param_names=[...], defaults={...})     │
│  class CNNActorCriticNet_v3(nn.Module): ...                  │
│                                                              │
│  每个网络类在定义时向 NETWORK_REGISTRY 注册自身的：           │
│    - 网络类引用 (用于实例化)                                  │
│    - 参数名列表 (用于文档/校验)                                │
│    - 默认超参数  (单一真理源)                                 │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  registry.py 统一 API:                                       │
│                                                              │
│  get_defaults(arch_type)        → 返回该架构的默认参数字典    │
│  get_network_class(arch_type)   → 返回该架构的 PyTorch 类    │
│  build_model_from_config()      → 默认值 + 覆盖 → 模型实例   │
│  build_model_from_checkpoint()  → 自动推断架构 + 重建模型     │
└───────────────────────┬──────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
┌──────────┐  ┌──────────────┐  ┌──────────────┐
│ 训练入口  │  │ 推理服务      │  │ 智能体        │
│az_train  │  │inference_svr │  │ az_agent.py   │
│pre_train │  │ .py           │  │               │
│pretrain  │  │               │  │               │
│                                                              │
│ 统一参数流:                                                  │
│ Config(arch_type='x') → get_defaults('x') → model(**params)  │
│ Config(arch_type='x') → arch_params → setattr(config, ...)  │
│ Config(...)           → checkpoint_dir = base/{arch_type}   │
└──────────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 实现文件 | 好处 |
|------|---------|------|
| **单一真理源** | `agents/neural/registry.py` | 增删架构只需改注册表 |
| **架构隔离存储** | 所有 Config 类 | `checkpoints/az_train/cnn_v3/` 各自独立，永不覆盖 |
| **参数覆盖机制** | `arch_params` 字典 | 命令行灵活，注册表保底 |
| **权重推断兜底** | `infer_arch_from_state_dict()` | 旧 checkpoint 无 `model_config` 也能加载 |
| **别名兼容** | `ARCH_ALIASES` 映射表 | `'cnn'` 自动映射为 `'cnn_v2'` |

### 为不同架构设置不同训练超参数

虽然架构参数由注册表自动管理，但**训练/搜索超参数**可以在 Config 构造时按架构分叉：

```python
from training.config import AlphaZeroConfig

def create_config(arch_type):
    """根据架构返回不同的训练超参数组合"""
    base = dict(arch_type=arch_type, num_iterations=200, games_per_iteration=200)

    if arch_type == 'cnn_v2':
        # CNN v9.2: 参数多，不容易过拟合 → 大 batch + 大学习率
        base.update(num_sims=600, batch_size=256, learning_rate=2e-4,
                    c_puct=3.0, dirichlet_epsilon=0.25)

    elif arch_type == 'cnn_v3':
        # CNN v9.3: 参数少 → 中等 batch + 中等学习率防止过拟合
        base.update(num_sims=400, batch_size=128, learning_rate=1e-4,
                    c_puct=2.5, dirichlet_epsilon=0.25)

    elif arch_type == 'transformer':
        # Transformer: 需要小学习率 + 高 dropout + 更多迭代
        base.update(num_sims=400, batch_size=64, learning_rate=5e-5,
                    c_puct=2.0, dirichlet_epsilon=0.15,
                    arch_params={'dropout': 0.15},       # 覆盖注册表默认值
                    num_iterations=300)                   # 更多迭代

    return AlphaZeroConfig(**base)

# 使用
config = create_config('cnn_v3')
```

**原理**：`AlphaZeroConfig.__init__` 的执行顺序为：
1. 调用 `get_defaults(arch_type)` 获取架构默认参数（如 `channels=64`）
2. 用 `arch_params` 字典覆盖（如 `arch_params={'dropout': 0.15}`）
3. 将所有参数（架构 + 训练）统一设为实例属性
4. 训练代码中通过 `config.channels`、`config.learning_rate` 等直接访问

---

## 调参方法教程

### 1. 调参总原则

```
先跑通 → 再调优 → 后赛马
  │         │         │
  ▼         ▼         ▼
默认参数    控制变量   多架构对比
小规模试跑  单参数微调  run_arena.py
```

### 2. 小规模试跑（Smoke Test）

在全面训练之前，用**小规模参数**快速验证 pipeline 是否正常（约 10 分钟跑完一轮）：

```python
from training.config import AlphaZeroConfig

config = AlphaZeroConfig(
    arch_type='cnn_v3',
    num_iterations=2,               # 只跑 2 轮
    games_per_iteration=10,         # 每轮 10 局自对弈
    train_steps_per_iteration=5,    # 每轮 5 步训练
    num_sims=50,                    # 极少的 MCTS 模拟
    baseline_eval_games=5,          # 评估 5 局
    arena_games=10,                 # 竞技场 10 局
    batch_size=32,                  # 小批次
    num_workers=4,                  # 少 worker
)
```

验证通过后再切回完整参数。

### 3. 各架构推荐超参数组合

#### 场景 A: 低资源 / 快速实验

适用：CPU 训练、笔记本 GPU、快速原型验证。

| 参数 | CNN v9.2 | CNN v9.3 | Transformer |
|------|:--------:|:--------:|:-----------:|
| `num_sims` | 200 | 200 | 200 |
| `batch_size` | 64 | 64 | 32 |
| `learning_rate` | 1e-4 | 1e-4 | 5e-5 |
| `games_per_iteration` | 100 | 100 | 100 |
| `num_iterations` | 50 | 50 | 100 |

#### 场景 B: 标准训练（推荐入门）

适用：单卡 GPU（如 RTX 3060+），首次训练。

| 参数 | CNN v9.2 | CNN v9.3 | Transformer |
|------|:--------:|:--------:|:-----------:|
| `num_sims` | 400 | 400 | 400 |
| `batch_size` | 128 | 128 | 64 |
| `learning_rate` | 1e-4 | 1e-4 | 1e-4 |
| `games_per_iteration` | 200 | 200 | 200 |
| `num_iterations` | 200 | 200 | 300 |

#### 场景 C: 高质量训练（追求最优）

适用：多 GPU、充足算力、追求最强模型。

| 参数 | CNN v9.2 | CNN v9.3 | Transformer |
|------|:--------:|:--------:|:-----------:|
| `num_sims` | 800 | 800 | 800 |
| `batch_size` | 256 | 256 | 128 |
| `learning_rate` | 2e-4 | 1e-4 | 5e-5 |
| `games_per_iteration` | 400 | 400 | 300 |
| `num_iterations` | 300 | 300 | 500 |

### 4. 核心参数调优指南

#### 🔑 学习率 (`learning_rate`)

| 现象 | 调整方向 |
|------|---------|
| loss 震荡剧烈、不收敛 | ↓ 减小（如 5e-5） |
| loss 下降过慢、欠拟合 | ↑ 增大（如 5e-4） |
| CNN 推荐范围 | 5e-5 ~ 5e-4 |
| Transformer 推荐范围 | 1e-5 ~ 1e-4 |

*Transformer 对学习率更敏感，建议从小开始。*

#### 🔑 MCTS 模拟次数 (`num_sims`)

| 值 | 效果 | 每局时间 (RTX 3060, CNN v9.3) |
|----|------|:---:|
| 100 | 弱决策，快速迭代 | ~2s |
| 400 | 中等强度，训练默认值 | ~8s |
| 800 | 强决策，竞技场推荐 | ~16s |
| 1600 | 极强决策，赛马对比用 | ~32s |

*时间与 `num_sims` 近似线性。训练阶段 400 够用，竞技场用 800+ 更准确。*

#### 🔑 批次大小 (`batch_size`)

| 考量 | 建议 |
|------|------|
| 显存充足 | 增大 batch_size（256+）→ 更快收敛 |
| 显存紧张 | 减小 batch_size（≤64）→ 更多噪声，更慢收敛 |
| CNN v9.3 | 128 推荐 |
| Transformer | 64 推荐（显存占用大） |

#### 🔑 PUCT 探索常数 (`c_puct`)

| 阶段 | 推荐值 | 原因 |
|------|:-----:|------|
| 预训练 (vs Agent) | 1.5 | 对手已随机，不需要太多探索 |
| 自对弈训练 | 2.5 | 均衡探索与利用 |
| 竞技场 / 推理 | 1.0 ~ 2.0 | 偏确定性，追求稳定发挥 |

#### 🔑 Dirichlet 噪声

| 参数 | 作用 | 调参建议 |
|------|------|---------|
| `dirichlet_alpha` | 噪声均匀度 | 0.03(集中) ~ 1.0(均匀)，默认 0.2 |
| `dirichlet_epsilon` | 噪声混合比例 | 0.0(关闭) ~ 0.5(高噪声)，默认 0.25 |

*探索不足（策略过早收敛到局部最优）时，增大两个参数。*

#### 🔑 权重衰减 (`weight_decay`)

| 现象 | 调整方向 |
|------|---------|
| 过拟合（训练 loss << 评估 loss） | ↑ 增大（如 5e-4） |
| 欠拟合（训练 loss 也居高不下） | ↓ 减小（如 1e-5） |
| 推荐范围 | 1e-5 ~ 1e-3 |
| Transformer | 稍大（1e-4 ~ 5e-4），因参数更集中 |

### 5. 控制变量实验法

进行 A/B 对比时，**每次只改一个参数**，其他保持一致：

```bash
# 实验 1: 对比学习率 1e-4 vs 5e-5 (其他参数相同)
python az_train.py --arch cnn_v3        # 默认 lr=1e-4

# 修改 config.py 或构造代码设 lr=5e-5 再跑一次
python az_train.py --arch cnn_v3

# 用 run_arena.py 对比两个产出的模型
python run_arena.py \
    --agent1 checkpoints/az_train/cnn_v3/best_model.pt \
    --agent2 checkpoints/az_train/cnn_v3_exp_lr5e-5/best_model.pt \
    --sims 800
```

### 6. 训练中的监控信号

| 信号 | 健康 | 异常 |
|------|------|------|
| **Policy Loss** | 缓慢下降 | 不降或震荡 → 调学习率 |
| **Value Loss** | 缓慢下降至稳定 | 震荡 → 增大 HuberLoss delta |
| **竞技场胜率** | 逐步上升 | 长期停滞 → 增大探索噪声 |
| **坍塌检测** | 策略熵 > 阈值 | 策略坍塌 → 增大 `dirichlet_epsilon` |
| **GPU 利用率** | > 80% | < 50% → 增大 `batch_size` 或 `num_workers` |

### 7. 常见调参错误与修正

| 错误 | 后果 | 修正 |
|------|------|------|
| Transformer 用 CNN 的学习率(1e-4) | 剧烈震荡，不收敛 | 降到 5e-5 或更低 |
| `min_replay_size` 太小 | 训练用噪声数据 | 至少 5000，推荐 10000 |
| `num_sims` 太低(<100) | 自对弈质量差 | 至少 200，推荐 400 |
| 关闭 Dirichlet 噪声 | 策略过早收敛 | 训练阶段至少保留 `epsilon=0.2` |
| `dirichlet_epsilon` 太大(>0.5) | 决策太随机 | 降回默认 0.25 |
| `weight_decay` 太大 | 欠拟合 | 降到 1e-5 |
| 不同架构共用同一个 `checkpoint_dir` | 权重互相覆盖 | **代码已自动隔离，无需手动处理** |

---

## 添加新架构

只需 **3 步**：

### 1. 创建网络文件

创建 `agents/neural/my_arch.py`，实现 `nn.Module` 子类并用 `@register` 注册：

```python
from agents.neural.registry import register

@register('my_arch',
    param_names=['channels', 'board_size'],
    defaults={'channels': 64, 'board_size': 15})
class MyNet(nn.Module):
    def __init__(self, channels=64, board_size=15):
        super().__init__()
        # ... 你的网络结构 ...

    def forward(self, x, return_value_only=False):
        # 返回 (policy_logits(B,225), value(B,))
        # 若 return_value_only=True, policy_logits 可为 None
        ...

    @property
    def arch_type(self):
        return "my_arch"

    def get_config(self):
        return {'arch_type': 'my_arch', 'channels': self.channels}
```

### 2. 注册到包

在 `agents/neural/__init__.py` 中添加：

```python
from agents.neural.my_arch import MyNet
```

### 3. 开始训练

```bash
python az_train.py --arch my_arch
```

---

## 常见问题

**Q: 如何切换训练架构？**

所有入口脚本都支持 `--arch` 参数。`checkpoint_dir` 会自动追加架构名以避免覆盖。

**Q: 旧 checkpoint 能加载吗？**

可以。系统会根据 `model_config.arch_type` 字段自动推断架构，旧 `'cnn'` 别名自动映射为 `'cnn_v2'`。

**Q: CNN v2/v3/Transformer 哪个更好？**

- **CNN v9.2** (124万参数)：最成熟稳定，需要更多训练数据
- **CNN v9.3** (41万参数)：轻量级，交叉注意力价值头提供更好全局判断
- **Transformer** (27万参数)：全局自注意力，推荐先用 `pre_train.py` 预训练

建议用 `run_arena.py` 进行赛马对比。

**Q: GPU 内存不足怎么办？**

减少 `batch_size` 和 `games_per_iteration`，或用 `--arch cnn_v3` (最轻量)。

---

## 许可

MIT License