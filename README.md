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

### AlphaZeroConfig (az_train.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_iterations` | 200 | 总训练迭代数 |
| `games_per_iteration` | 200 | 每轮自对弈局数 |
| `num_sims` | 400 | MCTS 模拟次数 |
| `batch_size` | 128 | 训练批次大小 |
| `learning_rate` | 1e-4 | 学习率 |
| `replay_buffer_size` | 500000 | 回放缓冲区容量 |
| `arena_games` | 50 | 竞技场评估局数 |

在 `training/config.py` 中修改默认值，或在代码中传参覆盖。

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