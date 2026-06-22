# 五子棋 AI 训练系统 — AlphaZero + 核采样

基于 **AlphaZero 强化学习**的五子棋训练系统。支持 **CNN v2 / v3 / Transformer / hybrid_v1** 多架构赛马，自对弈用 MCTS 搜索、竞技场评估用核采样快速裁定。

## 快速开始

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 人机对弈

```bash
# MCTS 模式 (默认，强)
python human_vs_ai.py --model checkpoints/az_train/cnn_v3/best_model.pt --color 1

# 核采样模式 (快速)
python human_vs_ai.py --mode nucleus --model checkpoints/az_train/cnn_v3/best_model.pt
```

### AI 竞技场

```bash
# CNN v3 vs CNN v2 (默认 MCTS)
python run_arena.py --agent1 cnn_v3 --agent2 cnn_v2

# hybrid_v1 vs CNN v3
python run_arena.py --agent1 hybrid_v1 --agent2 cnn_v3

# MCTS vs 核采样
python run_arena.py --agent1 cnn_v3 --agent2 cnn_v3 \
    --agent1-mode mcts --agent2-mode nucleus

# 纯核采样互殴
python run_arena.py --agent1 cnn_v3 --agent2 cnn_v3 \
    --agent1-mode nucleus --agent2-mode nucleus

# 规则引擎 vs 任意模型
python run_arena.py --agent1 rule_based --agent2 cnn_v3
```

---

## 核心概念

### 决策模式

系统支持两种决策方式，训练和外部对弈可独立选择：

| 模式 | 原理 | 速度 | 强度 | 适用场景 |
|------|------|:---:|:---:|----------|
| **MCTS** | PUCT 搜索 + 神经网络评估，每步 400 次模拟 | 慢 | 强 | 自对弈训练（需要高质量策略标签）、基准评估 |
| **核采样** | 直接取策略头输出，按 top-p 累积 60% 概率采样 | 快 (~400x) | 中 | 竞技场快速裁定、人机/机机对战 |

自对弈训练**必须**用 MCTS——只有 MCTS 才能产出高质量的 `(state, policy, value)` 训练样本。竞技场评估用核采样，不收集数据，仅判定新旧模型相对强弱。

### 训练循环

每轮迭代分 3 个阶段（`az_train.py`）：

```
Phase 1 自对弈 (MCTS)           Phase 2 网络训练              Phase 3 竞技场评估 (核采样)
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐
│ 200局自我对弈         │    │ 从 ReplayBuffer 采样  │    │ 新模型 vs 最佳模型        │
│ Dirichlet 噪声探索    │ => │ 8-way D4 增强        │ => │ 50局核采样对弈            │
│ 产出训练样本          │    │ HuberLoss + CE 训练  │    │ 胜率≥60%→晋升             │
│ → ReplayBuffer       │    │ Cosine LR + Warmup   │    │ 胜率<35%→坍塌回退          │
└─────────────────────┘    └──────────────────────┘    └─────────────────────────┘
```

关键设计：
- **竞技场数据不进入训练**——核采样没有 MCTS 策略标签，无法用于监督学习
- **两阶段安全存档**：Phase 2 训练完成后立即存档，Phase 3 崩溃不丢失进度
- **可选 Phase 4**：晋升后的最佳模型 vs 规则引擎基准评估（MCTS 搜索）

### 架构注册表

所有网络架构通过 `agents/neural/registry.py` 统一管理：

1. 网络类用 `@register('name', ['channels', ...], {defaults})` 装饰器注册
2. 模型加载统一走 `registry.build_model_from_checkpoint()` —— 自动推断架构、从权重解析参数、构造模型并加载
3. Checkpoint 必须包含 `model_config.arch_type`，旧 `'cnn'` 别名自动映射为 `'cnn_v2'`

添加新架构见下方「添加新架构」章节的完整检查清单。

### Checkpoint 格式

```json
{
    "model_state_dict": OrderedDict(...),
    "model_config": {
        "arch_type": "cnn_v3",
        "num_res_blocks": 5,
        "channels": 64,
        "board_size": 15
    }
}
```

所有 save 点统一调用 `model.get_config()`，消费者统一调用 `build_model_from_checkpoint()`。

> `hybrid_v1` 的 `model_config` 包含 `{'arch_type': 'hybrid_v1', 'num_res_blocks': 5, 'channels': 64, 'board_size': 15}`。

---

## 项目结构

```
wuziqi_agent/
├── az_train.py              # AlphaZero 自对弈训练
├── pre_train.py             # 联合预训练 (Behavior Cloning)
├── pretrain_vs_agent.py     # 神经网络 vs 规则引擎 对弈训练
├── run_arena.py             # AI 竞技场 (支持 MCTS/核采样)
├── human_vs_ai.py           # 人机对弈 GUI (支持 MCTS/核采样)
├── test.py                  # 模型推理测试
├── data_collector.py        # 训练数据采集
│
├── agents/
│   ├── rule_based.py        # 规则引擎 Agent
│   └── neural/
│       ├── registry.py      # 架构注册表 (单一真理源)
│       ├── cnn_v2.py        # CNN v2 (4×ResBlock, 128ch)
│       ├── cnn_v3.py        # CNN v3 (5×ResBlock, 64ch, Cross-Attn)
│       ├── hybrid_v1.py     # Hybrid CNN+Transformer (5×ResBlock + 1×Transformer)
│       ├── transformer.py   # Transformer (Pre-LN, 全局自注意力)
│       ├── rope.py          # 2D-RoPE 旋转位置编码（共享模块）
│       └── az_agent.py      # AZAgent (MCTS/核采样 双模式)
│
├── training/
│   ├── config.py            # 训练配置 (多架构统一参数)
│   ├── inference_server.py  # GPU 批量推理 (单/双模型)
│   └── replay_buffer.py     # 经验回放缓冲区
│
├── search/
│   ├── mcts.py              # MCTS 搜索树
│   └── sampling.py          # 核采样 (top-p 策略采样)
│
├── arena/
│   └── visual.py            # GUI 竞技场观战
│
├── core/
│   └── gamerules.py         # 五子棋规则 (15×15)
│
└── utils/                   # 棋盘增强、Zobrist 哈希、图片渲染
```

---

## 神经网络架构

| 架构 | 主干 | 参数 | 价值头 | 特点 |
|------|------|:---:|------|------|
| **CNN v2** `cnn_v2` | 4×ResBlock(128) | ~124万 | Conv→GAP→FC | 经典稳定 |
| **CNN v3** `cnn_v3` | 5×ResBlock(64) | ~41万 | Cross-Attn+MLP | 轻量高效，注意力价值头 |
| **Transformer** `transformer` | 5×Pre-LN | ~27万 | Cross-Attn+MLP | 全局视野，需预训练 |
| **hybrid_v1** `hybrid_v1` | 5×ResBlock(64) + 1×Transformer | ~43万 | Cross-Attn+MLP | CNN+Transformer 混合，局部与全局兼顾 |

接口统一：`(B, 3, 15, 15) → policy_logits(B, 225), value(B,)`。

---

## 命令行参考

### `az_train.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arch` | `cnn_v3` | 架构: `cnn_v2` / `cnn_v3` / `transformer` / `hybrid_v1` |
| `--initial_model` | (预训练权重) | 初始模型路径 |
| `--resume` | `False` | 从 checkpoint 续训 |

### `pre_train.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arch` | `cnn_v3` | 架构 |
| `--resume` | `False` | 续训 |
| `--max_epochs` | `50` | 最大训练轮数 |

### `run_arena.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--agent1` | `rule_based` | 黑方: `rule_based` / 架构名 / 模型路径 |
| `--agent2` | `cnn_v2` | 白方 (同上) |
| `--agent1-mode` | `mcts` | 黑方决策: `mcts` / `nucleus` |
| `--agent2-mode` | `mcts` | 白方决策: `mcts` / `nucleus` |
| `--sims` | `400` | MCTS 模拟次数 (MCTS 模式) |
| `--nucleus-p` | `0.6` | 核采样 top-p (nucleus 模式) |

### `human_vs_ai.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `checkpoints/az_train/best_model.pt` | 模型路径 |
| `--mode` | `mcts` | 决策模式: `mcts` / `nucleus` |
| `--sims` | `400` | MCTS 模拟次数 |
| `--nucleus-p` | `0.6` | 核采样 top-p |
| `--color` | `1` | 人类执子: 1=黑(先), 2=白(后) |

### `test.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | (无) | 模型路径 (自动推断架构) |
| `--arch` | `cnn_v3` | 无 checkpoint 时的随机权重架构 |

---

## 配置参考

### AlphaZeroConfig (`az_train.py`)

#### 训练循环

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_iterations` | `200` | 总训练迭代数 |
| `games_per_iteration` | `200` | 每轮自对弈局数 |
| `train_steps_per_iteration` | `80` | 每轮训练步数 |
| `replay_buffer_size` | `500000` | 回放缓冲区容量 |
| `min_replay_size` | `5000` | 开始训练的最小样本数 |

#### 自对弈 MCTS

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_sims` | `400` | MCTS 模拟次数 |
| `c_puct` | `2.5` | PUCT 探索常数 |
| `dirichlet_alpha` | `0.2` | Dirichlet 噪声 alpha |
| `dirichlet_epsilon` | `0.25` | Dirichlet 噪声混合比例 |
| `temp_threshold` | `4` | 温度衰减步数 (前4步 T=1.0, 之后 T=1e-3) |
| `candidate_radius` | `2` | 候选着法搜索半径 |
| `advantage_clip` | `1.0` | 优势值裁剪范围 |

#### 竞技场核采样

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `arena_games` | `50` | 竞技场对弈局数 |
| `arena_nucleus_p` | `0.6` | 核采样 top-p 阈值 |
| `arena_nucleus_temp_threshold` | `4` | 开局高温步数 |
| `arena_nucleus_early_temp` | `1.5` | 开局温度 (>1 使分布更平坦) |
| `arena_win_threshold` | `0.6` | 模型晋升胜率阈值 |
| `arena_collapse_threshold` | `0.35` | 坍塌检测阈值 |
| `arena_save_image_every_n_games` | `5` | 竞技场图片保存间隔 |

#### 基准评估 (最佳模型 vs 规则引擎)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `baseline_eval_games` | `40` | 基准评估局数 |
| `baseline_num_sims` | `400` | 基准评估 MCTS 模拟次数 |
| `baseline_agent_depth` | `4` | 规则引擎搜索深度 |
| `baseline_agent_max_candidates` | `10` | 规则引擎候选数 |

#### 优化器 & 并行

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | `128` | 训练批次大小 |
| `learning_rate` | `1e-4` | 学习率 |
| `lr_warmup_iterations` | `5` | LR 预热迭代数 |
| `weight_decay` | `1e-4` | 权重衰减 |
| `grad_clip` | `1.0` | 梯度裁剪 |
| `value_loss_delta` | `0.5` | HuberLoss delta |
| `num_workers` | `16` | Worker 进程数 |
| `max_batch_size` | `128` | 推理服务器最大批次 |

#### 其他

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `arch_type` | `cnn_v3` | 网络架构 |
| `arch_params` | `None` | 架构参数覆盖 (覆盖注册表默认值) |
| `checkpoint_dir` | `checkpoints/az_train` | 存档目录 (自动追加架构名) |
| `device` | `auto` | 计算设备 |
| `initial_model` | (预训练路径) | 预训练权重路径 |
| `resume` | `False` | 从 checkpoint 续训 |

### PretrainConfig & TrainConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **PretrainConfig** (`pretrain_vs_agent.py`) | | |
| `num_iterations` | `50` | 预训练轮次 |
| `games_per_iteration` | `100` | 每轮对弈局数 |
| `num_sims` | `400` | MCTS 模拟次数 |
| `c_puct` | `1.5` | PUCT (比 RL 保守) |
| `early_stop_patience` | `15` | 早停耐心 |
| **TrainConfig** (`pre_train.py`) | | |
| `data_path` | `collected_data/training_data.npz` | 训练数据 |
| `max_epochs` | `50` | 最大训练轮数 |
| `warmup_epochs` | `5` | LR 预热 |
| `patience` | `15` | 早停耐心 |

---

## 调参指南

### 小规模试跑 (Smoke Test)

先验证 pipeline 是否正常：

```python
config = AlphaZeroConfig(
    arch_type='cnn_v3',
    num_iterations=2, games_per_iteration=10,
    train_steps_per_iteration=5, num_sims=50,
    arena_games=10, batch_size=32, num_workers=4,
)
```

### 推荐参数组合

| 参数 | CNN v2 (低) | CNN v2 | CNN v3 (低) | CNN v3 | Trans (低) | Trans |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| `num_sims` | 200 | 400 | 200 | 400 | 200 | 400 |
| `batch_size` | 64 | 128 | 64 | 128 | 32 | 64 |
| `learning_rate` | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 5e-5 | 1e-4 |
| `games_per_iter` | 100 | 200 | 100 | 200 | 100 | 200 |
| `num_iterations` | 50 | 200 | 50 | 200 | 100 | 300 |

### 关键参数速查

| 参数 | 现象 | 调整 |
|------|------|------|
| `learning_rate` | loss 震荡不收敛 | ↓ 减小 |
| | loss 下降过慢 | ↑ 增大 |
| `c_puct` | 探索不足 | ↑ (最大 3.5) |
| | 决策太随机 | ↓ (最小 1.0) |
| `dirichlet_epsilon` | 策略过早收敛 | ↑ (最大 0.5) |
| `weight_decay` | 过拟合 (train << eval) | ↑ 增大 |
| `num_sims` | 自对弈质量差 | ↑ (400→800) |
| `batch_size` | GPU 利用率低 | ↑ (128→256) |
| `temp_threshold` | 开局单调 | ↑ (4→8) |

---

## 添加新架构

除 `@register` + `__init__.py` import 外，还需完成以下检查清单：

### 1. 创建网络文件

```python
# agents/neural/my_arch.py
from agents.neural.registry import register

@register('my_arch',
    param_names=['channels', 'board_size'],
    defaults={'channels': 64, 'board_size': 15})
class MyNet(nn.Module):
    def __init__(self, channels=64, board_size=15):
        super().__init__()
        # ...

    def forward(self, x):
        return policy_logits, value

    def get_config(self):
        return {'arch_type': 'my_arch', 'channels': self.channels}
```

### 2. 注册

在 `agents/neural/__init__.py` 添加：`from agents.neural.my_arch import MyNet`

### 3. `registry.py` — 3 处更新

- `ARCH_ALIASES` 新增条目（如有别名）
- `infer_arch_from_state_dict()` 新增独有键检测（必须在已有检测前插入）
- `build_model_from_checkpoint()` 纳入权重推断分支

### 4. `training/config.py` — 文档注释中 `arch_type` 说明新增架构名

### 5. 各入口脚本更新

| 脚本 | 更新内容 |
|------|---------|
| `az_train.py` | `--arch` help 文本新增 |
| `pre_train.py` | `--arch` 的 `choices` 列表新增 |
| `test.py` | `--arch` 的 `choices` 列表新增 |
| `run_arena.py` | `ARCH_TO_PATH` 映射表 + 文档注释 + help 文本新增 |

### 6. `pretrain_vs_agent.py` — weight decay 分组

`_create_optimizer()` 中按架构分支排除参数：
- **纯 CNN**（cnn_v2/v3）：排除 `bn` + `bias`
- **纯 Transformer**：排除 `ln` + `bias` + `norm`
- **混合架构**（hybrid_v1）：同时排除 `bn` + `ln` + `bias`

### 7. 共享模块

通用组件（如 RoPE2D）抽取到独立文件，避免循环依赖和多份拷贝。

### 8. 训练

```bash
python az_train.py --arch my_arch
```

---

## 常见问题

**Q: MCTS 和核采样模式怎么选？**

培训阶段自对弈必须用 MCTS（需要高质量策略目标）。竞技场用核采样（速度快，相对强弱可分辨）。人机/机机对战自由选择。

**Q: 如何切换训练架构？**

所有入口脚本支持 `--arch` 参数。`checkpoint_dir` 自动追加架构名避免覆盖。

**Q: 旧 checkpoint 能加载吗？**

可以。系统根据 `model_config.arch_type` 自动推断，旧 `'cnn'` 映射为 `'cnn_v2'`。

**Q: CNN v2/v3/Transformer/hybrid_v1 哪个更好？**

- **CNN v2** (124万): 最成熟，需更多数据
- **CNN v3** (41万): 轻量高效，推荐首选
- **hybrid_v1** (43万): CNN+Transformer 混合，兼具局部与全局，推荐作为 cnn_v3 的升级
- **Transformer** (27万): 全局视野，建议先预训练

建议用 `run_arena.py` 赛马对比。

**Q: GPU 显存不足？**

减少 `batch_size` 和 `games_per_iteration`，或用 `--arch cnn_v3`（最轻量）。

---

## 许可

MIT License
