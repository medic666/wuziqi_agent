# AGENTS.md

## 项目概况

- Python 五子棋 AlphaZero 训练系统 — 纯脚本运行，无包安装
- 依赖管理用 `uv`；Python ≥3.10；torch 2.1.2 锁定
- 所有脚本必须从仓库根目录执行

## 环境搭建

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## 入口脚本

| 脚本 | 用途 |
|---|---|
| `az_train.py` | AlphaZero 自对弈训练主循环 |
| `pre_train.py` | 联合预训练（Behavior Cloning） |
| `pretrain_vs_agent.py` | 神经网络 vs 规则引擎对弈采集数据 |
| `run_arena.py` | AI 竞技场（支持 MCTS/核采样赛马） |
| `human_vs_ai.py` | 人机对弈 GUI（支持 MCTS/核采样） |
| `test.py` | 模型推理可视化测试（无自动化测试套件） |

## 架构注册表

- `agents/neural/registry.py` — 网络架构单一真理源
- `@register('name', param_names, defaults)` 装饰器注册每个网络类
- 添加架构流程：(1) 新建文件加 `@register`，(2) `__init__.py` import，(3) `registry.py` 更新 ALIASES + `infer_arch_from_state_dict` + `build_model_from_checkpoint`，(4) 各入口脚本更新 `choices` 和 help 文本
- 加载 checkpoint：统一用 `registry.build_model_from_checkpoint()` — 自动推断架构、解析参数、构造模型
- 别名：`'cnn'` → `'cnn_v2'`

### 现有架构

| 架构 | 文件 | 骨干 | 策略头 | 价值头 | 参数量 |
|------|------|------|--------|--------|--------|
| `cnn_v2` | cnn_v2.py | 4×ResBlock(128) | Conv×3 | Conv+GAP+FC | ~124万 |
| `cnn_v3` | cnn_v3.py | 5×ResBlock(64) | Conv×3 | Cross-Attn+MLP | ~41万 |
| `transformer` | transformer.py | 5×Transformer+RoPE | MLP×3 | Cross-Attn+MLP | ~27万 |
| `hybrid_v1` | hybrid_v1.py | 5×CNN+1×Transformer+RoPE | 1×1 Conv×3 | Cross-Attn+MLP | ~43万 |

### 共享模块

- `agents/neural/rope.py` — 2D-RoPE 旋转位置编码，被 `transformer` 和 `hybrid_v1` 共用
- `agents/neural/registry.py` — 架构注册表

## Checkpoint 格式

- 必须包含 `model_config`，其中 `arch_type` 键标识架构
- 所有 save 点统一调用 `model.get_config()`
- 格式：`{'model_state_dict': OrderedDict(...), 'model_config': {'arch_type': 'cnn_v3', ...}}`
- `hybrid_v1` 的 `model_config` 包含 `{'arch_type': 'hybrid_v1', 'num_res_blocks': 5, 'channels': 64, 'board_size': 15}`

## 核心数据类型

- `GameState.board` 是 `bytearray`（225 字节），不是 numpy — `state_to_tensor()` 转换为 `np.ndarray(3,15,15) float32`
- MCTS 内部状态拷贝：`GameState(board=bytearray(root_state.board), ...)` 空历史（性能优化，约快 20-30%）
- `GomokuRules.apply_move_fast(state, action)` 就地修改棋盘（in-place mutation）

## 多进程

- `mp.set_start_method('spawn')` 在 `az_train.py` 模块级别设置
- GPU 推理：`InferenceServer`（单模型）/ `DualInferenceServer`（双模型）通过队列批量处理 worker 请求
- `DualInferenceServer`：model_id 0 = 最佳模型，model_id 1 = 新模型

## MCTS 树复用（仅自对弈）

1. `AZAgent.get_move()` 手动将 root 推进到自身上一步动作的子节点（`az_agent.py:165-171`）
2. `MCTS.search(last_action=...)` 推进 root 到对手的走法（`mcts.py:258-287`）
3. `raw_value` 缓存在 `MCTSNode` 上，避免重复评估已复用的子树
4. 双方共享同一棵 `MCTS` 树 — 整局连续复用

## 自对弈 vs 竞技场探索

| | 自对弈 | 竞技场 |
|---|---|---|
| 决策模式 | MCTS（400 模拟） | 核采样（无 MCTS） |
| Dirichlet 噪声 (epsilon) | 0.25（开启） | 无 |
| 温度阈值 | 4 步 | 4 步（核采样开局温度） |
| 低于阈值 | T=1.0 | T=1.5 |
| 高于阈值 | T=1e-3（近确定性） | Nucleus top-p=0.6, T=1.0 |

## 核采样（竞技场 & 外部 Agent）

- `search/sampling.py` — 共享 nucleus（top-p）采样函数：按策略概率降序排列，保留累积概率 ≥ `nucleus_p` 的走法，从中采样
- `AZAgent` 支持 `mode="mcts"`（默认）或 `mode="nucleus"`；`get_move()` 按模式分发
- `AZAgent.get_hint_move(state)` 返回推荐走法但不修改内部树状态（用于 `human_vs_ai.py` 提示按钮）
- 外部参数：`--mode mcts|nucleus`、`--nucleus-p 0.6`（human_vs_ai.py）；`--agent1-mode`/`--agent2-mode`（run_arena.py）

## 训练循环

- 每轮迭代 3 阶段：自对弈 → 训练 → 竞技场 →（可选）规则引擎基准评估
- Cosine LR + Warmup；HuberLoss 训练 value，log_softmax CE 训练 policy
- 每 batch 8-way D4 对称增强；训练时优势值裁剪
- 竞技场坍塌阈值：胜率 < 0.35 → 回退模型，重置优化器
- 竞技场数据不进 replay buffer（核采样无 MCTS 策略标签）
- 两阶段安全存档：Phase 2 训练完成后立即存档，Phase 3 崩溃不丢失进度

## 注意事项

- 项目无自动化测试套件、无 lint/typecheck 配置；验证代码靠 `test.py` 可视化检查
- 添加新网络架构后，记得在 `agents/neural/__init__.py` 中 import，否则注册表找不到
- 共享模块（如 RoPE2D）抽取到独立文件，避免循环依赖和多份拷贝

## 新建架构完整检查清单

添加新架构时，除 `@register` + `__init__.py` 外，还需检查：

1. **`registry.py`** — 3 处更新：
   - `ARCH_ALIASES` 新增条目
   - `infer_arch_from_state_dict()` 新增独有键检测（必须在已有 CNN/Transformer 检测前插入）
   - `build_model_from_checkpoint()` 纳入权重推断分支（如 stem 结构与 CNN 相同，加 `or 'hybrid_v1'`）
2. **`training/config.py`** — 文档注释中 `arch_type` 说明新增架构名
3. **`az_train.py`** — `--arch` help 文本新增
4. **`pre_train.py`** — `--arch` 的 `choices` 列表新增
5. **`test.py`** — `--arch` 的 `choices` 列表新增
6. **`run_arena.py`** — `ARCH_TO_PATH` 映射表 + 文档注释 + help 文本新增
7. **`pretrain_vs_agent.py`** — 检查 `_create_optimizer()` 中的 weight decay 分组逻辑：
   - 纯 CNN（cnn_v2/v3）：排除 `bn` + `bias`
   - 纯 Transformer：排除 `ln` + `bias` + `norm`
   - **混合架构（hybrid_v1）**：同时排除 `bn` + `ln` + `bias`

## 设计约定

### Transformer 块 Dropout 位置（4 层 Dropout）

遵照原论文 "Attention Is All You Need" 和 PyTorch `TransformerEncoderLayer` 标准：
1. `F.scaled_dot_product_attention(dropout_p=0.1)` — 注意力权重 dropout（softmax 后）
2. `self.dropout1` — 注意力子层输出 dropout（残差加和前）
3. FFN 内部 `Dropout(0.1)` — FFN 隐藏层 dropout（激活后）
4. `self.dropout2` — FFN 子层输出 dropout（残差加和前）

⚠️ `F.scaled_dot_product_attention(dropout_p=0.1)` 在 `eval()` 和 `torch.no_grad()` 下**仍施加 dropout**（与 `nn.MultiheadAttention` 不同），当前项目所有 Transformer 模块均保持此行为一致。

### CNN 总参数量 ≈ 42.6 万
