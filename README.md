# wuziqi_agent

基于 Python 的五子棋 AI 训练系统：从数据收集 → 预训练 → AlphaZero 强化学习。

## 项目结构 (v2.0 模块化重构)

```
wuziqi_agent/
├── core/                          # 核心游戏逻辑
│   └── gamerules.py               # GameState + GomokuRules（纯规则层）
│
├── agents/                        # 智能体层（插件式架构）
│   ├── base.py                    # Agent 抽象基类
│   ├── rule_based.py              # ADAgent 规则引擎（博弈树搜索）
│   └── neural/                    # 神经网络智能体
│       ├── network.py             # ActorCriticNet 模型定义
│       └── az_agent.py            # AZAgent（MCTS + 神经网络）
│
├── search/                        # 搜索算法层
│   └── mcts.py                    # MCTS 蒙特卡洛树搜索
│
├── training/                      # 训练基础设施
│   ├── replay_buffer.py           # 经验回放缓冲区
│   ├── inference_server.py        # GPU 推理服务器（单/双模型）
│   └── config.py                  # 训练配置类（3种配置集中管理）
│
├── arena/                         # 竞技场/UI层
│   └── visual.py                  # Tkinter 可视化对弈界面
│
├── utils/                         # 工具函数
│   ├── transforms.py              # 8向对称变换（D4 群）
│   ├── board_image.py             # 棋谱图片生成
│   └── zobrist.py                 # Zobrist 哈希工具
│
├── az_train.py                    # AlphaZero 训练入口
├── pre_train.py                   # 联合预训练入口
├── pretrain_vs_agent.py           # 神经网络 vs AgentAD 预训练
├── data_collector.py              # 对弈数据收集
├── human_vs_ai.py                 # 人机对弈 GUI
├── test.py                        # 模型测试脚本
└── run_arena.py                   # AI 竞技场入口（也可用 python -m arena.visual）
```

## Python版本 3.10.11
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2"
```

## 人工评估的 agent_ad.py → agents/rule_based.py:
```
depth: int = 4, 
max_candidates: int = 10, 
use_quiescence: bool = True,
vct_depth: int = 8, 
quiescence_depth: int = 2
```
算杀比较强，但是防杀比较傻。

## 模块化设计优势

### 1. 去除冗余
- **ReplayBuffer**: 从 2 份重复合并为 `training/replay_buffer.py`
- **Zobrist 工具**: 从 3 处重复合并为 `utils/zobrist.py`
- **配置类**: 3个文件中的配置类集中到 `training/config.py`
- **推理服务器**: InferenceServer + DualInferenceServer 合并到 `training/inference_server.py`

### 2. 充分解耦
- `agents/base.py` 定义统一 Agent 接口，新架构只需实现 `get_move()`
- `agents/neural/` 封装当前网络架构，未来可平行添加 `agents/transformer/`
- `search/mcts.py` 通过 `eval_fn` 回调与具体网络解耦
- `core/gamerules.py` 为纯规则层，不依赖任何 AI 模块

### 3. 中文注释
所有模块文件均包含完整的中文 docstring，关键函数标注了参数和返回值。

---

## 数据生成 data_collector.py:
这是一个强 cpu 占用模块，NUM_WORKERS = 10，设置为高于 cpu 核心数低于超线程数。运行 agent_ad 自对弈时，速度为 0.16 局/秒。

---

## 神经网络 network.py → agents/neural/network.py:
stem+n个残差块+双头。
当前致力于训练 4resblock+128 通道，一共 124 万参数。

---

## 预训练 pre_train.py:
预训练比较简单，但是之前因为样本过少所以一训练就过拟合。batch_size:128 配合 lr: 1e-4，是可行的。

---

## pretrain_vs_agent.py
用的都是 pretrain 和 az_train 的参数。实践证明神经网络自对弈比较有效，很快就能对机械 agent 胜率 100% 了。这一步可以忽略

---

## 自对弈 az_train.py
num_workers: int = 16, 算存之间的通道堵塞，cpu 和 gpu 都吃不满，16 个 workers 往 gpu 塞数据，效率比较折中。
续训系统会在 checkpoints/az_train/ 目录下寻找 latest_checkpoint.pt 文件，replay_buffer.npz 文件，self_play_model.pt 文件，best_model.pt 文件，请全部保留

---

## 结论
当前实际训练下来的问题就是，可能因为历史通道只有上一步的棋子。所以 ai 被调度后，大概率遗忘之前的威胁点。不过强势的地方是 ai 进攻很好，有一点学习到了机械 agent 的 vct 搜杀。限于计算机性能和时间，不能再往上提高 ai 水平了(可能的方向是增加历史通道，增加残差块数量，增加自对弈盘数)。不过当前的无禁手五子棋实验中，神经网络通过自对弈明显胜过了机械 AI 老师，并且价值网络判断黑棋的胜率为 90% 以上，包括竞技场的自对弈中也基本上是黑棋胜，已经证明了神经网络的学习能力。项目在此结案。

## 实际游玩
```
python human_vs_ai.py --model "checkpoints/az_train/best_model_iter_59.pt"