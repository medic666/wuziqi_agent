# training/config.py
"""
训练配置类（集中管理）

所有训练入口脚本的配置集中于此，避免散落各处。
每个配置类独立，不互相依赖，可由各自入口脚本按需导入。

合并来源:
  - AlphaZeroConfig: az_train.py L42-104
  - PretrainConfig: pretrain_vs_agent.py L45-108
  - TrainConfig: pre_train.py L30-75
"""

from typing import Optional
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════
#  AlphaZero 训练配置 (az_train.py) — 统一版本支持多架构
# ═══════════════════════════════════════════════════════════════

class AlphaZeroConfig:
    """
    AlphaZero 自我对弈强化学习训练配置 (统一版)。

    支持 CNN v9.2 / v9.3 / Transformer 三种架构，
    通过 arch_type 参数切换。

    使用示例:
        # CNN v9.3 (默认)
        config = AlphaZeroConfig(arch_type='cnn_v3')

        # CNN v9.2
        config = AlphaZeroConfig(arch_type='cnn_v2')

        # Transformer
        config = AlphaZeroConfig(arch_type='transformer')

        # 覆盖架构特定参数
        config = AlphaZeroConfig(arch_type='cnn_v3',
                                 arch_params={'channels': 128, 'num_res_blocks': 6})
    """
    def __init__(
        self,
        # ── ★ 架构选择 (统一入口) ──
        arch_type: str = 'cnn_v3',               # 网络架构: 'cnn_v2' | 'cnn_v3' | 'transformer'
        arch_params: dict = None,                 # 架构参数覆盖 (None=使用注册表默认值)
        # ── 训练循环 ──
        num_iterations: int = 200,              # 总迭代次数
        games_per_iteration: int = 300,          # 每轮自对弈局数
        train_steps_per_iteration: int = 960,     # 每轮训练步数
        baseline_eval_games: int = 100,           # 基准评估局数
        arena_games: int = 100,                   # 竞技场局数
        # ── MCTS 自对弈参数 ──
        num_sims: int = 300,                     # MCTS 模拟次数
        c_puct: float = 1.5,                     # PUCT 探索常数
        dirichlet_alpha: float = 0.2,            # Dirichlet 噪声 alpha
        dirichlet_epsilon: float = 0.25,         # Dirichlet 噪声混合比例
        temp_threshold: int = 4,                 # 温度阈值（步数）
        candidate_radius: int = 2,               # 候选着法半径
        advantage_clip: float = 1.0,             # 优势裁剪范围
        # ── 竞技场参数 ──
        arena_win_threshold: float = 0.56,        # 模型更新阈值
        arena_collapse_threshold: float = 0.35,  # 坍塌检测阈值
        arena_save_image_every_n_games: int = 10, # 竞技场图片保存间隔
        # 核采样参数（竞技场无 MCTS，直接策略头核采样）
        arena_nucleus_p: float = 0.6,            # 核采样 top-p 累积概率阈值
        arena_nucleus_temp_threshold: int = 4,   # 开局高温步数阈值
        arena_nucleus_early_temp: float = 1,   # 开局温度（>1 使分布更平坦，开局更丰富）
        # 竞技场 MCTS 参数（已废弃，核采样化后不再使用）
        arena_c_puct: float = 2.5,               # [deprecated]
        arena_dirichlet_alpha: float = 0.2,      # [deprecated]
        arena_dirichlet_epsilon: float = 0.0,    # [deprecated]
        arena_temperature: float = 1e-3,         # [deprecated]
        # ── 基准评估参数 ──
        baseline_num_sims: int = 400,            # [deprecated] 基准评估 MCTS 模拟次数（核采样化后废弃）
        baseline_agent_depth: int = 4,           # 基准 Agent 搜索深度
        baseline_agent_max_candidates: int = 10, # 基准 Agent 候选数
        # 基准评估核采样参数（无 MCTS，直接策略头核采样）
        baseline_nucleus_p: float = 0.6,            # 基准评估核采样 top-p 阈值
        baseline_nucleus_temp_threshold: int = 0,   # 基准评估开局高温步数阈值
        baseline_nucleus_early_temp: float = 1.5,   # 基准评估开局温度
        # ── 训练参数 ──
        replay_buffer_size: int = 200000,        # 回放缓冲区容量
        min_replay_size: int = 5000,             # 最小训练样本数
        batch_size: int = 128,                   # 批次大小
        learning_rate: float = 1e-4,             # 学习率
        lr_warmup_iterations: int = 5,           # LR 预热迭代数
        weight_decay: float = 1e-4,              # 权重衰减
        grad_clip: float = 1.0,                  # 梯度裁剪
        policy_loss_weight: float = 1.0,         # 策略损失权重
        value_loss_weight: float = 1.0,          # 价值损失权重
        value_loss_delta: float = 0.5,           # HuberLoss delta
        # ── 并行参数 ──
        num_workers: int = 16,                   # Worker 数
        max_batch_size: int = 128,               # 最大批大小
        # ── 存档参数 ──
        checkpoint_dir: str = "checkpoints/az_train",
        save_interval: int = 1,                  # 存档间隔（迭代）
        save_replay_interval: int = 1,           # 回放缓冲区存档间隔
        save_image_every_n_games: int = 10,      # 图片保存间隔
        # ── 设备 ──
        device: str = "auto",
        initial_model: Optional[str] = "checkpoints/joint_pretrain/best_model.pt",
        resume: bool = False,
    ):
        self.arch_type = arch_type

        # 从注册表获取架构默认参数，再用 arch_params 覆盖
        from agents.neural.registry import get_defaults
        self.arch_params = get_defaults(arch_type)
        if arch_params:
            self.arch_params.update(arch_params)

        # 为向后兼容，将架构参数直接设为属性
        for k, v in self.arch_params.items():
            setattr(self, k, v)

        # 将其他所有参数设置为实例属性
        for k, v in locals().items():
            if k not in ('self', 'arch_params'):
                # arch_params 已处理，跳过
                if k == 'arch_type' or k not in self.__dict__:
                    setattr(self, k, v)

        # ★ 将架构名追加到 checkpoint_dir，避免不同架构互相覆盖
        # 如果用户显式指定了非默认路径（包含架构名），不追加
        base_dir = self.checkpoint_dir
        # 默认值 "checkpoints/az_train" 不包含架构子目录
        if f"/{arch_type}" not in base_dir and not base_dir.endswith(f"/{arch_type}"):
            self.checkpoint_dir = f"{base_dir}/{arch_type}"

    def get_model_config(self) -> dict:
        """返回网络模型配置字典（用于构造网络和存档）。"""
        return {'arch_type': self.arch_type, **self.arch_params}

    def to_dict(self) -> dict:
        """导出为字典（用于存档）。"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict):
        """从字典恢复配置。"""
        valid_keys = cls().__dict__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ═══════════════════════════════════════════════════════════════
#  预训练 vs Agent 配置 (pretrain_vs_agent.py)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PretrainConfig:
    """
    预训练配置：神经网络 vs 规则引擎 (AgentAD) 对弈。
    
    对弈即评估，胜率驱动早停，逻辑极简。
    
    支持多架构: 通过 arch_type + arch_params 指定
    """
    # ── 网络架构 ──
    arch_type: str = 'cnn_v3'                   # 网络架构: 'cnn_v2' | 'cnn_v3' | 'transformer'
    arch_params: dict = None                    # 架构参数覆盖 (None=使用注册表默认值)
    # 向后兼容字段 (由 arch_params 自动填充)
    num_res_blocks: int = 5                      # 默认 v9.3
    channels: int = 64                           # 默认 v9.3
    board_size: int = 15

    # ── 预训练轮次 ──
    num_iterations: int = 50
    games_per_iteration: int = 100

    # ── MCTS 参数 ──
    num_sims: int = 400
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.2
    dirichlet_epsilon: float = 0        # 关掉噪声，因为对手 agent 比较随机
    temp_threshold: int = 6
    candidate_radius: int = 3
    advantage_clip: float = 1.0

    # ── AgentAD 对手参数 ──
    agent_depth: int = 4
    agent_max_candidates: int = 10
    agent_use_quiescence: bool = True
    agent_vct_depth: int = 8

    # ── 训练参数 ──
    replay_buffer_size: int = 500000
    min_replay_size: int = 5000
    batch_size: int = 128
    train_steps_per_iteration: int = 40
    learning_rate: float = 1e-4
    lr_warmup_iterations: int = 3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    value_loss_delta: float = 0.5

    # ── 早停参数 ──
    early_stop_patience: int = 15
    early_stop_min_delta: float = 0.02

    # ── 并行参数 ──
    num_workers: int = 16
    max_batch_size: int = 128

    # ── 存档参数 ──
    checkpoint_dir: str = "checkpoints/pretrain_vs_agent"
    initial_model: Optional[str] = "checkpoints/pretrain_vs_agent/best_model_old.pt"

    # ── 图片保存 ──
    save_images: bool = True
    save_image_every_n_games: int = 10

    # ── 置换表参数 ──
    tt_save_interval: int = 10
    tt_inherit_from_worker0: bool = True

    def __post_init__(self):
        """追加 arch_type 到 checkpoint_dir，避免不同架构互相覆盖"""
        if f"/{self.arch_type}" not in self.checkpoint_dir and not self.checkpoint_dir.endswith(f"/{self.arch_type}"):
            self.checkpoint_dir = f"{self.checkpoint_dir}/{self.arch_type}"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = cls().__dict__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ═══════════════════════════════════════════════════════════════
#  Transformer 训练配置 — 已废弃
#  请使用 AlphaZeroConfig(arch_type='transformer') 替代
#  TransformerConfig 类已移除，保留此注释供参考
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
#  预训练配置 (pre_train.py) — 从数据学习
# ═══════════════════════════════════════════════════════════════

class TrainConfig:
    """
    联合预训练配置：从收集的数据集中学习策略和价值。
    
    使用 Behavior Cloning + Value Regression，
    从 data_collector.py 产出的数据中学习。

    支持多架构: 通过 arch_type + arch_params 指定
    """
    def __init__(
        self,
        data_path: str = "collected_data/training_data.npz",
        val_ratio: float = 0.1,                  # 验证集比例
        max_samples: int = 0,                    # 最大样本数（0=全部）
        # ── 网络架构 ──
        arch_type: str = 'cnn_v3',               # 网络架构: 'cnn_v2' | 'cnn_v3' | 'transformer'
        arch_params: dict = None,                 # 架构参数覆盖
        # 向后兼容字段 (由 arch_params 自动填充)
        num_res_blocks: int = 5,
        channels: int = 64,
        board_size: int = 15,
        # ── 训练参数 ──
        batch_size: int = 128,
        max_epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        actor_loss_weight: float = 1.0,          # 策略损失权重
        critic_loss_weight: float = 1.0,         # 价值损失权重
        loss_type: str = "huber",                # 损失类型: "huber" or "mse"
        grad_clip: float = 1.0,
        scheduler_type: str = "cosine",          # 学习率调度: "cosine" or "plateau"
        warmup_epochs: int = 5,
        patience: int = 15,                      # 早停耐心
        min_delta: float = 1e-5,                # 早停最小改善
        checkpoint_dir: str = "checkpoints/joint_pretrain",
        save_interval: int = 5,
        device: str = "auto",
        num_workers: int = 8,                    # DataLoader worker数
        pin_memory: bool = True,
        resume: bool = False,
        resume_path: Optional[str] = None,
        load_weights: Optional[str] = None,      # 仅加载权重
    ):
        # 将全部参数设置为实例属性
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)

        # ★ 追加 arch_type 到 checkpoint_dir
        if f"/{arch_type}" not in self.checkpoint_dir and not self.checkpoint_dir.endswith(f"/{arch_type}"):
            self.checkpoint_dir = f"{self.checkpoint_dir}/{arch_type}"

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = cls().__dict__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})