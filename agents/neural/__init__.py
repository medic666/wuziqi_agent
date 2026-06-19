"""
agents.neural - 神经网络智能体子包

当前实现:
  - cnn_v2.py: CNN v9.2, 4×ResBlock(128) + GAP价值头, ~124万参数
  - cnn_v3.py: CNN v9.3, 5×ResBlock(64) + Cross-Attn价值头, ~41万参数
  - transformer.py: GoBangTransformer_v2, 5×Transformer, ~20万参数
  - registry.py: 架构注册表 (单一真理源)
  - az_agent.py: AZAgent 智能体，封装 MCTS + 神经网络推理（架构无关）

架构对比:
    ┌─────────────────┬──────────────┬──────────────┬──────────────────┐
    │                 │ CNN v9.2     │ CNN v9.3     │ Transformer      │
    ├─────────────────┼──────────────┼──────────────┼──────────────────┤
    │ 特征提取        │ Pre-act ResNet│ Pre-act ResNet│ Pre-LN Transformer│
    │ 位置信息        │ 隐式(卷积)   │ 隐式(卷积)   │ 显式 2D-RoPE     │
    │ 价值头          │ Conv→GAP→FC  │ Cross-Attn+MLP│ Cross-Attn+MLP  │
    │ 参数量          │ ~124万       │ ~41万        │ ~20万            │
    │ 输入接口        │ (B,3,15,15)  │ (B,3,15,15)  │ (B,3,15,15)      │
    └─────────────────┴──────────────┴──────────────┴──────────────────┘

使用:
   所有网络通过同一 AZAgent 智能体接入 MCTS/竞技场/训练流程，
   从 checkpoint 的 arch_type 字段自动推断架构。

赛马用法:
    python az_train.py --arch cnn_v2
    python az_train.py --arch cnn_v3
    python az_train.py --arch transformer
"""

# ── 网络模块 ──
from agents.neural.cnn_v2 import ActorCriticNet_v2, ResBlock
from agents.neural.cnn_v3 import ActorCriticNet_v3
from agents.neural.transformer import (
    GoBangTransformer_v2,
    RoPE2D,
    MultiHeadSelfAttention2D,
    TransformerBlock,
)

# ── 注册表 API ──
from agents.neural.registry import (
    NETWORK_REGISTRY,
    ARCH_ALIASES,
    get_network_class,
    get_param_names,
    get_defaults,
    list_architectures,
    resolve_arch,
    infer_arch_from_state_dict,
    register,
)

# ── Agent ──
from agents.neural.az_agent import AZAgent

# ── 向后兼容别名 ──
# 旧代码中 from agents.neural.network import ActorCriticNet 仍可使用
ActorCriticNet = ActorCriticNet_v2

__all__ = [
    # 网络类
    'ActorCriticNet',           # 向后兼容别名 (→ v2)
    'ActorCriticNet_v2',
    'ActorCriticNet_v3',
    'ResBlock',
    'GoBangTransformer_v2',
    'RoPE2D',
    'MultiHeadSelfAttention2D',
    'TransformerBlock',
    # 注册表 API
    'NETWORK_REGISTRY',
    'ARCH_ALIASES',
    'get_network_class',
    'get_param_names',
    'get_defaults',
    'list_architectures',
    'resolve_arch',
    'infer_arch_from_state_dict',
    'register',
    # Agent
    'AZAgent',
]