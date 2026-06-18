"""
agents.neural - 神经网络智能体子包

当前实现:
  - network.py: 基于残差网络的 CNN 架构 (ActorCriticNet + ResBlock)
  - transformer_network.py: 基于 Transformer 的架构 (GoBangTransformer_v2 + RoPE2D)
  - az_agent.py: AZAgent 智能体，封装 MCTS + 神经网络推理（架构无关）

架构对比:
  ┌─────────────────┬──────────────────────┬──────────────────────┐
  │                 │ ActorCriticNet (CNN) │ GoBangTransformer_v2 │
  ├─────────────────┼──────────────────────┼──────────────────────┤
  │ 特征提取        │ Pre-activation ResNet│ Pre-LN Transformer   │
  │ 位置信息        │ 隐式(卷积核空间结构) │ 显式 2D-RoPE         │
  │ 全局视野        │ 堆深层等效           │ 自注意力天然全局     │
  │ 参数量(默认)    │ ~124万               │ ~75万                │
  │ 输入接口        │ (B, 3, 15, 15)       │ (B, 3, 15, 15) 兼容  │
  │ 输出接口        │ (B, 225), (B,)       │ (B, 225), (B,) 兼容  │
  └─────────────────┴──────────────────────┴──────────────────────┘

使用:
  两个网络通过同一 AZAgent 智能体接入 MCTS/竞技场/训练流程，
  从 checkpoint 的 arch_type 字段自动推断架构，或通过 network_cls 显式指定。
"""

from agents.neural.network import ActorCriticNet, ResBlock
from agents.neural.transformer_network import (
    GoBangTransformer_v2,
    RoPE2D,
    MultiHeadSelfAttention2D,
    TransformerBlock,
)
from agents.neural.az_agent import AZAgent

__all__ = [
    'ActorCriticNet',
    'ResBlock',
    'GoBangTransformer_v2',
    'RoPE2D',
    'MultiHeadSelfAttention2D',
    'TransformerBlock',
    'AZAgent',
]