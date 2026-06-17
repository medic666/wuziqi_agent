"""
agents.neural - 神经网络智能体子包

当前实现: 基于残差网络的 AlphaZero 风格 Actor-Critic 智能体

架构组成:
  - network.py: 神经网络模型定义 (ActorCriticNet + ResBlock)
  - az_agent.py: AZAgent 智能体，封装 MCTS + 神经网络推理

未来扩展: 
  如需添加新架构（如 Transformer），在此目录下新建平行子包即可，
  例如 agents/transformer/ — 只需实现 agents.base.Agent 接口。
"""