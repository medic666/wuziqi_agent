"""
agents - 五子棋智能体层

本包封装所有 AI 智能体实现，通过 agents.base.Agent 抽象基类统一接口。
支持插件式扩展：未来任何新架构只需实现 Agent 接口即可接入整个系统。

当前实现:
  - rule_based.py: 基于博弈树搜索的规则引擎智能体 (ADAgent)
  - neural/: 基于神经网络的 AlphaZero 风格智能体
"""