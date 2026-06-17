"""
search - 搜索算法层

提供蒙特卡洛树搜索 (MCTS) 等博弈搜索算法。
通过 eval_fn 回调与具体智能体/网络解耦，不依赖任何特定网络架构。

当前实现:
  - mcts.py: MCTS + MCTSNode + 辅助函数 (state_to_tensor, create_local_eval_fn)
"""