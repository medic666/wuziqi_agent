"""
utils - 工具函数模块

提供训练系统各模块共用的通用工具函数。

组件:
  - transforms.py: 8向对称变换（D4二面体群）
  - board_image.py: 棋谱图片生成（PNG 导出）
  - zobrist.py: Zobrist 哈希工具

来源: utils.py 拆分 + data_collector.py/pretrain_vs_agent.py Zobrist 代码合并去重
"""