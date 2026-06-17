# agents/base.py
"""
智能体抽象基类

定义所有智能体的统一接口。竞技场、训练流程等上层模块只依赖此接口，
不关心具体实现（规则引擎 or 神经网络）。

设计原则:
  - 接口极简：只暴露 get_move() 一个核心方法
  - 状态只读：get_move() 承诺不修改传入的 GameState
  - 生命周期管理：reset_incremental_cache() 用于新局开始时清理缓存
"""

from abc import ABC, abstractmethod
from typing import Tuple
from core.gamerules import GameState


class Agent(ABC):
    """
    所有智能体的抽象基类。
    
    上层模块（竞技场、数据收集器、训练流程）仅依赖此接口，
    不关心具体是规则引擎、神经网络还是未来的新架构。
    
    子类必须实现:
      - get_move(state) -> (r, c): 根据局面返回合法落子坐标
    
    子类可选覆写:
      - reset_incremental_cache(): 新一局开始时清理增量缓存
    """
    
    @abstractmethod
    def get_move(self, state: GameState) -> Tuple[int, int]:
        """
        根据当前局面返回一步合法的落子坐标。
        
        重要约定:
          - 此方法不得修改传入的 state，应将其视为只读
          - 返回值必须是棋盘上的合法空位
        
        Args:
            state: 当前游戏状态（只读）
            
        Returns:
            (行, 列) 落子坐标
        """
        ...
    
    def reset_incremental_cache(self):
        """
        新一局开始时调用，清除增量缓存。
        
        默认空实现。子类如使用增量更新（如置换表、MCTS树等），
        应覆写此方法来重置自身状态。
        """
        pass