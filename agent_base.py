# agent_base.py
from abc import ABC, abstractmethod
from typing import Tuple
from gamerules import GameState

class Agent(ABC):
    """所有智能体的抽象基类，竞技场只依赖此接口"""
    
    @abstractmethod
    def get_move(self, state: GameState) -> Tuple[int, int]:
        """
        根据当前局面返回一步合法的落子坐标。
        注意：此方法不得修改传入的 state，应将其视为只读。
        """
        ...
    
    def reset_incremental_cache(self):
        """新一局开始时调用，清除增量缓存。子类可覆写。"""
        pass