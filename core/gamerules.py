# core/gamerules.py
"""
五子棋核心游戏规则模块

定义纯粹的游戏状态(GameState)和规则层(GomokuRules)。
职责边界:
  - GameState: 只存游戏数据，不包含任何智能体私有字段
  - GomokuRules: 封装所有规则相关的纯函数（落子、胜负判定、候选生成）
  
原则: 此模块为整个系统的基础层，不应依赖任何其他项目模块。
"""

from typing import Optional, Set, Tuple, List
from dataclasses import dataclass, field


@dataclass
class GameState:
    """
    纯粹的游戏状态数据容器。
    
    存储当前棋盘、轮到谁走、落子历史等完整游戏信息。
    注意：此类不包含任何智能体私有字段（如评估分数、置换表等）。
    
    Attributes:
        board: 15x15 棋盘，扁平化存储 (bytearray)，0=空 1=黑 2=白
        current_player: 当前轮到谁走，1=黑方, 2=白方
        history: 落子序列 [(r1,c1), (r2,c2), ...]
        last_move: 上一步落子坐标，开局时为 None
    """
    board: bytearray                    # 15x15, 0=空 1=黑 2=白
    current_player: int                 # 1 黑 / 2 白
    history: List[Tuple[int, int]] = field(default_factory=list)
    last_move: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        """构造后验证：确保棋盘大小为 225 (15x15)"""
        if len(self.board) != 225:
            raise ValueError("棋盘大小必须为 225 (15x15)")


class GomokuRules:
    """
    五子棋规则层，封装所有规则相关的纯函数。
    
    所有方法均为静态方法，不持有任何状态。
    提供落子验证、执行落子、胜负判定、候选走法生成等功能。
    """
    
    BOARD_SIZE = 15  # 标准五子棋棋盘大小

    # ═══════════════════════════════════════════════════════════
    #  落子操作
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def is_valid_move(state: GameState, move: Tuple[int, int]) -> bool:
        """
        验证落子是否合法。
        
        检查条件:
          1. 坐标在棋盘范围内 (0-14)
          2. 目标位置为空
          3. 游戏尚未结束（无胜者）
        
        Args:
            state: 当前游戏状态
            move: 候选落子 (行, 列)
            
        Returns:
            是否可合法落子
        """
        r, c = move
        # 检查坐标范围
        if not (0 <= r < GomokuRules.BOARD_SIZE and 0 <= c < GomokuRules.BOARD_SIZE):
            return False
        # 检查目标位置为空
        if state.board[r * 15 + c] != 0:
            return False
        # 检查游戏是否已结束
        if GomokuRules.check_winner(state) is not None:
            return False
        return True

    @staticmethod
    def apply_move(state: GameState, move: Tuple[int, int]) -> None:
        """
        执行落子操作（带合法性验证）。
        
        适用于外部调用（如人类落子、竞技场等高安全性场景）。
        内部会调用 is_valid_move 验证。
        
        Args:
            state: 当前游戏状态（原地修改）
            move: 落子坐标 (行, 列)
            
        Raises:
            ValueError: 如果落子不合法
        """
        r, c = move
        if not GomokuRules.is_valid_move(state, move):
            raise ValueError(f"非法落子: {move}")
        idx = r * 15 + c
        state.board[idx] = state.current_player
        state.history.append(move)
        state.last_move = move
        state.current_player = 3 - state.current_player  # 切换执子方 (1->2, 2->1)

    @staticmethod
    def apply_move_fast(state: GameState, move: Tuple[int, int]) -> None:
        """
        快速落子：跳过合法性验证和胜负检查。
        
        专用于 MCTS 和自对弈内部循环等高性能场景。
        调用方需自行保证落子合法性。
        
        Args:
            state: 当前游戏状态（原地修改）
            move: 落子坐标 (行, 列)
        """
        r, c = move
        idx = r * 15 + c
        state.board[idx] = state.current_player
        state.history.append(move)
        state.last_move = move
        state.current_player = 3 - state.current_player

    # ═══════════════════════════════════════════════════════════
    #  胜负判定
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def is_board_full(state: GameState) -> bool:
        """
        检查棋盘是否已满。
        
        用于和棋判定，不依赖 history 长度。
        性能：O(225) 扫描，仅在需要时调用。
        
        Returns:
            True 表示棋盘已无空位
        """
        return 0 not in state.board

    @staticmethod
    def check_winner(state: GameState) -> Optional[int]:
        """
        检查当前游戏是否有胜者。
        
        仅检查 last_move 处的棋子是否形成五连（优化：不遍历全盘）。
        四个方向：水平、垂直、主对角线、反对角线。
        
        Args:
            state: 当前游戏状态
            
        Returns:
            None: 游戏继续
            0: 平局（棋盘已满）
            1: 黑方胜
            2: 白方胜
        """
        # 无 last_move 说明尚未落子，不可能有胜者
        if state.last_move is None:
            return None
        
        r, c = state.last_move
        board = state.board
        player = board[r * 15 + c]
        
        # 防御：last_move 处不应为空
        if player == 0:
            return None

        # 检查四个方向
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in dirs:
            count = 1  # 包含 last_move 本身
            
            # 正向延伸
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < 15 and 0 <= nc < 15 and board[nr * 15 + nc] == player:
                    count += 1
                else:
                    break
            
            # 反向延伸
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < 15 and 0 <= nc < 15 and board[nr * 15 + nc] == player:
                    count += 1
                else:
                    break
            
            # 五连即胜
            if count >= 5:
                return player

        # 检查平局（棋盘已满）
        if GomokuRules.is_board_full(state):
            return 0
        
        return None

    # ═══════════════════════════════════════════════════════════
    #  候选走法生成
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def get_candidates(state: GameState, radius: int = 2) -> Set[Tuple[int, int]]:
        """
        生成候选走法集合。
        
        策略：以所有已有棋子为中心，半径 radius 范围内的空位作为候选。
        这样避免了全盘扫描，同时保证了所有有意义的走法都被覆盖。
        
        特殊情况：空棋盘时返回天元 (7, 7) 作为唯一候选。
        
        Args:
            state: 当前游戏状态
            radius: 搜索半径（默认2，即距离已有棋子曼哈顿距离≤2的空位）
            
        Returns:
            候选走法坐标集合
        """
        # 空棋盘：返回天元
        if not state.history:
            return {(7, 7)}
        
        candidates = set()
        board = state.board
        
        # 遍历所有已有棋子，生成半径范围内的空位
        for idx in range(225):
            if board[idx] == 0:
                continue
            r, c = idx // 15, idx % 15
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 15 and 0 <= nc < 15 and board[nr * 15 + nc] == 0:
                        candidates.add((nr, nc))
        
        return candidates