# gamerules.py
from typing import Optional, Set, Tuple, List
from dataclasses import dataclass, field

@dataclass
class GameState:
    """纯粹的游戏状态，不包含任何智能体私有字段"""
    board: bytearray                    # 15x15, 0=空 1=黑 2=白
    current_player: int                 # 1 黑 / 2 白
    history: List[Tuple[int, int]] = field(default_factory=list)
    last_move: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        if len(self.board) != 225:
            raise ValueError("棋盘大小必须为 225 (15x15)")

class GomokuRules:
    """五子棋规则层，封装所有规则相关的纯函数"""
    
    BOARD_SIZE = 15

    @staticmethod
    def is_valid_move(state: GameState, move: Tuple[int, int]) -> bool:
        r, c = move
        if not (0 <= r < GomokuRules.BOARD_SIZE and 0 <= c < GomokuRules.BOARD_SIZE):
            return False
        if state.board[r * 15 + c] != 0:
            return False
        if GomokuRules.check_winner(state) is not None:
            return False
        return True

    @staticmethod
    def apply_move(state: GameState, move: Tuple[int, int]) -> None:
        r, c = move
        if not GomokuRules.is_valid_move(state, move):
            raise ValueError(f"非法落子: {move}")
        idx = r * 15 + c
        state.board[idx] = state.current_player
        state.history.append(move)
        state.last_move = move
        state.current_player = 3 - state.current_player

    @staticmethod
    def apply_move_fast(state: GameState, move: Tuple[int, int]) -> None:
        """快速落子：跳过合法性验证和胜负检查，专用于 MCTS 和自对弈内部循环"""
        r, c = move
        idx = r * 15 + c
        state.board[idx] = state.current_player
        state.history.append(move)
        state.last_move = move
        state.current_player = 3 - state.current_player

    @staticmethod
    def is_board_full(state: GameState) -> bool:
        """检查棋盘是否已满（用于和棋判定，不依赖 history）"""
        return 0 not in state.board

    @staticmethod
    def check_winner(state: GameState) -> Optional[int]:
        if state.last_move is None:
            return None
        r, c = state.last_move
        board = state.board
        player = board[r * 15 + c]
        if player == 0:
            return None

        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in dirs:
            count = 1
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < 15 and 0 <= nc < 15 and board[nr * 15 + nc] == player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < 15 and 0 <= nc < 15 and board[nr * 15 + nc] == player:
                    count += 1
                else:
                    break
            if count >= 5:
                return player

        # ✅ 改用棋盘满判定，不再依赖 len(state.history)
        if GomokuRules.is_board_full(state):
            return 0
        return None

    @staticmethod
    def get_candidates(state: GameState, radius: int = 2) -> Set[Tuple[int, int]]:
        if not state.history:
            return {(7, 7)}
        
        candidates = set()
        board = state.board
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