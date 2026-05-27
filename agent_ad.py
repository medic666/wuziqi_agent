# agent_ad.py 融合最终版 (评估函数模式匹配重写版 + 增量缓存优化 + VCT防卡死修复)

import random
from typing import List, Tuple, Optional, Dict, Set
from collections import OrderedDict
from dataclasses import dataclass, field
from agent_base import Agent as BaseAgent
from gamerules import GameState


@dataclass
class PrivateState:
    plate: bytearray
    player: int
    score_black: int = 0
    score_white: int = 0
    candidates: Set[Tuple[int, int]] = field(default_factory=set)
    history: List[Tuple[int, int]] = field(default_factory=list)
    zhash: int = 0


_ZOBRIST_RNG = random.Random(42)
_ZOBRIST_TABLE = [[[_ZOBRIST_RNG.getrandbits(64) for _ in range(3)] for _ in range(15)] for _ in range(15)]


class Agent(BaseAgent):
    ZOBRIST_TABLE = _ZOBRIST_TABLE

    SCORE_FIVE = 10000000
    SCORE_LIVE_FOUR = 100000
    SCORE_RUSH_FOUR = 10000
    SCORE_LIVE_THREE = 5000
    SCORE_SLEEP_THREE = 500
    SCORE_LIVE_TWO = 200
    SCORE_SLEEP_TWO = 20
    SCORE_LIVE_ONE = 10
    SCORE_SLEEP_ONE = 1

    _PATTERNS = [
        ('11111',  SCORE_FIVE),
        ('011110', SCORE_LIVE_FOUR),
        ('211110', SCORE_RUSH_FOUR), ('011112', SCORE_RUSH_FOUR),
        ('10111',  SCORE_RUSH_FOUR), ('11011',  SCORE_RUSH_FOUR), ('11101',  SCORE_RUSH_FOUR),
        ('001110', SCORE_LIVE_THREE), ('011100', SCORE_LIVE_THREE),
        ('010110', SCORE_LIVE_THREE), ('011010', SCORE_LIVE_THREE),
        ('211100', SCORE_SLEEP_THREE), ('001112', SCORE_SLEEP_THREE),
        ('210110', SCORE_SLEEP_THREE), ('011012', SCORE_SLEEP_THREE),
        ('201110', SCORE_SLEEP_THREE), ('011102', SCORE_SLEEP_THREE),
        ('211010', SCORE_SLEEP_THREE), ('010112', SCORE_SLEEP_THREE),
        ('10011',  SCORE_SLEEP_THREE), ('11001',  SCORE_SLEEP_THREE), ('10101', SCORE_SLEEP_THREE),
        ('000110', SCORE_LIVE_TWO), ('001100', SCORE_LIVE_TWO), ('011000', SCORE_LIVE_TWO),
        ('001010', SCORE_LIVE_TWO), ('010100', SCORE_LIVE_TWO), ('010010', SCORE_LIVE_TWO),
        ('200110', SCORE_SLEEP_TWO), ('001102', SCORE_SLEEP_TWO), ('011002', SCORE_SLEEP_TWO),
        ('201010', SCORE_SLEEP_TWO), ('010102', SCORE_SLEEP_TWO),
        ('210010', SCORE_SLEEP_TWO), ('010012', SCORE_SLEEP_TWO),
        ('201100', SCORE_SLEEP_TWO),
        ('20011',  SCORE_SLEEP_TWO), ('11002',  SCORE_SLEEP_TWO), ('10001', SCORE_SLEEP_TWO),
    ]

    OPENING_BOOK = {
        (7,6): [{"name": "浦月", "b3": (6,6), "w4": [(6,7), (8,7), (5,5), (7,7)]},
                 {"name": "瑞星", "b3": (7,8), "w4": [(6,7), (8,7), (7,7), (8,8)]},
                 {"name": "峡月", "b3": (6,8), "w4": [(5,7), (7,7), (5,8), (6,7)]},
                 {"name": "水月", "b3": (8,6), "w4": [(8,7), (7,7), (9,5)]}],
        (6,7): [{"name": "花月", "b3": (6,6), "w4": [(7,6), (5,6), (7,7), (8,6)]},
                 {"name": "新月", "b3": (8,7), "w4": [(7,6), (7,7), (8,6), (8,8)]},
                 {"name": "恒月", "b3": (8,6), "w4": [(7,6), (7,7), (9,6)]},
                 {"name": "岚月", "b3": (5,6), "w4": [(6,6), (7,6), (4,7)]}],
        (8,7): [{"name": "云月", "b3": (7,6), "w4": [(6,6), (8,8), (7,7), (8,6)]},
                 {"name": "山月", "b3": (6,7), "w4": [(7,8), (7,7), (6,6), (5,7)]},
                 {"name": "松月", "b3": (7,8), "w4": [(8,8), (7,7), (6,7), (6,8)]}],
        (7,8): [{"name": "雨月", "b3": (6,7), "w4": [(6,6), (8,8), (7,7), (8,7)]},
                 {"name": "丘月", "b3": (7,6), "w4": [(8,7), (7,7), (8,8), (6,7)]},
                 {"name": "望月", "b3": (6,8), "w4": [(5,7), (7,7), (5,8)]}],
        (6,6): [{"name": "寒星", "b3": (7,6), "w4": [(6,7), (8,7), (7,7), (8,6)]},
                 {"name": "彗星", "b3": (8,8), "w4": [(7,6), (7,8), (7,7)]},
                 {"name": "银月", "b3": (6,8), "w4": [(7,7), (5,6), (5,8)]},
                 {"name": "夏月", "b3": (8,6), "w4": [(7,7), (8,7), (7,6)]}],
        (8,6): [{"name": "溪月", "b3": (7,6), "w4": [(6,7), (8,7), (7,7), (6,6)]},
                 {"name": "明星", "b3": (6,8), "w4": [(7,8), (7,7), (5,8)]},
                 {"name": "晨月", "b3": (8,8), "w4": [(7,7), (7,8), (7,6)]}],
        (8,8): [{"name": "金星", "b3": (7,8), "w4": [(8,7), (7,6), (7,7), (6,8)]},
                 {"name": "疏星", "b3": (7,6), "w4": [(6,7), (8,7), (7,7), (6,6)]},
                 {"name": "游星", "b3": (6,6), "w4": [(7,8), (6,7), (5,5)]},
                 {"name": "春月", "b3": (6,8), "w4": [(7,7), (5,8), (7,8)]}],
        (6,8): [{"name": "残月", "b3": (7,8), "w4": [(8,7), (7,7), (8,8)]},
                 {"name": "斜月", "b3": (8,6), "w4": [(7,6), (7,7), (9,5)]},
                 {"name": "汀月", "b3": (6,6), "w4": [(7,7), (5,7), (5,5)]}],
        (7,5): [{"name": "长星", "b3": (7,6), "w4": [(6,6), (8,6), (7,7), (6,5)]},
                 {"name": "远星", "b3": (6,6), "w4": [(7,6), (5,5), (6,7)]},
                 {"name": "极星", "b3": (8,6), "w4": [(7,6), (8,7), (7,7)]}],
        (5,7): [{"name": "辰星", "b3": (6,7), "w4": [(6,6), (6,8), (7,7), (5,6)]},
                 {"name": "幽星", "b3": (6,6), "w4": [(7,6), (6,7), (5,6)]}],
        (9,7): [{"name": "巨星", "b3": (8,7), "w4": [(8,6), (8,8), (7,7), (9,6)]},
                 {"name": "耀星", "b3": (8,8), "w4": [(8,7), (7,8), (7,7)]}],
        (7,9): [{"name": "流星", "b3": (7,8), "w4": [(6,8), (8,8), (7,7), (7,10)]},
                 {"name": "影星", "b3": (8,8), "w4": [(7,8), (8,7), (7,7)]}],
    }

    def __init__(self, depth: int = 4, max_candidates: int = 10, use_quiescence: bool = True,
                 vct_depth: int = 8, quiescence_depth: int = 2, max_trans_size: int = 1_000_000,
                 max_nodes: int = 500000, name: str = "ADAgent"):
        self.name = name
        self.base_depth = depth
        self.base_max_candidates = max_candidates
        self.depth = depth
        self.max_candidates = max_candidates
        self.use_quiescence = use_quiescence
        self.vct_depth = vct_depth
        self.quiescence_depth = quiescence_depth
        self.max_trans_size = max_trans_size
        self.max_nodes = max_nodes

        self.trans_table = OrderedDict()
        self.node_count = 0
        self.timeout = False

        self._chosen_opening = None
        self._opening_step = 0

        # ✅ 增量更新缓存
        self._prev_private_state: Optional[PrivateState] = None
        self._prev_history_len: int = -1

    # ==================== 增量更新核心 ====================

    def reset_incremental_cache(self):
        """新一局开始时调用，清除增量缓存"""
        self._prev_private_state = None
        self._prev_history_len = -1

    def get_move(self, state: GameState) -> Tuple[int, int]:
        self.depth = self.base_depth
        self.max_candidates = self.base_max_candidates
        work = self._get_or_build_private_state(state)
        move = self._reaction(work)
        self._prev_private_state = work
        self._prev_history_len = len(state.history)
        return move

    def _get_or_build_private_state(self, state: GameState) -> PrivateState:
        """增量构建 PrivateState，支持 1~N 步增量更新"""
        prev = self._prev_private_state
        hist = state.history
        prev_len = self._prev_history_len

        if prev is not None and prev_len >= 0 and len(hist) > prev_len:
            new_moves = hist[prev_len:]

            valid = True
            for move in new_moves:
                r, c = move
                if not (0 <= r < 15 and 0 <= c < 15) or prev.plate[r * 15 + c] != 0:
                    valid = False
                    break

            if valid:
                undo_stack = []
                try:
                    for move in new_moves:
                        undo_info = self.apply_move_full(prev, move)
                        undo_stack.append((move, undo_info))

                    if prev.player == state.current_player:
                        return prev
                    else:
                        for move, (oh, db, dw, add) in reversed(undo_stack):
                            self.undo_move_full(prev, move, oh, db, dw, add)
                except Exception:
                    for move, (oh, db, dw, add) in reversed(undo_stack):
                        try:
                            self.undo_move_full(prev, move, oh, db, dw, add)
                        except Exception:
                            break

        work = self._build_private_state(state)
        return work

    # ==================== 以下与原版完全一致 ====================

    def _build_private_state(self, state: GameState) -> PrivateState:
        plate = bytearray(state.board)
        zhash = 0
        for r in range(15):
            for c in range(15):
                p = plate[r * 15 + c]
                if p != 0:
                    zhash ^= self.ZOBRIST_TABLE[r][c][p]
        candidates = self._generate_candidates(plate, state.history)
        score_black, score_white = self._compute_global_score(plate)
        return PrivateState(plate=plate, player=state.current_player,
                            score_black=score_black, score_white=score_white,
                            candidates=candidates, history=list(state.history), zhash=zhash)

    def _generate_candidates(self, plate, history):
        if not history:
            return {(7, 7)}
        cand = set()
        for r in range(15):
            for c in range(15):
                if plate[r * 15 + c] != 0: continue
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 15 and 0 <= nc < 15 and plate[nr * 15 + nc] != 0:
                            cand.add((r, c)); break
                    else: continue
                    break
        return cand

    def _compute_global_score(self, plate):
        score_b, score_w = 0, 0
        for r in range(15):
            line = [plate[r * 15 + c] for c in range(15)]
            sb, sw = self._evaluate_line(line)
            score_b += sb; score_w += sw
        for c in range(15):
            line = [plate[r * 15 + c] for r in range(15)]
            sb, sw = self._evaluate_line(line)
            score_b += sb; score_w += sw
        for d in range(-10, 11):
            r, c = max(0, d), max(0, -d)
            line = []
            while r < 15 and c < 15:
                line.append(plate[r * 15 + c]); r += 1; c += 1
            if len(line) >= 5:
                sb, sw = self._evaluate_line(line)
                score_b += sb; score_w += sw
        for d in range(4, 25):
            r, c = max(0, d - 14), min(14, d)
            line = []
            while r < 15 and c >= 0:
                line.append(plate[r * 15 + c]); r += 1; c -= 1
            if len(line) >= 5:
                sb, sw = self._evaluate_line(line)
                score_b += sb; score_w += sw
        return score_b, score_w

    def _evaluate_line(self, line):
        score_b, score_w = 0, 0
        n = len(line)
        if n < 5:
            return 0, 0
        for player in [1, 2]:
            s = ['2']
            for p in line:
                if p == player: s.append('1')
                elif p == 0: s.append('0')
                else: s.append('2')
            s.append('2')
            s_str = ''.join(s)
            total_len = len(s_str)
            score = 0
            used = [False] * total_len
            for pattern, value in self._PATTERNS:
                plen = len(pattern)
                start = 0
                while True:
                    idx = s_str.find(pattern, start)
                    if idx == -1: break
                    overlap = False
                    for k in range(idx, idx + plen):
                        if used[k]:
                            overlap = True; break
                    if not overlap:
                        score += value
                        for k in range(idx, idx + plen):
                            used[k] = True
                    start = idx + 1
            if player == 1: score_b = score
            else: score_w = score
        return score_b, score_w

    @staticmethod
    def _is_terminal(state, last_move):
        r, c = last_move
        plate = state.plate
        player = plate[r * 15 + c]
        if player == 0: return None
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count = 1
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < 15 and 0 <= nc < 15 and plate[nr * 15 + nc] == player: count += 1
                else: break
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < 15 and 0 <= nc < 15 and plate[nr * 15 + nc] == player: count += 1
                else: break
            if count >= 5: return player
        if len(state.history) == 225: return 0
        return None

    def _quick_evaluate_point(self, state, r, c, player):
        best_score = 0
        plate = state.plate
        for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            count, block = 1, 0
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < 15 and 0 <= nc < 15:
                    if plate[nr * 15 + nc] == player: count += 1
                    else:
                        if plate[nr * 15 + nc] != 0: block += 1
                        break
                else: block += 1; break
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < 15 and 0 <= nc < 15:
                    if plate[nr * 15 + nc] == player: count += 1
                    else:
                        if plate[nr * 15 + nc] != 0: block += 1
                        break
                else: block += 1; break
            if count >= 5: return self.SCORE_FIVE
            score = 0
            if count == 4: score = self.SCORE_LIVE_FOUR if block == 0 else (self.SCORE_RUSH_FOUR if block == 1 else 0)
            elif count == 3: score = self.SCORE_LIVE_THREE if block == 0 else (self.SCORE_SLEEP_THREE if block == 1 else 0)
            elif count == 2: score = self.SCORE_LIVE_TWO if block == 0 else (self.SCORE_SLEEP_TWO if block == 1 else 0)
            elif count == 1: score = self.SCORE_LIVE_ONE if block == 0 else (self.SCORE_SLEEP_ONE if block == 1 else 0)
            if score > best_score: best_score = score
        return best_score

    def _sort_candidates(self, state, trans_best=None):
        player, opp = state.player, 3 - state.player
        win_fives, def_fives = [], []
        win_fours, def_fours = [], []
        scored = []
        for act in state.candidates:
            my = self._quick_evaluate_point(state, act[0], act[1], player)
            opp_val = self._quick_evaluate_point(state, act[0], act[1], opp)
            if my >= self.SCORE_FIVE: win_fives.append(act)
            elif opp_val >= self.SCORE_FIVE: def_fives.append(act)
            elif my >= self.SCORE_LIVE_FOUR: win_fours.append(act)
            elif opp_val >= self.SCORE_LIVE_FOUR: def_fours.append(act)
            else: scored.append((max(my, int(opp_val * 1.1)), act))
        random.shuffle(win_fives); random.shuffle(def_fives)
        random.shuffle(win_fours); random.shuffle(def_fours)
        if win_fives: result = list(win_fives)
        elif def_fives: result = list(def_fives)
        elif def_fours: result = list(def_fours)
        elif win_fours: result = list(win_fours)
        else:
            random.shuffle(scored)
            scored.sort(reverse=True, key=lambda x: x[0])
            dynamic_max = self.max_candidates
            if len(state.history) < 12: dynamic_max = min(self.max_candidates + 5, 20)
            result = [act for _, act in scored[:dynamic_max]]
        if trans_best is not None and trans_best in result:
            result.remove(trans_best); result.insert(0, trans_best)
        elif trans_best is not None:
            result.insert(0, trans_best)
            if len(result) > self.max_candidates + 5: result.pop()
        return result

    def _gen_threat_moves(self, state):
        moves = []
        player, opp = state.player, 3 - state.player
        for r, c in state.candidates:
            my = self._quick_evaluate_point(state, r, c, player)
            opp_val = self._quick_evaluate_point(state, r, c, opp)
            score = max(my, opp_val)
            if score >= self.SCORE_RUSH_FOUR: moves.append((score, r, c))
        random.shuffle(moves)
        moves.sort(reverse=True, key=lambda x: x[0])
        return [(r, c) for _, r, c in moves[:8]]

    def _find_five_moves(self, state, player):
        fives = []
        plate = state.plate
        for r, c in state.candidates:
            # ✅ 防卡死：节点计数与超时退出
            self.node_count += 1
            if self.node_count > self.max_nodes or self.timeout:
                self.timeout = True
                return []

            idx = r * 15 + c
            if plate[idx] != 0: continue
            if self._quick_evaluate_point(state, r, c, player) >= self.SCORE_FIVE:
                plate[idx] = player
                if self._is_terminal(state, (r, c)) == player: fives.append((r, c))
                plate[idx] = 0
        random.shuffle(fives)
        return fives

    def _get_fours_and_defenses_v2(self, state, player):
        potential_fours = []
        plate = state.plate
        for r, c in state.candidates:
            # ✅ 防卡死：节点计数与超时退出
            self.node_count += 1
            if self.node_count > self.max_nodes or self.timeout:
                self.timeout = True
                return []

            if plate[r * 15 + c] != 0: continue
            if self._quick_evaluate_point(state, r, c, player) >= self.SCORE_RUSH_FOUR:
                potential_fours.append((r, c))
        if len(potential_fours) > 15: potential_fours = potential_fours[:15]
        fours = []
        for r, c in potential_fours:
            # ✅ 防卡死：循环前检查超时
            if self.timeout: return []

            idx = r * 15 + c
            plate[idx] = player
            five_moves = []
            for r2, c2 in state.candidates:
                # ✅ 防卡死：内层节点计数与超时退出
                self.node_count += 1
                if self.node_count > self.max_nodes or self.timeout:
                    self.timeout = True
                    plate[idx] = 0
                    return []

                idx2 = r2 * 15 + c2
                if plate[idx2] != 0: continue
                if self._quick_evaluate_point(state, r2, c2, player) >= self.SCORE_FIVE:
                    plate[idx2] = player
                    if self._is_terminal(state, (r2, c2)) == player: five_moves.append((r2, c2))
                    plate[idx2] = 0
            plate[idx] = 0
            if five_moves: fours.append(((r, c), list(set(five_moves))))
        return fours

    def _vct_search(self, state, depth):
        # ✅ 防卡死：节点计数与超时退出
        self.node_count += 1
        if self.node_count > self.max_nodes or self.timeout:
            self.timeout = True
            return None

        if depth <= 0: return None
        player = state.player; opp = 3 - player

        # ✅ 防卡死：递归内部检查超时
        if self.timeout: return None

        fives = self._find_five_moves(state, player)
        if self.timeout: return None  # ✅ 防卡死：子函数超时检查
        if fives: return random.choice(fives)

        opp_fives = self._find_five_moves(state, opp)
        if self.timeout: return None  # ✅ 防卡死：子函数超时检查
        if opp_fives: return None

        fours = self._get_fours_and_defenses_v2(state, player)
        if self.timeout: return None  # ✅ 防卡死：子函数超时检查
        if not fours: return None

        winning_moves = []
        random.shuffle(fours)
        for move, defenses in fours:
            # ✅ 防卡死：循环前检查超时
            if self.timeout: return None

            oh, db, dw, add = self.apply_move_full(state, move)
            if self._find_five_moves(state, opp):
                self.undo_move_full(state, move, oh, db, dw, add); continue
            if self.timeout:  # ✅ 防卡死：子函数超时检查
                self.undo_move_full(state, move, oh, db, dw, add); return None

            if len(defenses) >= 2:
                self.undo_move_full(state, move, oh, db, dw, add)
                winning_moves.append(move); continue
            def_move = defenses[0]
            if state.plate[def_move[0] * 15 + def_move[1]] != 0:
                self.undo_move_full(state, move, oh, db, dw, add); continue
            oh2, db2, dw2, add2 = self.apply_move_full(state, def_move)
            result = self._vct_search(state, depth - 1)
            self.undo_move_full(state, def_move, oh2, db2, dw2, add2)
            if self.timeout:  # ✅ 防卡死：深层递归超时退出
                self.undo_move_full(state, move, oh, db, dw, add); return None
            if result is not None:
                self.undo_move_full(state, move, oh, db, dw, add)
                winning_moves.append(move); continue
            self.undo_move_full(state, move, oh, db, dw, add)
        if winning_moves: return random.choice(winning_moves)
        return None

    def _extract_line(self, plate, r, c, dr, dc, override=None):
        line = []
        nr, nc = r - dr, c - dc
        while 0 <= nr < 15 and 0 <= nc < 15:
            line.append(plate[nr * 15 + nc]); nr -= dr; nc -= dc
        line.reverse()
        line.append(plate[r * 15 + c] if override is None else override)
        nr, nc = r + dr, c + dc
        while 0 <= nr < 15 and 0 <= nc < 15:
            line.append(plate[nr * 15 + nc]); nr += dr; nc += dc
        return line

    def apply_move_full(self, state, act):
        r, c = act
        player = state.player
        idx = r * 15 + c
        old_hash = state.zhash
        old_scores = [self._evaluate_line(self._extract_line(state.plate, r, c, dr, dc))
                      for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]]
        state.plate[idx] = player
        state.zhash ^= Agent.ZOBRIST_TABLE[r][c][player]
        delta_b, delta_w = 0, 0
        for i, (dr, dc) in enumerate([(1, 0), (0, 1), (1, 1), (1, -1)]):
            new_sb, new_sw = self._evaluate_line(self._extract_line(state.plate, r, c, dr, dc))
            delta_b += new_sb - old_scores[i][0]
            delta_w += new_sw - old_scores[i][1]
        state.score_black += delta_b; state.score_white += delta_w
        added_candidates = set()
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 15 and 0 <= nc < 15 and state.plate[nr * 15 + nc] == 0:
                    if (nr, nc) not in state.candidates: added_candidates.add((nr, nc))
                    state.candidates.add((nr, nc))
        state.candidates.discard((r, c))
        state.history.append((r, c))
        state.player = 3 - player
        return old_hash, delta_b, delta_w, added_candidates

    def undo_move_full(self, state, act, old_hash, delta_b, delta_w, added_cand):
        r, c = act
        state.player = 3 - state.player
        state.score_black -= delta_b; state.score_white -= delta_w
        state.plate[r * 15 + c] = 0
        state.zhash = old_hash
        state.history.pop()
        state.candidates.add((r, c))
        for nr, nc in added_cand:
            has_neighbor = False
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if 0 <= nr + dr < 15 and 0 <= nc + dc < 15 and state.plate[(nr + dr) * 15 + (nc + dc)] != 0:
                        has_neighbor = True; break
                if has_neighbor: break
            if not has_neighbor: state.candidates.discard((nr, nc))

    def _evaluate_state(self, state):
        if state.player == 2: return state.score_white - state.score_black
        else: return state.score_black - state.score_white

    def _quiescence(self, state, alpha, beta, depth):
        self.node_count += 1
        if self.node_count > self.max_nodes:
            self.timeout = True
            return self._evaluate_state(state)
        stand_pat = self._evaluate_state(state)
        if depth == 0: return stand_pat
        if stand_pat >= beta: return beta
        if stand_pat > alpha: alpha = stand_pat
        moves = self._gen_threat_moves(state)
        if not moves: return stand_pat
        for act in moves:
            oh, db, dw, add = self.apply_move_full(state, act)
            val = -self._quiescence(state, -beta, -alpha, depth - 1)
            self.undo_move_full(state, act, oh, db, dw, add)
            if self.timeout: return alpha
            if val >= beta: return beta
            if val > alpha: alpha = val
        return alpha

    def _lookup_trans(self, state, depth):
        key = state.zhash
        if key in self.trans_table:
            tt_depth, flag, tt_val, tt_move = self.trans_table[key]
            if tt_depth >= depth:
                self.trans_table.move_to_end(key)
                return flag, tt_val, tt_move
            else:
                self.trans_table.move_to_end(key)
                return None, None, tt_move
        return None, None, None

    def _store_trans(self, state, depth, flag, value, move):
        key = state.zhash
        if key in self.trans_table:
            if depth >= self.trans_table[key][0]:
                self.trans_table[key] = (depth, flag, value, move)
            self.trans_table.move_to_end(key)
        else:
            if len(self.trans_table) >= self.max_trans_size:
                self.trans_table.popitem(last=False)
            self.trans_table[key] = (depth, flag, value, move)

    def _get_opening_move(self, state):
        hist = state.history
        if not hist or hist[0] != (7, 7):
            self._chosen_opening = None; return None
        if state.player == 1:
            if len(hist) == 1: return None
            if len(hist) == 2:
                w2 = hist[1]
                if w2 in self.OPENING_BOOK:
                    variant = random.choice(self.OPENING_BOOK[w2])
                    self._chosen_opening = variant
                    b3 = variant["b3"]
                    if b3 in state.candidates: return b3
                self._chosen_opening = None; return None
            self._chosen_opening = None; return None
        else:
            if len(hist) == 1:
                w2_candidates = list(self.OPENING_BOOK.keys())
                w2 = random.choice(w2_candidates)
                variant = random.choice(self.OPENING_BOOK[w2])
                self._chosen_opening = variant
                return w2
            if len(hist) == 3:
                if self._chosen_opening is None: return None
                if hist[2] != self._chosen_opening["b3"]:
                    self._chosen_opening = None; return None
                w4_list = self._chosen_opening["w4"]
                available = [m for m in w4_list if m in state.candidates]
                if available:
                    chosen = random.choice(available)
                    self._chosen_opening = None
                    return chosen
                self._chosen_opening = None; return None
            self._chosen_opening = None; return None

    def _reaction(self, state):
        opening_move = self._get_opening_move(state)
        if opening_move is not None and opening_move in state.candidates:
            return opening_move
        opp = 3 - state.player
        opp_fives = self._find_five_moves(state, opp)
        if opp_fives:
            my_fives = self._find_five_moves(state, state.player)
            if my_fives: return my_fives[0]
            return opp_fives[0]
        quick = self._sort_candidates(state)
        if len(quick) == 1: return quick[0]

        # ✅ 修复：在 VCT 搜索前清零，让 max_nodes 限制生效
        self.node_count = 0
        self.timeout = False

        if self.vct_depth > 0:
            vct_move = self._vct_search(state, self.vct_depth)
            if vct_move is not None: return vct_move

        root_scores = {}
        for d in range(1, self.depth + 1):
            alpha = -float('inf')
            root_scores.clear()
            for act in quick:
                oh, db, dw, add = self.apply_move_full(state, act)
                val, _ = self._negamax(state, d - 1, -float('inf'), -alpha)
                val = -val
                self.undo_move_full(state, act, oh, db, dw, add)
                root_scores[act] = val
                if val > alpha: alpha = val
                if self.timeout: break
            if self.timeout: break
        if not root_scores:
            return quick[0] if quick else next(iter(state.candidates))
        best_val = max(root_scores.values())
        threshold = max(200, int(abs(best_val) * 0.03))
        near_best = [act for act, val in root_scores.items() if val >= best_val - threshold]
        return random.choice(near_best)

    def _negamax(self, state, depth, alpha, beta):
        self.node_count += 1
        if self.node_count > self.max_nodes:
            self.timeout = True
            return self._evaluate_state(state), None
        if state.history:
            winner = self._is_terminal(state, state.history[-1])
            if winner is not None:
                if winner == 0: return 0, None
                sign = 1 if winner == state.player else -1
                return sign * self.SCORE_FIVE * (depth + 1), None
        tt_flag, tt_val, tt_move = self._lookup_trans(state, depth)
        if tt_flag is not None:
            if tt_flag == 0: return tt_val, tt_move
            elif tt_flag == 1:
                if tt_val >= beta: return tt_val, tt_move
                alpha = max(alpha, tt_val)
            elif tt_flag == 2:
                if tt_val <= alpha: return tt_val, tt_move
                beta = min(beta, tt_val)
        if depth == 0:
            q_val = self._quiescence(state, alpha, beta, self.quiescence_depth) if self.use_quiescence else self._evaluate_state(state)
            return q_val, None
        trans_best = tt_move
        actions = self._sort_candidates(state, trans_best)
        if not actions: return 0, None
        best_move, best_value, orig_alpha = None, -float('inf'), alpha
        for act in actions:
            oh, db, dw, add = self.apply_move_full(state, act)
            val, _ = self._negamax(state, depth - 1, -beta, -alpha)
            val = -val
            self.undo_move_full(state, act, oh, db, dw, add)
            if self.timeout: return val, act
            if val > best_value:
                best_value = val; best_move = act
            alpha = max(alpha, val)
            if alpha >= beta: break
        flag = 2 if best_value <= orig_alpha else (1 if best_value >= beta else 0)
        self._store_trans(state, depth, flag, best_value, best_move)
        return best_value, best_move
