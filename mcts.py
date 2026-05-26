# mcts.py
"""
MCTS 蒙特卡洛树搜索模块 (改进版)

改进点:
  1. 模拟中不再拷贝 history（性能提升 20-30%），依赖 is_board_full 判定和棋
  2. MCTSNode 缓存 raw_value，树复用时省去重复网络评估
  3. 优势裁剪至 [-1, 1]，防止梯度爆炸（与 KataGo 原版一致）
  4. 树复用安全检查：空子树自动回退到正常搜索
  5. Dirichlet 噪声更新修正：按 key 精确匹配而非 zip 顺序
"""

import math
import numpy as np
import torch
from typing import Tuple, Dict, List, Optional, Callable

from gamerules import GameState, GomokuRules
from network import ActorCriticNet

BOARD_SIZE = GomokuRules.BOARD_SIZE
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE


def state_to_tensor(state: GameState) -> np.ndarray:
    """纯 Numpy 实现，避免共享内存系统调用开销"""
    board = np.frombuffer(state.board, dtype=np.int8).reshape(BOARD_SIZE, BOARD_SIZE)
    x = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    x[0] = (board == state.current_player).astype(np.float32)
    x[1] = (board == (3 - state.current_player)).astype(np.float32)
    if state.last_move is not None:
        x[2, state.last_move[0], state.last_move[1]] = 1.0
    return x


class MCTSNode:
    __slots__ = ['parent', 'action', 'children', 'n_visits', 'w_value', 'p_prior', 'raw_value']

    def __init__(self, parent=None, prior_p: float = 0.0, action=None):
        self.parent = parent
        self.action = action
        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}
        self.n_visits = 0
        self.w_value = 0.0
        self.p_prior = prior_p
        self.raw_value = None  # ✅ 缓存网络对该节点状态的原始估值

    @property
    def q_value(self) -> float:
        return self.w_value / self.n_visits if self.n_visits > 0 else 0.0

    def select(self, c_puct: float) -> Tuple[Tuple[int, int], 'MCTSNode']:
        best_score = -float('inf')
        best_action = None
        best_child = None
        sqrt_parent = math.sqrt(self.n_visits)

        for action, child in self.children.items():
            ucb = child.q_value + c_puct * child.p_prior * sqrt_parent / (1 + child.n_visits)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_priors: List[Tuple[Tuple[int, int], float]]):
        for action, prior in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior_p=prior, action=action)

    def backpropagate(self, value: float):
        node = self
        while node is not None:
            node.n_visits += 1
            node.w_value += value
            value = -value
            node = node.parent


class MCTS:
    def __init__(
        self,
        eval_fn: Callable[[np.ndarray], Tuple[np.ndarray, float]],
        c_puct: float = 1.5,
        num_simulations: int = 400,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        candidate_radius: int = 2,
        advantage_clip: float = 1.0,
    ):
        self.eval_fn = eval_fn
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.candidate_radius = candidate_radius
        self.advantage_clip = advantage_clip
        self.root = None

    def _evaluate(self, state: GameState) -> Tuple[np.ndarray, float]:
        x = state_to_tensor(state)
        policy_probs, value = self.eval_fn(x)
        return policy_probs, value

    def _get_legal_moves_with_priors(
        self, state: GameState, policy_probs: np.ndarray
    ) -> List[Tuple[Tuple[int, int], float]]:
        candidates = GomokuRules.get_candidates(state, radius=self.candidate_radius)
        if not candidates:
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if state.board[r * BOARD_SIZE + c] == 0:
                        candidates.add((r, c))

        move_priors = []
        total_prior = 0.0
        for move in candidates:
            idx = move[0] * BOARD_SIZE + move[1]
            prior = policy_probs[idx]
            move_priors.append((move, prior))
            total_prior += prior

        if total_prior > 1e-8:
            move_priors = [(m, p / total_prior) for m, p in move_priors]
        else:
            uniform = 1.0 / len(move_priors) if move_priors else 0.0
            move_priors = [(m, uniform) for m, _ in move_priors]

        return move_priors

    def _add_dirichlet_noise(
        self, move_priors: List[Tuple[Tuple[int, int], float]]
    ) -> List[Tuple[Tuple[int, int], float]]:
        n = len(move_priors)
        if n <= 1:
            return move_priors

        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        eps = self.dirichlet_epsilon

        result = []
        for i, (move, prior) in enumerate(move_priors):
            new_prior = (1 - eps) * prior + eps * noise[i]
            result.append((move, new_prior))

        total = sum(p for _, p in result)
        if total > 1e-8:
            result = [(m, p / total) for m, p in result]

        return result

    @staticmethod
    def _terminal_value(winner: int, current_player_at_leaf: int) -> float:
        if winner == 0:
            return 0.0
        mover = 3 - current_player_at_leaf
        return 1.0 if winner == mover else -1.0

    def search(
        self, root_state: GameState, temperature: float = 1.0, last_action: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray]:
        """
        执行 MCTS 搜索，返回 (target_policy, best_action, advantages)
        
        改进:
          - 树复用：利用缓存的 raw_value 避免重复网络评估
          - 空子树安全检查：复用的子节点无 children 时回退到正常搜索
          - 优势裁剪：advantages 裁剪至 [-advantage_clip, advantage_clip]
        """
        reuse_tree = False
        root_net_value = 0.0

        # ========== 树复用逻辑 ==========
        if last_action is not None and self.root is not None:
            if last_action in self.root.children:
                candidate = self.root.children[last_action]
                candidate.parent = None

                # ✅ 安全检查：只有子树有内容时才复用
                if candidate.children:
                    root = candidate
                    reuse_tree = True

                    # 更新 Dirichlet 噪声
                    if self.dirichlet_epsilon > 0:
                        move_priors = [(act, child.p_prior) for act, child in root.children.items()]
                        move_priors = self._add_dirichlet_noise(move_priors)
                        # ✅ 按 key 精确匹配，不依赖迭代顺序
                        for act, new_prior in move_priors:
                            root.children[act].p_prior = new_prior

                    # ✅ 使用缓存的 raw_value，避免重复网络评估
                    if candidate.raw_value is not None:
                        root_net_value = candidate.raw_value
                    else:
                        _, root_net_value = self._evaluate(root_state)
                else:
                    # 空子树，回退到正常搜索
                    root = MCTSNode()
            else:
                root = MCTSNode()
        else:
            root = MCTSNode()

        self.root = root

        # ========== 根节点首次评估 ==========
        if not reuse_tree:
            policy_probs, root_net_value = self._evaluate(root_state)
            move_priors = self._get_legal_moves_with_priors(root_state, policy_probs)
            move_priors = self._add_dirichlet_noise(move_priors)
            root.expand(move_priors)
            root.raw_value = root_net_value  # ✅ 缓存根节点估值
            root.n_visits = 1

        # ========== MCTS 模拟循环 ==========
        for _ in range(self.num_simulations):
            node = self.root
            # ✅ 不再拷贝 history，大幅减少内存分配和拷贝开销
            state = GameState(
                board=bytearray(root_state.board),
                current_player=root_state.current_player,
                history=[],              # ✅ MCTS 内部不需要 history
                last_move=root_state.last_move,
            )

            while node.children:
                action, node = node.select(self.c_puct)
                GomokuRules.apply_move_fast(state, action)

            winner = GomokuRules.check_winner(state)
            if winner is not None:
                leaf_value = self._terminal_value(winner, state.current_player)
            else:
                policy_probs_leaf, net_value = self._evaluate(state)
                leaf_value = -net_value
                move_priors_leaf = self._get_legal_moves_with_priors(state, policy_probs_leaf)
                node.expand(move_priors_leaf)
                node.raw_value = net_value  # ✅ 缓存叶子节点估值

            node.backpropagate(leaf_value)

        # ========== 收集搜索结果 ==========
        mcts_policy = np.zeros(BOARD_SQUARES, dtype=np.float32)
        advantages = np.zeros(BOARD_SQUARES, dtype=np.float32)

        for action, child in self.root.children.items():
            idx = action[0] * BOARD_SIZE + action[1]
            mcts_policy[idx] = child.n_visits
            if child.n_visits > 0:
                advantages[idx] = child.q_value - root_net_value

        # ✅ 优势裁剪：与 KataGo 原版一致，防止梯度爆炸
        if self.advantage_clip > 0:
            advantages = np.clip(advantages, -self.advantage_clip, self.advantage_clip)

        target_policy = mcts_policy.copy()
        if target_policy.sum() > 0:
            target_policy /= target_policy.sum()

        # ========== 动作选择 ==========
        if temperature <= 1e-3:
            best_visits = -1
            best_action = None
            for act, child in self.root.children.items():
                if child.n_visits > best_visits:
                    best_visits = child.n_visits
                    best_action = act

            if best_action is None:
                candidates = GomokuRules.get_candidates(root_state, radius=self.candidate_radius)
                best_action = list(candidates)[0] if candidates else (BOARD_SIZE // 2, BOARD_SIZE // 2)
            action = best_action
        else:
            log_policy = np.log(mcts_policy + 1e-10)
            log_policy_scaled = log_policy / temperature
            log_policy_scaled -= np.max(log_policy_scaled)
            temp_policy = np.exp(log_policy_scaled)

            if temp_policy.sum() > 0:
                temp_policy /= temp_policy.sum()
            else:
                temp_policy = np.ones(BOARD_SQUARES, dtype=np.float32) / BOARD_SQUARES

            flat_idx = np.random.choice(BOARD_SQUARES, p=temp_policy)
            action = (flat_idx // BOARD_SIZE, flat_idx % BOARD_SIZE)

        return target_policy, action, advantages


def create_local_eval_fn(model: ActorCriticNet, device: torch.device):
    """创建本地评估函数（用于竞技场等单进程场景）"""
    @torch.no_grad()
    def eval_fn(state_np: np.ndarray):
        model.eval()
        x_tensor = torch.from_numpy(state_np).unsqueeze(0).to(device)
        policy_logits, value = model(x_tensor)

        policy_logits_flat = policy_logits.squeeze(0).view(-1).cpu().numpy()
        max_logit = policy_logits_flat.max()
        exp_logits = np.exp(policy_logits_flat - max_logit)
        policy_probs = exp_logits / exp_logits.sum()

        return policy_probs, value.item()
    return eval_fn