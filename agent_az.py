# agent_az.py
import torch
from typing import Tuple, Optional
from agent_base import Agent as BaseAgent
from gamerules import GameState
from network import ActorCriticNet
from mcts import MCTS, create_local_eval_fn

# ═══════════════════════════════════════════════════════════════
#  AZAgent: AlphaZero 神经网络 Agent
# ═══════════════════════════════════════════════════════════════

class AZAgent:
    """AlphaZero 神经网络 Agent，使用 MCTS + 神经网络进行决策

    支持树复用：在连续走子间复用 MCTS 搜索树，大幅减少重复计算。
    参考:
      - az_train._arena_phase: create_local_eval_fn 创建本地评估函数
      - pretrain_vs_agent.worker_loop_vs_agent: MCTS 树两步复用模式
    """

    def __init__(
        self,
        model_path: str,
        num_sims: int = 400,
        c_puct: float = 2.5,
        temperature: float = 0.0,
        dirichlet_alpha: float = 0.2,
        dirichlet_epsilon: float = 0.0,
        candidate_radius: int = 3,
        advantage_clip: float = 1.0,
        name: str = "AlphaZero",
        device: str = "auto",
    ):
        self.name = name
        self.temperature = temperature

        # ── 确定设备 ──
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── 加载模型（参考 inference_server 的加载逻辑，自动推断架构） ──
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)

        channels = state_dict['stem_conv.weight'].shape[0]
        res_block_indices = [
            int(k.split('.')[1]) for k in state_dict if k.startswith('res_blocks.')
        ]
        num_blocks = max(res_block_indices) + 1 if res_block_indices else 4

        self.model = ActorCriticNet(
            num_res_blocks=num_blocks, channels=channels
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        # ── 创建 MCTS（参考 az_train._arena_phase 的做法） ──
        eval_fn = create_local_eval_fn(self.model, self.device)
        self.mcts = MCTS(
            eval_fn=eval_fn,
            c_puct=c_puct,
            num_simulations=num_sims,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            candidate_radius=candidate_radius,
            advantage_clip=advantage_clip,
        )

        self._my_last_action = None

    def new_game(self):
        """新一局开始时调用，重置搜索树和记录"""
        self.mcts.root = None
        self._my_last_action = None

    def get_move(self, state):
        """选择落子，支持 MCTS 树复用

        树复用逻辑（参考 pretrain_vs_agent.worker_loop_vs_agent）：
          1. 先推进过自己上一步的子节点（手动推进）
          2. 再通过 search(last_action=对手上一步) 推进过对手的子节点（MCTS 内部处理）
          3. 在复用后的子树上继续搜索，避免每步从零开始
        """
        # 步骤1: 推进过自己上一步
        #   搜索结束后，root 的 children 包含自己的候选动作
        #   推进到实际选择的那步，其 children 就是对手的候选响应
        if self._my_last_action is not None and self.mcts.root is not None:
            if self._my_last_action in self.mcts.root.children:
                child = self.mcts.root.children[self._my_last_action]
                child.parent = None  # 切断反向传播链接，防止内存泄漏
                self.mcts.root = child
            else:
                self.mcts.root = None

        # 步骤2: search 内部通过 last_action 推进过对手上一步，并在复用子树上搜索
        #   state.last_move 就是对手的落子，传给 search 实现自动树复用
        _, action, _ = self.mcts.search(
            state, temperature=self.temperature, last_action=state.last_move
        )

        self._my_last_action = action
        return action