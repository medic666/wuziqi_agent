# agents/neural/az_agent.py
"""
AlphaZero 神经网络智能体 (AZAgent)

支持两种决策模式:
  - mode="mcts":    MCTS 搜索 + 神经网络评估 (默认)
  - mode="nucleus": 核采样：直接使用策略头输出进行 top-p 采样，无 MCTS

关键设计:
  - 树复用: 手动推进自己的走子 + search() 内部处理对手走子 (仅 MCTS 模式)
  - 模型加载: 自动推断网络架构（CNN vs Transformer），支持显式指定
  - 设备管理: 支持 auto/cuda/cpu 指定
  - 架构无关: 通过 network_cls 参数或 checkpoint 中的 arch_type 字段
              自动选择网络类，CNN 和 Transformer 可复用同一 AZAgent

搬自: agent_az.py
"""

import torch
from typing import Tuple, Optional, Type
from core.gamerules import GameState
from agents.neural.registry import build_model_from_checkpoint
from search.mcts import MCTS, create_local_eval_fn, state_to_tensor
from search.sampling import nucleus_sample_action


def _infer_network_from_checkpoint(ckpt: dict, device: torch.device):
    """
    从 checkpoint 自动推断网络架构并实例化。

    委托给 registry.build_model_from_checkpoint，该函数是单一真理源。

    Args:
        ckpt: 加载的 checkpoint 字典
        device: 目标设备

    Returns:
        实例化的网络模型 (已加载权重，eval模式)
    """
    model, _, _ = build_model_from_checkpoint(ckpt, device=device)
    return model


class AZAgent:
    """
    AlphaZero 神经网络智能体（架构无关）。

    支持两种决策模式:
      - mode="mcts":    MCTS 搜索 + 神经网络评估 (默认)，支持树复用
      - mode="nucleus": 核采样：直接使用策略头输出进行 top-p 采样，无 MCTS

    通过 new_game() 方法在新一局开始时重置搜索树（仅 MCTS 模式）。

    Args:
        model_path: 模型权重文件路径 (.pt)
        mode: 决策模式 "mcts" | "nucleus"
        num_sims: MCTS 模拟次数 (仅 MCTS 模式)
        c_puct: PUCT 探索参数 (仅 MCTS 模式)
        temperature: 温度参数 (0=确定性; MCTS 模式控制 visit softmax, nucleus 模式控制概率缩放)
        dirichlet_alpha: Dirichlet 噪声 alpha (仅 MCTS 模式)
        dirichlet_epsilon: Dirichlet 噪声混合比例 (仅 MCTS 模式)
        candidate_radius: 候选着法搜索半径
        advantage_clip: 优势值裁剪范围 (仅 MCTS 模式)
        nucleus_p: 核采样 top-p 阈值 (仅 nucleus 模式)
        name: 智能体名称（用于竞技场显示）
        device: 计算设备 ("auto"/"cuda"/"cpu")
        network_cls: 显式指定网络类（None=自动从checkpoint推断）
    """

    def __init__(
        self,
        model_path: str,
        mode: str = "mcts",
        num_sims: int = 400,
        c_puct: float = 2.5,
        temperature: float = 0.0,
        dirichlet_alpha: float = 0.2,
        dirichlet_epsilon: float = 0.0,
        candidate_radius: int = 3,
        advantage_clip: float = 1.0,
        nucleus_p: float = 0.6,
        name: str = "AlphaZero",
        device: str = "auto",
        network_cls: Optional[Type] = None,
    ):
        self.name = name
        self.mode = mode
        self.temperature = temperature
        self.nucleus_p = nucleus_p
        self.candidate_radius = candidate_radius

        # ── 确定计算设备 ──
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── 加载模型 ──
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        if network_cls is not None:
            state_dict = ckpt.get('model_state_dict', ckpt)
            config = ckpt.get('model_config', {})
            self.model = network_cls(**config).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            self.model = _infer_network_from_checkpoint(ckpt, self.device)

        # CUDA 优化
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        # ── 共享推理接口 ──
        self._eval_fn = create_local_eval_fn(self.model, self.device)

        # ── 按模式创建决策组件 ──
        if mode == "mcts":
            self.mcts = MCTS(
                eval_fn=self._eval_fn,
                c_puct=c_puct,
                num_simulations=num_sims,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
                candidate_radius=candidate_radius,
                advantage_clip=advantage_clip,
            )
        else:
            self.mcts = None

        self._my_last_action = None

    def new_game(self):
        """
        新一局开始时调用，重置搜索树和记录。
        nucleus 模式下无操作。
        """
        self._my_last_action = None
        if self.mcts is not None:
            self.mcts.root = None

    def get_move(self, state: GameState) -> Tuple[int, int]:
        """
        选择落子。按 mode 分发到 MCTS 搜索或核采样。

        MCTS 模式: 树复用 → 搜索 → 返回最佳着法
        nucleus 模式: 网络推理 → 核采样 → 返回着法
        """
        if self.mode == "nucleus":
            return self._get_move_nucleus(state)
        return self._get_move_mcts(state)

    def _get_move_mcts(self, state: GameState) -> Tuple[int, int]:
        """MCTS 搜索 + 树复用"""
        if self._my_last_action is not None and self.mcts.root is not None:
            if self._my_last_action in self.mcts.root.children:
                child = self.mcts.root.children[self._my_last_action]
                child.parent = None
                self.mcts.root = child
            else:
                self.mcts.root = None

        _, action, _ = self.mcts.search(
            state, temperature=self.temperature, last_action=state.last_move
        )

        self._my_last_action = action
        return action

    def _get_move_nucleus(self, state: GameState) -> Tuple[int, int]:
        """核采样：直接策略头输出 + top-p 采样"""
        state_tensor = state_to_tensor(state)
        policy_probs, _ = self._eval_fn(state_tensor)
        return nucleus_sample_action(
            state, policy_probs, self.nucleus_p,
            self.temperature, self.candidate_radius
        )

    def get_hint_move(self, state: GameState) -> Tuple[int, int]:
        """
        返回推荐的着法，不修改智能体内部状态（用于 UI 提示等场景）。

        MCTS 模式: 临时运行 MCTS 搜索 (不干扰游戏树)
        nucleus 模式: 策略头 argmax
        """
        if self.mode == "nucleus":
            state_tensor = state_to_tensor(state)
            policy_probs, _ = self._eval_fn(state_tensor)
            return nucleus_sample_action(
                state, policy_probs, self.nucleus_p, 0.0, self.candidate_radius
            )
        else:
            saved_root = self.mcts.root
            _, action, _ = self.mcts.search(state, temperature=0.0, last_action=None)
            self.mcts.root = saved_root
            return action