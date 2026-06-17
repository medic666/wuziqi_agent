# agents/neural/az_agent.py
"""
AlphaZero 神经网络智能体 (AZAgent)

使用 MCTS + 神经网络进行决策。支持树复用：在连续走子间复用
MCTS 搜索树，大幅减少重复计算。

关键设计:
  - 树复用: 手动推进自己的走子 + search() 内部处理对手走子
  - 模型加载: 自动推断网络架构（通道数、残差块数）
  - 设备管理: 支持 auto/cuda/cpu 指定

搬自: agent_az.py
"""

import torch
from typing import Tuple, Optional
from core.gamerules import GameState
from agents.neural.network import ActorCriticNet
from search.mcts import MCTS, create_local_eval_fn


class AZAgent:
    """
    AlphaZero 神经网络智能体。
    
    使用 MCTS + 神经网络进行决策。
    通过 new_game() 方法在新一局开始时重置搜索树。
    
    典型用法:
        agent = AZAgent(
            model_path="checkpoints/az_train/best_model.pt",
            num_sims=400,
            temperature=0.0,       # 确定性走子
            dirichlet_epsilon=0.0,  # 竞技场不加噪声
            name="AlphaZero",
        )
        move = agent.get_move(state)
    
    Args:
        model_path: 模型权重文件路径 (.pt)
        num_sims: MCTS 模拟次数
        c_puct: PUCT 探索参数
        temperature: 温度参数 (0=确定性选最大访问数)
        dirichlet_alpha: Dirichlet 噪声 alpha
        dirichlet_epsilon: Dirichlet 噪声混合比例 (0=不加噪声)
        candidate_radius: 候选着法搜索半径
        advantage_clip: 优势值裁剪范围 [-clip, clip]
        name: 智能体名称（用于竞技场显示）
        device: 计算设备 ("auto"/"cuda"/"cpu")
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

        # ── 确定计算设备 ──
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── 加载模型（自动推断架构参数） ──
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)

        # 从权重推断通道数
        channels = state_dict['stem_conv.weight'].shape[0]
        # 从权重推断残差块数
        res_block_indices = [
            int(k.split('.')[1]) for k in state_dict if k.startswith('res_blocks.')
        ]
        num_blocks = max(res_block_indices) + 1 if res_block_indices else 4

        self.model = ActorCriticNet(
            num_res_blocks=num_blocks, channels=channels
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # CUDA 优化
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        # ── 创建 MCTS ──
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

        # 树复用状态
        self._my_last_action = None

    def new_game(self):
        """
        新一局开始时调用，重置搜索树和记录。
        
        遵循 Agent 生命周期管理约定。
        """
        self.mcts.root = None
        self._my_last_action = None

    def get_move(self, state: GameState) -> Tuple[int, int]:
        """
        选择落子，支持 MCTS 树复用。
        
        树复用逻辑:
          1. 先推进过自己上一步的子节点（手动推进）
          2. 再通过 search(last_action=对手上一步) 推进过对手的子节点
          3. 在复用后的子树上继续搜索，避免每步从零开始
        
        Args:
            state: 当前游戏状态
            
        Returns:
            (行, 列) 落子坐标
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