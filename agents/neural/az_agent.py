# agents/neural/az_agent.py
"""
AlphaZero 神经网络智能体 (AZAgent)

使用 MCTS + 神经网络进行决策。支持树复用：在连续走子间复用
MCTS 搜索树，大幅减少重复计算。

关键设计:
  - 树复用: 手动推进自己的走子 + search() 内部处理对手走子
  - 模型加载: 自动推断网络架构（CNN vs Transformer），支持显式指定
  - 设备管理: 支持 auto/cuda/cpu 指定
  - 架构无关: 通过 network_cls 参数或 checkpoint 中的 arch_type 字段
              自动选择网络类，CNN 和 Transformer 可复用同一 AZAgent

搬自: agent_az.py
"""

import torch
from typing import Tuple, Optional, Type
from core.gamerules import GameState
from agents.neural.network import ActorCriticNet
from agents.neural.transformer_network import GoBangTransformer_v2
from search.mcts import MCTS, create_local_eval_fn

# 架构注册表：arch_type → (网络类, 构造函数参数名映射)
_NETWORK_REGISTRY = {
    'cnn': (ActorCriticNet, ['num_res_blocks', 'channels', 'board_size']),
    'transformer': (GoBangTransformer_v2, ['d_model', 'num_heads', 'num_layers', 'ff_expand', 'dropout', 'board_size']),
}


def _infer_network_from_checkpoint(ckpt: dict, device: torch.device):
    """
    从 checkpoint 自动推断网络架构并实例化。

    根据 checkpoint 中保存的 arch_type 或从权重键名推断架构类型，
    提取对应的超参数，实例化正确的网络类。

    Args:
        ckpt: 加载的 checkpoint 字典
        device: 目标设备

    Returns:
        实例化的网络模型 (已加载权重，eval模式)
    """
    state_dict = ckpt.get('model_state_dict', ckpt)
    config = ckpt.get('model_config', {})

    # 优先从 config 读取 arch_type
    arch_type = config.get('arch_type', None)

    # 如果 config 中没有，从权重键名自动推断
    if arch_type is None:
        any_key = next(iter(state_dict))
        if any_key.startswith('stem_conv.'):
            arch_type = 'cnn'
        elif any_key == 'embed.weight' or any_key.startswith('blocks.'):
            arch_type = 'transformer'
        elif any_key.startswith('res_blocks.'):
            arch_type = 'cnn'
        else:
            arch_type = 'cnn'  # 默认回退

    if arch_type not in _NETWORK_REGISTRY:
        raise ValueError(f"未知架构类型 '{arch_type}'，已知类型: {list(_NETWORK_REGISTRY.keys())}")

    network_cls, param_names = _NETWORK_REGISTRY[arch_type]

    # 构建构造函数参数
    kwargs = {}
    for pname in param_names:
        if pname in config:
            kwargs[pname] = config[pname]

    # CNN 特殊处理：从权重推断通道数和残差块数（兼容旧checkpoint）
    if arch_type == 'cnn':
        if 'channels' not in kwargs:
            kwargs['channels'] = state_dict['stem_conv.weight'].shape[0]
        if 'num_res_blocks' not in kwargs:
            res_block_indices = [
                int(k.split('.')[1]) for k in state_dict if k.startswith('res_blocks.')
            ]
            kwargs['num_res_blocks'] = max(res_block_indices) + 1 if res_block_indices else 4
        if 'board_size' not in kwargs:
            kwargs['board_size'] = 15

    # Transformer 特殊处理：从权重推断参数（兼容未保存config的checkpoint）
    if arch_type == 'transformer':
        if 'd_model' not in kwargs:
            kwargs['d_model'] = state_dict['embed.weight'].shape[1]
        if 'num_layers' not in kwargs:
            block_indices = [
                int(k.split('.')[1]) for k in state_dict if k.startswith('blocks.')
            ]
            kwargs['num_layers'] = max(block_indices) + 1 if block_indices else 5
        if 'num_heads' not in kwargs:
            # 从 q_proj 权重形状推断
            kwargs['num_heads'] = 4  # 默认值
        if 'board_size' not in kwargs:
            kwargs['board_size'] = 15
        # 设置默认值
        kwargs.setdefault('ff_expand', 4)
        kwargs.setdefault('dropout', 0.1)

    # 检查必需参数
    for pname in param_names:
        if pname not in kwargs:
            kwargs[pname] = _get_default_param(arch_type, pname)

    # 实例化模型
    model = network_cls(**kwargs).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def _get_default_param(arch_type: str, param_name: str):
    """获取参数的默认值。"""
    defaults = {
        'cnn': {'num_res_blocks': 4, 'channels': 128, 'board_size': 15},
        'transformer': {'d_model': 64, 'num_heads': 4, 'num_layers': 5, 'ff_expand': 4, 'dropout': 0.1, 'board_size': 15},
    }
    return defaults.get(arch_type, {}).get(param_name, 15)


class AZAgent:
    """
    AlphaZero 神经网络智能体（架构无关）。

    使用 MCTS + 神经网络进行决策。通过 new_game() 方法在新一局开始时重置搜索树。

    支持 CNN (ActorCriticNet) 和 Transformer (GoBangTransformer_v2) 两种架构，
    从 checkpoint 自动推断或通过 network_cls 显式指定。

    典型用法 (CNN):
        agent = AZAgent(
            model_path="checkpoints/az_train/best_model.pt",
            num_sims=400,
            temperature=0.0,
            name="AlphaZero_CNN",
        )

    典型用法 (Transformer):
        agent = AZAgent(
            model_path="checkpoints/transformer_train/best_model.pt",
            num_sims=400,
            temperature=0.0,
            name="AlphaZero_Transformer",
        )

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
        network_cls: 显式指定网络类（None=自动从checkpoint推断）。
                     用于强制指定架构，例如竞技场中CNN vs Transformer对战。
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
        network_cls: Optional[Type] = None,
    ):
        self.name = name
        self.temperature = temperature

        # ── 确定计算设备 ──
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── 加载模型 ──
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        if network_cls is not None:
            # 显式指定网络类：使用 checkpoint 中的 config 构造
            state_dict = ckpt.get('model_state_dict', ckpt)
            config = ckpt.get('model_config', {})
            self.model = network_cls(**config).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            # 自动推断架构
            self.model = _infer_network_from_checkpoint(ckpt, self.device)

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