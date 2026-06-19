# agents/neural/registry.py
"""
网络架构注册表 — 单一真理源 (Single Source of Truth)

新增架构只需:
  1. 创建 agents/neural/xxx.py，用 @register 装饰器注册
  2. 在 __init__.py 中 import
  3. 命令行 python az_train.py --arch xxx 即可

设计原则:
  - 注册表是唯一的架构定义源，az_agent / inference_server /
    训练入口脚本全部引用此注册表
  - 旧 checkpoint 通过别名 + 权重推断自动兼容
"""

from typing import Dict, Tuple, Type, List, Optional
import torch.nn as nn


_NetworkEntry = Tuple[Type[nn.Module], List[str], dict]
NETWORK_REGISTRY: Dict[str, _NetworkEntry] = {}

# ═══════════════════════════════════════════════════════════════
#  别名映射 — 兼容旧 checkpoint 的 arch_type 字段
# ═══════════════════════════════════════════════════════════════
ARCH_ALIASES: Dict[str, str] = {
    'cnn':          'cnn_v2',      # 旧命名 (v9.2 及之前)
    'cnn_v2':       'cnn_v2',
    'cnn_v3':       'cnn_v3',
    'transformer':  'transformer',
}


def register(arch_type: str, param_names: List[str], defaults: dict):
    """
    装饰器: 将网络类注册到全局注册表。

    用法:
        @register('my_arch', ['channels', 'board_size'], {'channels': 64, 'board_size': 15})
        class MyNet(nn.Module):
            ...
    """
    def decorator(cls):
        NETWORK_REGISTRY[arch_type] = (cls, param_names, defaults)
        return cls
    return decorator


def resolve_arch(arch_type: str) -> str:
    """解析别名到正规键名，不存在的返回原值（由调用方报错）"""
    return ARCH_ALIASES.get(arch_type, arch_type)


def get_network_class(arch_type: str) -> Type[nn.Module]:
    """根据 arch_type (或别名) 获取网络类"""
    resolved = resolve_arch(arch_type)
    if resolved not in NETWORK_REGISTRY:
        available = list(ARCH_ALIASES.keys())
        raise ValueError(
            f"未知架构 '{arch_type}'，可用: {available}\n"
            f"  注: 旧checkpoint的 'cnn' 自动映射为 'cnn_v2'"
        )
    return NETWORK_REGISTRY[resolved][0]


def get_param_names(arch_type: str) -> List[str]:
    """获取某架构的构造参数名列表"""
    resolved = resolve_arch(arch_type)
    return NETWORK_REGISTRY[resolved][1]


def get_defaults(arch_type: str) -> dict:
    """获取某架构的默认参数 (返回副本，防止外部修改)"""
    resolved = resolve_arch(arch_type)
    return NETWORK_REGISTRY[resolved][2].copy()


def list_architectures() -> List[str]:
    """列出所有可用架构名称 (含别名)"""
    return list(ARCH_ALIASES.keys())


def infer_arch_from_state_dict(state_dict: dict) -> str:
    """
    从权重键名推断架构类型 (兼容没有 arch_type 字段的旧 checkpoint)。

    推断规则:
      - embed.weight / blocks.* → transformer
      - stem_conv.* / res_blocks.* → cnn_v2 (无法从权重区分 v2/v3, 回退 v2)

    Returns:
        arch_type 字符串 (已解析别名后的正规键名)
    """
    if not state_dict:
        return 'cnn_v2'

    any_key = next(iter(state_dict))

    if any_key == 'embed.weight' or any_key.startswith('blocks.'):
        return 'transformer'
    elif any_key.startswith('stem_conv.') or any_key.startswith('res_blocks.'):
        # CNN v2 和 v3 权重结构相同，无法区分，保守回退 v2
        # 用户如需 v3 应显式传入 arch_type
        return 'cnn_v2'

    # 未知结构回退
    return 'cnn_v2'


def build_model_from_checkpoint(ckpt: dict, device=None):
    """
    ★ 统一入口：从 checkpoint 字典构造模型并加载权重。

    消除 az_agent / inference_server / test.py / pre_train.py / pretrain_vs_agent.py
    中重复的"推断架构 + 从权重推断参数 + 构造 + load_state_dict"逻辑。

    Args:
        ckpt: torch.load() 返回的 checkpoint 字典
        device: 目标设备 (None = CPU)

    Returns:
        (model, arch_type, kwargs) — 已 eval() 的模型, 解析后的架构名, 构造参数
    """
    import torch

    state_dict = ckpt.get('model_state_dict', ckpt)
    config = ckpt.get('model_config', {})

    # 1. 推断架构
    arch_type = config.get('arch_type', None)
    if arch_type is not None:
        arch_type = resolve_arch(arch_type)
    else:
        arch_type = infer_arch_from_state_dict(state_dict)

    # 2. 构建参数：默认值 → config 覆盖 → 权重推断
    kwargs = get_defaults(arch_type)
    for k in config:
        if k != 'arch_type' and k in kwargs:
            kwargs[k] = config[k]

    # 从权重推断（唯一实现点，杜绝各处 shape[0]/shape[1] 不一致 bug）
    if arch_type in ('cnn_v2', 'cnn_v3'):
        kwargs['channels'] = state_dict['stem_conv.weight'].shape[0]
        res_idx = [int(k.split('.')[1]) for k in state_dict if k.startswith('res_blocks.')]
        kwargs['num_res_blocks'] = max(res_idx) + 1 if res_idx else kwargs['num_res_blocks']

    if arch_type == 'transformer':
        kwargs['d_model'] = state_dict['embed.weight'].shape[0]
        blk_idx = [int(k.split('.')[1]) for k in state_dict if k.startswith('blocks.')]
        kwargs['num_layers'] = max(blk_idx) + 1 if blk_idx else kwargs['num_layers']

    # 3. 构造 + 加载 + eval
    network_cls = get_network_class(arch_type)
    model = network_cls(**kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    if device is not None:
        model = model.to(device)

    return model, arch_type, kwargs


def build_model_from_config(arch_type: str, arch_params: dict = None, device=None):
    """
    从架构名和可选参数覆盖构造一个随机权重模型。

    Args:
        arch_type: 架构名 (支持别名)
        arch_params: 参数覆盖 (None = 使用注册表默认值)
        device: 目标设备

    Returns:
        model — 已初始化权重的模型 (train 模式)
    """
    import torch
    arch_type = resolve_arch(arch_type)
    kwargs = get_defaults(arch_type)
    if arch_params:
        kwargs.update(arch_params)
    network_cls = get_network_class(arch_type)
    model = network_cls(**kwargs)
    if device is not None:
        model = model.to(device)
    return model
