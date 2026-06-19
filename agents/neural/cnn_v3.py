# agents/neural/cnn_v3.py
"""
CNN v9.3: 预激活残差块 + 交叉注意力价值头 + 纯净卷积策略头

改动 (相对于 v9.2):
  - 5 个残差块，通道数降至 64
  - 价值头改为：展平序列 + cls token 交叉注意力(4头) + MLP → tanh
  - 价值头交叉注意力带 key_padding_mask 屏蔽已落子位置
  - 策略头掩码使用 -1e4 (FP16 安全)
  - value_mlp 最后一层无 bias
  - 约 41 万参数

预激活范式 (He et al. 2016):
  Stem:     Conv (无BN/ReLU)
  ResBlock: BN → ReLU → Conv → BN → ReLU → Conv → (+x)
  尾部:     BN → ReLU → 送入头部

输入: (B, 3, 15, 15), 输出: policy_logits (B, 225), value (B,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.neural.registry import register


class ResBlock(nn.Module):
    """预激活残差块: BN → ReLU → Conv → BN → ReLU → Conv → (+x)"""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + x


@register(
    arch_type='cnn_v3',
    param_names=['num_res_blocks', 'channels', 'board_size'],
    defaults={'num_res_blocks': 5, 'channels': 64, 'board_size': 15},
)
class ActorCriticNet_v3(nn.Module):
    """
    Actor-Critic 双头神经网络 (v9.3)

    结构:
      Stem → 5× 预激活 ResBlock(64) → BN+ReLU → 分叉
                                                ├→ 策略头(Conv×3) → logits (225维)
                                                └→ 价值头(Cross-Attn + MLP) → tanh (-1~1)

    输入: (B, 3, 15, 15) - 3通道 (己方/对方/上一步标记)
    输出: policy_logits (B, 225), value (B,)

    Args:
        num_res_blocks: 残差块数量 (默认5)
        channels: 特征通道数 (默认64)
        board_size: 棋盘大小 (默认15)
    """

    def __init__(self, num_res_blocks: int = 5, channels: int = 64, board_size: int = 15):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.board_squares = board_size * board_size

        # ── Stem ──
        self.stem_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)

        # ── 残差塔 ──
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])

        # ── 尾部归一化 ──
        self.final_bn = nn.BatchNorm2d(channels)

        # ── 策略头 (不变结构，通道适配) ──
        self.policy_conv1 = nn.Conv2d(channels, 64, kernel_size=1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(64)
        self.policy_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.policy_bn2 = nn.BatchNorm2d(32)
        self.policy_conv3 = nn.Conv2d(32, 1, kernel_size=1, bias=False)

        # ── 价值头 (交叉注意力 + MLP) ──
        # 可学习的 cls token，作为 query
        self.cls_token = nn.Parameter(torch.randn(1, 1, channels))

        # 多头交叉注意力（num_heads=4，与 Transformer 对齐）
        self.value_cross_attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=4, batch_first=True
        )
        self.value_ln = nn.LayerNorm(channels)

        # MLP 读出头（最后一层无 bias）
        self.value_mlp = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1, bias=False),
        )

        # 安全掩码值（FP16 可表示，足够小但非 -inf）
        self.mask_val = -1e4

        self._init_weights()

    def _init_weights(self):
        """初始化卷积、BN、线性层和 cls_token"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # cls_token 用 trunc_normal 风格初始化
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, return_value_only: bool = False):
        """
        前向传播。

        Args:
            x: (B, 3, 15, 15) 输入状态
            return_value_only: 只返回价值（跳过策略头计算）

        Returns:
            (policy_logits, value) 或 (None, value)
        """
        B, C, H, W = x.shape

        # ── 占用掩码（用于策略头 + 价值头）──
        occupied_mask = (x[:, 0, :, :] + x[:, 1, :, :]).view(B, -1) > 0  # (B, 225)

        # ── Stem ──
        out = self.stem_conv(x)  # (B, C, 15, 15)

        # ── 残差塔 ──
        for block in self.res_blocks:
            out = block(out)  # 保持 (B, C, 15, 15)

        # ── 尾部激活 ──
        out = F.relu(self.final_bn(out))  # (B, C, 15, 15)

        # ═══════════ 价值头 (交叉注意力 + key_padding_mask) ═══════════
        # 展平为序列，作为 key/value
        kv = out.flatten(2).transpose(1, 2)  # (B, 225, C)
        # cls token 扩展 batch 维度，作为 query
        query = self.cls_token.expand(B, -1, -1)  # (B, 1, C)

        # 交叉注意力：query 关注 kv，mask 已落子位置切断梯度污染
        attn_out, _ = self.value_cross_attn(
            query=query,
            key=kv,
            value=kv,
            key_padding_mask=occupied_mask,
        )  # (B, 1, C)

        # 残差连接 + LayerNorm
        attn_out = attn_out + query
        attn_out = self.value_ln(attn_out)  # (B, 1, C)

        # MLP 输出标量，再 tanh
        value = self.value_mlp(attn_out)  # (B, 1, 1)
        value = torch.tanh(value).squeeze(-1).squeeze(-1)  # (B,)

        if return_value_only:
            return None, value

        # ═══════════ 策略头 ═══════════
        p = F.relu(self.policy_bn1(self.policy_conv1(out)))
        p = F.relu(self.policy_bn2(self.policy_conv2(p)))
        p = self.policy_conv3(p)  # (B, 1, H, W)
        policy_logits = p.view(B, -1)  # (B, 225)

        # 掩码已落子位置（FP16 安全）
        policy_logits = policy_logits.masked_fill(occupied_mask, self.mask_val)

        return policy_logits, value

    @property
    def arch_type(self) -> str:
        return "cnn_v3"

    def get_config(self) -> dict:
        return {
            'arch_type': 'cnn_v3',
            'num_res_blocks': len(self.res_blocks),
            'channels': self.channels,
            'board_size': self.board_size,
        }