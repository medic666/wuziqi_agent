# agents/neural/hybrid_v1.py
"""
CNN + Transformer 混合骨干 (Hybrid v1)

架构: 5× 预激活 ResBlock + 1× 主干 Transformer(RoPE+GELU) + 双头

   Stem → 5× ResBlock(64) → ★主干Transformer+RoPE → BN+ReLU → 分叉
                                                            ├→ 策略头(1×1 Conv×3) → logits
                                                            └→ 价值头(Cross-Attn + MLP) → tanh

改动 (相对于 cnn_v3):
  - 主干新增 1 层 Pre-LN Transformer 块 (Self-Attention + RoPE + FFN(GELU))
  - 策略头简化为纯 1×1 Conv 降维 (64→32→16→1)
  - 价值头不变 (CLS token 交叉注意力池化)

预激活范式 (He et al. 2016):
  Stem:     Conv (无BN/ReLU)
  ResBlock: BN → ReLU → Conv → BN → ReLU → Conv → (+x)
  Transformer: Pre-LN → Self-Attn(RoPE) → + → Pre-LN → FFN(GELU) → +
  尾部:     BN → ReLU → 送入头部

输入: (B, 3, 15, 15), 输出: policy_logits (B, 225), value (B,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.neural.registry import register
from agents.neural.rope import RoPE2D


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
    arch_type='hybrid_v1',
    param_names=['num_res_blocks', 'channels', 'board_size'],
    defaults={'num_res_blocks': 5, 'channels': 64, 'board_size': 15},
)
class HybridNet_v1(nn.Module):
    """
    CNN + Transformer 混合骨干双头网络 (Hybrid v1)

    结构:
      Stem → 5× 预激活 ResBlock(64) → 主干Transformer+RoPE → BN+ReLU → 分叉
                                                                    ├→ 策略头(1×1 Conv×3) → logits (225维)
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

        # ── 主干 Transformer 块 (Pre-LN, GELU, RoPE) ──
        self.num_heads = 4
        self.head_dim = channels // self.num_heads
        assert self.head_dim * self.num_heads == channels, \
            f"channels({channels}) 必须整除 num_heads({self.num_heads})"
        assert self.head_dim % 2 == 0, f"head_dim({self.head_dim}) 必须为偶数"

        self.trunk_rope = RoPE2D(dim=self.head_dim, board_size=board_size)

        self.trunk_qkv = nn.Linear(channels, channels * 3, bias=False)
        self.trunk_attn_out = nn.Linear(channels, channels, bias=False)
        self.trunk_dropout1 = nn.Dropout(0.1)
        self.trunk_ln1 = nn.LayerNorm(channels)

        self.trunk_ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(channels * 2, channels),
        )
        self.trunk_dropout2 = nn.Dropout(0.1)
        self.trunk_ln2 = nn.LayerNorm(channels)

        # ── 尾部归一化 ──
        self.final_bn = nn.BatchNorm2d(channels)

        # ── 策略头 (纯 1×1 Conv 降维) ──
        self.policy_conv1 = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(32)
        self.policy_conv2 = nn.Conv2d(32, 16, kernel_size=1, bias=False)
        self.policy_bn2 = nn.BatchNorm2d(16)
        self.policy_conv3 = nn.Conv2d(16, 1, kernel_size=1, bias=False)

        # ── 价值头 (交叉注意力 + MLP) ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, channels))

        self.value_cross_attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=4, batch_first=True
        )
        self.value_ln = nn.LayerNorm(channels)

        self.value_mlp = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1, bias=False),
        )

        # 安全掩码值（FP16 可表示，足够小但非 -inf）
        self.mask_val = -1e4

        self._init_weights()

    def _init_weights(self):
        """分类初始化：根据激活函数选择初始化策略"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # 策略头最后卷积层无 ReLU 跟随
                if 'policy_conv3' in name:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 主干 Transformer QKV / AttnOut 投影 (无激活函数)
                if 'trunk_qkv' in name or 'trunk_attn_out' in name:
                    nn.init.xavier_uniform_(m.weight)
                # 主干 FFN 第一层 (GELU 跟随)
                elif 'trunk_ffn.0' in name:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                # 主干 FFN 第二层 (无激活函数)
                elif 'trunk_ffn.3' in name:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
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
        N = self.board_squares  # 225

        # ── 占用掩码（用于策略头 + 价值头）──
        occupied_mask = (x[:, 0, :, :] + x[:, 1, :, :]).view(B, -1) > 0  # (B, 225)

        # ── Stem ──
        out = self.stem_conv(x)  # (B, C, 15, 15)

        # ── 残差塔 ──
        for block in self.res_blocks:
            out = block(out)  # (B, C, 15, 15)

        # ═══════════ 主干 Transformer + RoPE ═══════════
        out_flat = out.flatten(2).transpose(1, 2)  # (B, 225, C)

        # ── 子层1: 自注意力 + RoPE (Pre-LN) ──
        residual = out_flat
        x_norm = self.trunk_ln1(out_flat)

        qkv = self.trunk_qkv(x_norm)                               # (B, 225, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)                             # 各 (B, 225, C)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, hd)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.trunk_rope(q)
        k = self.trunk_rope(k)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.1, scale=scale
        )  # (B, H, N, hd)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.channels)
        attn_out = self.trunk_attn_out(attn_out)
        out_flat = residual + self.trunk_dropout1(attn_out)

        # ── 子层2: FFN (Pre-LN, GELU) ──
        residual = out_flat
        x_norm = self.trunk_ln2(out_flat)
        ffn_out = self.trunk_ffn(x_norm)
        out_flat = residual + self.trunk_dropout2(ffn_out)

        # Reshape 回卷积格式
        out = out_flat.transpose(1, 2).view(B, self.channels, H, W)

        # ── 尾部激活 ──
        out = F.relu(self.final_bn(out))  # (B, C, 15, 15)

        # ═══════════ 价值头 (交叉注意力 + key_padding_mask) ═══════════
        kv = out.flatten(2).transpose(1, 2)  # (B, 225, C)
        query = self.cls_token.expand(B, -1, -1)  # (B, 1, C)

        attn_out, _ = self.value_cross_attn(
            query=query,
            key=kv,
            value=kv,
            key_padding_mask=occupied_mask,
        )  # (B, 1, C)

        attn_out = attn_out + query
        attn_out = self.value_ln(attn_out)  # (B, 1, C)

        value = self.value_mlp(attn_out)  # (B, 1, 1)
        value = torch.tanh(value).squeeze(-1).squeeze(-1)  # (B,)

        if return_value_only:
            return None, value

        # ═══════════ 策略头 (纯 1×1 Conv 降维) ═══════════
        p = F.relu(self.policy_bn1(self.policy_conv1(out)))
        p = F.relu(self.policy_bn2(self.policy_conv2(p)))
        p = self.policy_conv3(p)  # (B, 1, H, W)
        policy_logits = p.view(B, -1)  # (B, 225)

        # 掩码已落子位置（FP16 安全）
        policy_logits = policy_logits.masked_fill(occupied_mask, self.mask_val)

        return policy_logits, value

    @property
    def arch_type(self) -> str:
        return "hybrid_v1"

    def get_config(self) -> dict:
        return {
            'arch_type': 'hybrid_v1',
            'num_res_blocks': len(self.res_blocks),
            'channels': self.channels,
            'board_size': self.board_size,
        }
