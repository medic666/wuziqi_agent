"""
2D 旋转位置编码 (RoPE) — 独立共享模块

用于 CNN Hybrid 策略头和 Transformer 架构的注意力 Q/K 位置编码。

原理: 对 15×15 棋盘的每个位置，在注意力 Q/K 上施加逐对旋转，
      使注意力分数天然编码相对位置 (Δrow, Δcol)。
"""

import torch
import torch.nn as nn


class RoPE2D(nn.Module):
    """2D 旋转位置编码，对注意力 Q/K 施加逐对旋转。"""

    def __init__(self, dim: int = 64, board_size: int = 15, base: float = 20.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim 必须为偶数，当前值: {dim}")

        self.dim = dim
        self.board_size = board_size
        self.base = base
        half_dim = dim // 2

        # 频率生成
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        inv_freq_row = inv_freq[0::2]  # 偶数索引：行
        inv_freq_col = inv_freq[1::2]  # 奇数索引：列

        rows = torch.arange(board_size).repeat_interleave(board_size)  # (225,)
        cols = torch.arange(board_size).repeat(board_size)             # (225,)

        theta_row = rows[:, None].float() * inv_freq_row[None, :]
        theta_col = cols[:, None].float() * inv_freq_col[None, :]

        theta = torch.empty(board_size * board_size, half_dim)
        theta[:, 0::2] = theta_row
        theta[:, 1::2] = theta_col

        self.register_buffer('cos_table', torch.cos(theta))  # (225, half_dim)
        self.register_buffer('sin_table', torch.sin(theta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, N, D = x.shape
        half_dim = D // 2
        x = x.view(B, H, N, half_dim, 2)
        x_even, x_odd = x[..., 0], x[..., 1]

        cos = self.cos_table[None, None, :, :]
        sin = self.sin_table[None, None, :, :]

        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos

        out = torch.stack([out_even, out_odd], dim=-1)
        return out.view(B, H, N, D)
