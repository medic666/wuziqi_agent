"""
基于 Transformer 架构的五子棋神经网络 (GoBangTransformer_v2)

架构: 3通道输入 + Pre-LN Transformer块 + FFN + 改进价值头
  - 2D-RoPE 旋转位置编码（施加于注意力 Q/K，15×15棋盘逐对旋转）
  - Pre-LN 残差Transformer块 × 5（含Dropout正则化）
  - 策略头: MLP 逐位置投影 → logits (225维)（使用安全掩码 -1e4）
  - 价值头: 可学习全局查询 + 末端交叉注意力池化（带 Mask）→ MLP → tanh[-1,1]

输入: (B, 3, 15, 15) - 3通道 (己方/对方/上一步标记) [兼容现有接口]
输出: policy_logits (B, 225), value (B,)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  2D 旋转位置编码
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  多头自注意力 (含 2D-RoPE，修正 scale)
# ═══════════════════════════════════════════════════════════════

class MultiHeadSelfAttention2D(nn.Module):
    """多头自注意力：Q/K/V 投影 + 2D-RoPE + 缩放点积注意力（scale = 1/√head_dim）"""

    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.0, board_size: int = 15):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.rope_qk = RoPE2D(dim=self.head_dim, board_size=board_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,N,hd)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 施加 RoPE
        q = self.rope_qk(q)
        k = self.rope_qk(k)

        # ★ 修正 scale：默认为 1/√E，此处显式传入安全值
        scale = 1.0 / math.sqrt(self.head_dim)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if isinstance(self.attn_dropout, nn.Dropout) else 0.0,
            scale=scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


# ═══════════════════════════════════════════════════════════════
#  Pre-LN Transformer 块
# ═══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Pre-LN 残差块：MHA + FFN（GELU 激活）"""

    def __init__(self, d_model: int = 64, num_heads: int = 4, ff_expand: int = 4,
                 dropout: float = 0.1, board_size: int = 15):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention2D(d_model, num_heads, dropout, board_size)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        ffn_hidden = d_model * ff_expand
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 子层1
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.dropout1(x)
        x = residual + x

        # 子层2
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x
        return x


# ═══════════════════════════════════════════════════════════════
#  主模型：GoBangTransformer_v2（改进价值头 + 安全掩码）
# ═══════════════════════════════════════════════════════════════

class GoBangTransformer_v2(nn.Module):
    """
    改进版 Transformer 五子棋网络。

    改动:
      - 注意力 scale 修正为 1/√E
      - 价值头采用可学习查询 + 末端交叉注意力池化（带 Mask），彻底隔离占用位置梯度
      - 掩码使用 -1e4（FP16 安全）
      - 权重初始化适配各层激活类型
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, num_layers: int = 5,
                 ff_expand: int = 4, dropout: float = 0.1, board_size: int = 15):
        super().__init__()
        self.board_size = board_size
        self.board_squares = board_size * board_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_expand = ff_expand
        self.dropout_rate = dropout

        # ── 嵌入层 ──
        self.embed = nn.Linear(3, d_model)
        self.ln_embed = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # ── Transformer 块 ──
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_expand, dropout, board_size)
            for _ in range(num_layers)
        ])

        # ── 策略头 ──
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # ── 价值头（新设计：可学习查询 + 交叉注意力池化）──
        self.value_query = nn.Parameter(torch.empty(1, 1, d_model))
        # 使用 PyTorch 内置多头注意力，batch_first=True
        self.value_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # 安全掩码值（FP16 可表示，足够小但非 -inf）
        self.mask_val = -1e4

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """
        分类初始化，避免 Kaiming 增益错误。
        - 无激活函数的线性层（Q/K/V/O、嵌入、价值查询投影、价值 MLP 第一层前的 Linear）：xavier_uniform
        - 带 GELU 激活的 FFN 第一层：kaiming_uniform（针对 relu 增益，但 GELU 近似可接受）
        - LayerNorm：标准常数
        - value_query：xavier_uniform 仅支持 2D，对 Parameter 单独处理
        """
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # 判断该 Linear 后面是否紧接激活函数（通过所在模块名字简单判断）
                if 'ffn.0' in name or 'ffn.3' in name:  # FFN 的第一层 (0) 或第二层 (3，后面无激活)
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # GELU 约等于 ReLU 增益
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    # 其他 Linear（Q/K/V/O、嵌入、策略头、价值头 MLP 内部）使用 xavier_uniform
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 专门初始化 value_query 参数
        nn.init.xavier_uniform_(self.value_query)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播。
        Args:
            x: (B, 3, 15, 15)
        Returns:
            (policy_logits, value): logits (B,225), value (B,)
        """
        B, C, H, W = x.shape

        # 占用掩码（嵌入前计算，保证准确）
        # True 表示该位置已有棋子 (需被屏蔽)
        occupied_mask = (x[:, 0, :, :] + x[:, 1, :, :]).view(B, -1) > 0  # (B, 225)

        # 展平并嵌入
        x = x.view(B, C, -1).transpose(1, 2)               # (B, 225, 3)
        x = self.embed(x)                                    # (B, 225, d_model)
        x = self.ln_embed(x)
        x = self.embed_dropout(x)

        # Transformer 骨干
        for block in self.blocks:
            x = block(x)                                     # (B, 225, d_model)

        # --- 策略头 ---
        p = self.policy_head(x).squeeze(-1)                  # (B, 225)
        # 安全掩码：使用 FP16 安全的负数代替 -inf
        p = p.masked_fill(occupied_mask, self.mask_val)

        # --- 价值头（CLS 注意力池化）---
        query = self.value_query.expand(B, -1, -1)           # (B, 1, d_model)
        
        # ★ 修复 Bug：传入 key_padding_mask，屏蔽已占用位置，切断其梯度污染
        attn_out, _ = self.value_attn(
            query=query,
            key=x,
            value=x,
            key_padding_mask=occupied_mask
        )                                                    # (B, 1, d_model)
        
        value = self.value_mlp(attn_out.squeeze(1))          # (B, 1)
        value = value.squeeze(-1)                             # (B,)
        value = torch.tanh(value)                            # (B,) 范围 [-1, 1]

        return p, value

    @property
    def arch_type(self) -> str:
        return "transformer"

    def get_config(self) -> dict:
        return {
            'arch_type': 'transformer',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_expand': self.ff_expand,
            'dropout': self.dropout_rate,
            'board_size': self.board_size,
        }