# agents/neural/transformer_network.py
"""
基于 Transformer 架构的五子棋神经网络 (GoBangTransformer_v2)

架构: 3通道输入 + Pre-LN Transformer块 + FFN + GAP价值头
  - 2D-RoPE 旋转位置编码（15×15棋盘逐对旋转）
  - Pre-LN 残差Transformer块 × 5（含Dropout正则化）
  - 策略头: MLP 逐位置投影 → logits (225维)
  - 价值头: MLP + GAP + tanh → [-1, 1]

输入: (B, 3, 15, 15) - 3通道 (己方/对方/上一步标记) [兼容现有接口]
输出: policy_logits (B, 225), value (B,)

与 ActorCriticNet(CNN) 保持完全相同的输入输出接口，
可无缝接入 MCTS、AZAgent、竞技场等现有系统。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  2D 旋转位置编码 (RoPE)
# ═══════════════════════════════════════════════════════════════

class RoPE2D(nn.Module):
    """
    修正版 2D 旋转位置编码。

    将 head_dim 维度的特征按每两个维度组成一对（复数对），
    对每一对施加由行索引(row)和列索引(col)共同决定的旋转角度。
    行角度和列角度交错排列：[row0, col0, row1, col1, ...]

    设计动机:
      - 五子棋棋盘具有2D几何结构，需要编码行列信息
      - 交错排列避免了行列信息的纠缠
      - 逐对旋转保持正交性，不改变向量模长

    预计算:
      - 为棋盘每个格子 (row, col) 预计算 cos/sin 表，形状 (225, half_dim)
      - 频率按 θ = 10000^(-2i/d) 生成，i = 0, 2, 4, 6 (共4个频率)

    Args:
        head_dim: 注意力头维度（必须是偶数，默认16）
        board_size: 棋盘大小（默认15）
    """

    def __init__(self, head_dim: int = 16, board_size: int = 15):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim 必须为偶数，当前值: {head_dim}")

        self.head_dim = head_dim
        self.board_size = board_size
        half_dim = head_dim // 2  # 8对

        # 生成频率（每对维度共享一个频率）
        # inv_freq shape: (half_dim // 2,) = (4,)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim)
        )

        # 为棋盘每个格子生成行索引和列索引
        rows = torch.arange(board_size).repeat_interleave(board_size)  # (225,)
        cols = torch.arange(board_size).repeat(board_size)             # (225,)

        # 计算角度: 每个位置产生 4个行角度 + 4个列角度
        theta_row = rows[:, None].float() * inv_freq[None, :]  # (225, 4)
        theta_col = cols[:, None].float() * inv_freq[None, :]  # (225, 4)

        # 交错拼接成 (225, 8)，顺序: [row0, col0, row1, col1, row2, col2, row3, col3]
        theta = torch.empty(board_size * board_size, half_dim)
        theta[:, 0::2] = theta_row  # 偶数索引放行角度
        theta[:, 1::2] = theta_col  # 奇数索引放列角度

        # 预计算余弦、正弦表，注册为 buffer（不参与梯度但随模型移动）
        self.register_buffer('cos_table', torch.cos(theta))  # (225, half_dim)
        self.register_buffer('sin_table', torch.sin(theta))  # (225, half_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量施加 2D RoPE 旋转。

        Args:
            x: (B, H, N, D) 其中 D=head_dim, N=board_size*board_size

        Returns:
            施加旋转后的张量，形状同输入 (B, H, N, D)
        """
        B, H, N, D = x.shape
        half_dim = D // 2  # 8

        # 将最后一维重塑为 half_dim 对: (偶, 奇)
        x = x.view(B, H, N, half_dim, 2)  # (B, H, N, 8, 2)
        x_even = x[..., 0]  # (B, H, N, 8) 偶数位置
        x_odd = x[..., 1]   # (B, H, N, 8) 奇数位置

        # 取出对应位置的 cos/sin，自动广播为 (1, 1, N, half_dim)
        cos = self.cos_table[None, None, :, :]  # (1, 1, 225, 8)
        sin = self.sin_table[None, None, :, :]

        # 逐对旋转公式 (将每对维度视为复数 a+ib 乘以 e^{iθ})
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # 拼回原形状 (B, H, N, D)
        out = torch.stack([out_even, out_odd], dim=-1)  # (B, H, N, 8, 2)
        out = out.view(B, H, N, D)

        return out


# ═══════════════════════════════════════════════════════════════
#  多头自注意力 (含 2D-RoPE)
# ═══════════════════════════════════════════════════════════════

class MultiHeadSelfAttention2D(nn.Module):
    """
    含修正 2D-RoPE 的多头自注意力（Pre-LN 兼容）。

    流程: q/k/v投影 → 重塑多头 → RoPE2D施加(q,k) → 缩放点积注意力 → 输出投影

    Args:
        d_model: 模型维度（默认64）
        num_heads: 注意力头数（默认4）
        dropout: 注意力 dropout 率（默认0.0）
        board_size: 棋盘大小（默认15）
    """

    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.0, board_size: int = 15):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 64/4 = 16

        # Q/K/V 投影（无 bias）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # 2D RoPE
        self.rope_2d = RoPE2D(head_dim=self.head_dim, board_size=board_size)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) 其中 N = board_size * board_size

        Returns:
            (B, N, d_model)
        """
        B, N, D = x.shape

        # 投影
        q = self.q_proj(x)  # (B, N, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 重塑为多头: (B, H, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 施加 2D RoPE（仅对 q 和 k）
        q = self.rope_2d(q)
        k = self.rope_2d(k)

        # ★ 使用 PyTorch 内置 scaled_dot_product_attention
        #   自动选择最优后端: FlashAttention (CUDA+fp16/bf16) → Memory Efficient → 朴素实现
        #   dropout 仅在训练时生效，eval 模式自动忽略
        #   RoPE 在 q,k 上已施加完毕，FlashAttention 完全兼容
        scale = math.sqrt(self.head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if isinstance(self.attn_dropout, nn.Dropout) else 0.0,
            scale=scale,
        )  # (B, H, N, head_dim)

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        # 输出投影
        out = self.out_proj(out)

        return out


# ═══════════════════════════════════════════════════════════════
#  Pre-LN Transformer 块
# ═══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Pre-LN 残差 Transformer 块。

    结构:
        x = x + MHA(LN1(x))
        x = x + FFN(LN2(x))

    Pre-LN 优势：归一化在残差分支内部，梯度可通过 skip connection 直传，
    训练更稳定，尤其适合较小模型。

    FFN 采用 4 倍扩展 + GELU 激活，标准 Transformer 配置。

    Args:
        d_model: 模型维度（默认64）
        num_heads: 注意力头数（默认4）
        ff_expand: FFN 扩展倍数（默认4）
        dropout: Dropout 率（默认0.1）
        board_size: 棋盘大小（默认15）
    """

    def __init__(
        self, d_model: int = 64, num_heads: int = 4, ff_expand: int = 4,
        dropout: float = 0.1, board_size: int = 15
    ):
        super().__init__()

        # 子层1: Pre-LN + MHA
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention2D(
            d_model=d_model, num_heads=num_heads,
            dropout=dropout, board_size=board_size
        )
        self.dropout1 = nn.Dropout(dropout)

        # 子层2: Pre-LN + FFN
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
        """
        Args:
            x: (B, N, d_model)

        Returns:
            (B, N, d_model)
        """
        # 子层1: Pre-LN + MHA + 残差
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.dropout1(x)
        x = residual + x

        # 子层2: Pre-LN + FFN + 残差
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x

        return x


# ═══════════════════════════════════════════════════════════════
#  GoBangTransformer_v2 主模型
# ═══════════════════════════════════════════════════════════════

class GoBangTransformer_v2(nn.Module):
    """
    改进版 Transformer 五子棋网络。

    架构:
        Embedding: Linear(3→64) + LayerNorm
        ×5 TransformerBlock (Pre-LN, 含2D-RoPE + Dropout)
        策略头: Linear(64→32)→ReLU→Linear(32→16)→ReLU→Linear(16→1)
        价值头: Linear(64→32)→ReLU→Linear(32→16)→ReLU→Linear(16→1) → GAP → tanh

    掩码策略: 在嵌入前基于原始3通道输入计算 occupied_mask，
    策略头输出后用 occupied_mask 将已占用位置设为 -inf。

    输入: (B, 3, 15, 15)  输出: policy_logits (B, 225), value (B,)

    Args:
        d_model: 模型维度（默认64）
        num_heads: 注意力头数（默认4）
        num_layers: Transformer 块数（默认5）
        ff_expand: FFN 扩展倍数（默认4）
        dropout: Dropout 率（默认0.1）
        board_size: 棋盘大小（默认15）
    """

    def __init__(
        self, d_model: int = 64, num_heads: int = 4, num_layers: int = 5,
        ff_expand: int = 4, dropout: float = 0.1, board_size: int = 15
    ):
        super().__init__()
        self.board_size = board_size
        self.board_squares = board_size * board_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_expand = ff_expand
        self.dropout_rate = dropout

        # ═══════════════ 嵌入层 ═══════════════
        # 每个位置有3个值(己方/对方/上一步) → d_model 维
        self.embed = nn.Linear(3, d_model)
        self.ln_embed = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # ═══════════════ Transformer 块 × N ═══════════════
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, num_heads=num_heads,
                ff_expand=ff_expand, dropout=dropout, board_size=board_size
            )
            for _ in range(num_layers)
        ])

        # ═══════════════ 策略头 (Policy Head) ═══════════════
        # 逐位置独立 MLP: 64→32→16→1
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # ═══════════════ 价值头 (Value Head) ═══════════════
        # 逐位置 MLP + 全局平均池化 → tanh[-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化。

        - Linear: Xavier Uniform (适合 Transformer + GELU/ReLU)
        - LayerNorm: weight=1, bias=0
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        前向传播。

        Args:
            x: (B, 3, 15, 15) - 3通道棋盘张量（与 ActorCriticNet 接口一致）

        Returns:
            (policy_logits, value):
              - policy_logits: (B, 225) 已对非法位置施加 -inf 掩码
              - value: (B,) 全局胜率估计，tanh 压缩到 [-1, 1]
        """
        B, C, H, W = x.shape

        # ── 在嵌入前根据原始输入计算占用掩码 ──
        # 通道0=己方, 通道1=对方, 任一为1即表示该位置已被占用
        occupied_mask = (x[:, 0, :, :] + x[:, 1, :, :]).view(B, -1) > 0  # (B, 225) bool

        # ── 1. 重塑并嵌入 ──
        # (B, 3, 15, 15) → (B, 225, 3)
        x = x.view(B, C, -1).transpose(1, 2)  # (B, 225, 3)

        # Linear 嵌入: (B, 225, 3) → (B, 225, 64)
        x = self.embed(x)
        x = self.ln_embed(x)
        x = self.embed_dropout(x)

        # ── 2. 通过 N 个 Pre-LN Transformer 块 ──
        for block in self.blocks:
            x = block(x)  # (B, 225, d_model)

        # ── 3. 策略头 ──
        p = self.policy_head(x)          # (B, 225, 1)
        policy_logits = p.squeeze(-1)    # (B, 225)

        # ★ 使用嵌入前保存的正确掩码（避免坐标信息稀释后的Bug）
        policy_logits = policy_logits.masked_fill(occupied_mask, -float('inf'))

        # ── 4. 价值头 ──
        v = self.value_head(x)           # (B, 225, 1)
        v = v.squeeze(-1)                # (B, 225)
        value = v.mean(dim=1)            # (B,) 全局平均池化
        value = torch.tanh(value)        # (B,) 压缩到 [-1, 1]

        return policy_logits, value

    @property
    def arch_type(self) -> str:
        """返回架构标识，用于 AZAgent 自动推断网络类型。"""
        return "transformer"

    def get_config(self) -> dict:
        """返回网络配置字典，用于存档和恢复。"""
        return {
            'arch_type': 'transformer',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_expand': self.ff_expand,
            'dropout': self.dropout_rate,
            'board_size': self.board_size,
        }