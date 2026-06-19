# test_transformer.py
"""
五子棋 Transformer 架构 (GoBangTransformer_v2) 可视化测试脚本
用法:
  python test_transformer.py
  python test_transformer.py --model_path checkpoints/transformer_pretrain/best_model.pt
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

# ====== 全局设置中文字体防乱码 ======
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# ====================================

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

        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        inv_freq_row = inv_freq[0::2]
        inv_freq_col = inv_freq[1::2]

        rows = torch.arange(board_size).repeat_interleave(board_size)
        cols = torch.arange(board_size).repeat(board_size)

        theta_row = rows[:, None].float() * inv_freq_row[None, :]
        theta_col = cols[:, None].float() * inv_freq_col[None, :]

        theta = torch.empty(board_size * board_size, half_dim)
        theta[:, 0::2] = theta_row
        theta[:, 1::2] = theta_col

        self.register_buffer('cos_table', torch.cos(theta))
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
#  多头自注意力 (含 2D-RoPE)
# ═══════════════════════════════════════════════════════════════

class MultiHeadSelfAttention2D(nn.Module):
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
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.rope_qk(q)
        k = self.rope_qk(k)

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
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = self.dropout1(x)
        x = residual + x

        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x
        return x


# ═══════════════════════════════════════════════════════════════
#  主模型：GoBangTransformer_v2
# ═══════════════════════════════════════════════════════════════

class GoBangTransformer_v2(nn.Module):
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

        self.embed = nn.Linear(3, d_model)
        self.ln_embed = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_expand, dropout, board_size)
            for _ in range(num_layers)
        ])

        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.value_query = nn.Parameter(torch.empty(1, 1, d_model))
        self.value_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        self.mask_val = -1e4
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'ffn.0' in name or 'ffn.3' in name:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.value_query)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        occupied_mask = (x[:, 0, :, :] + x[:, 1, :, :]).view(B, -1) > 0

        x = x.view(B, C, -1).transpose(1, 2)
        x = self.embed(x)
        x = self.ln_embed(x)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x)

        p = self.policy_head(x).squeeze(-1)
        p = p.masked_fill(occupied_mask, self.mask_val)

        query = self.value_query.expand(B, -1, -1)
        attn_out, _ = self.value_attn(
            query=query,
            key=x,
            value=x,
            key_padding_mask=occupied_mask
        )
        value = self.value_mlp(attn_out.squeeze(1)).squeeze(-1)
        value = torch.tanh(value)

        return p, value


# ═══════════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════════

def create_test_board(my_pieces, opp_pieces, last_move):
    board = np.zeros((3, 15, 15), dtype=np.float32)
    for r, c in my_pieces: board[0, r, c] = 1.0
    for r, c in opp_pieces: board[1, r, c] = 1.0
    if last_move is not None: board[2, last_move[0], last_move[1]] = 1.0
    return torch.from_numpy(board).unsqueeze(0)


def visualize_board(my_pieces, opp_pieces, last_move):
    board_str = ""
    for r in range(15):
        for c in range(15):
            if (r, c) == last_move: board_str += "X "
            elif (r, c) in my_pieces: board_str += "● "
            elif (r, c) in opp_pieces: board_str += "○ "
            else: board_str += "· "
        board_str += "\n"
    return board_str


def plot_policy_heatmap(policy_probs, my_pieces, opp_pieces, last_move, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.cm.Reds
    cmap.set_under('white')

    masked_probs = np.ma.masked_where(policy_probs < 0.001, policy_probs)
    im = ax.imshow(masked_probs, cmap=cmap, vmin=0.001, vmax=max(policy_probs.max(), 0.001), interpolation='nearest')

    for i in range(16): ax.axhline(i-0.5, color='black', linewidth=0.5)
    for i in range(16): ax.axvline(i-0.5, color='black', linewidth=0.5)

    for r, c in opp_pieces: ax.plot(c, r, 'o', markersize=15, markeredgecolor='black', markerfacecolor='white')
    for r, c in my_pieces: ax.plot(c, r, 'o', markersize=15, markeredgecolor='black', markerfacecolor='black')
    if last_move: ax.plot(last_move[1], last_move[0], 'x', markersize=12, markeredgewidth=2, color='blue')

    top3_idx = np.argsort(policy_probs.ravel())[-3:][::-1]
    for rank, idx in enumerate(top3_idx):
        r, c = divmod(idx, 15)
        prob = policy_probs[r, c]
        ax.text(c, r-0.4, f"{prob:.1%}", ha='center', va='center', fontsize=9, fontweight='bold', color='darkblue')

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, shrink=0.8, label='概率')
    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════
#  主程序
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="checkpoints/transformer_pretrain/best_model.pt")
    args = parser.parse_args()

    # 实例化 Transformer 模型，参数需与训练时一致
    model = GoBangTransformer_v2(
        d_model=64,
        num_heads=4,
        num_layers=5,
        ff_expand=4,
        dropout=0.1,
        board_size=15
    )

    if args.model_path:
        print(f"加载模型: {args.model_path}")
        try:
            ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state)
            print("✓ 权重加载成功！\n")
        except Exception as e:
            print(f"❌ 加载失败: {e}，使用随机权重\n")
    model.eval()

    # ==============================================
    #  多维度战术测试用例
    # ==============================================
    test_cases = [
        {
            "name": "基础1: 我方(黑)活四 - 必胜",
            "my_pieces": [(7,5), (7,6), (7,7), (7,8)],
            "opp_pieces": [(6,5), (6,6)],
            "last_move": (6,6),
            "expected_top1": [(7,4), (7,9)]
        },
        {
            "name": "基础2: 我方(黑)冲四活三 - 必胜",
            "my_pieces": [(7,5), (7,6), (7,7), (5,8), (6,8)],
            "opp_pieces": [(7,4), (8,7)],
            "last_move": (8,7),
            "expected_top1": [(7,8)]
        },
        {
            "name": "基础3: 对方(白)活三 - 需紧急防守",
            "my_pieces": [(8,4), (8,5), (9,6)],
            "opp_pieces": [(7,5), (7,6), (7,7)],
            "last_move": (7,7),
            "expected_top1": [(7,4), (7,8)]
        },
        {
            "name": "强制防守: 对方冲四 - 唯一解",
            "my_pieces": [(5,5), (6,6), (9,9)],
            "opp_pieces": [(7,5), (7,6), (7,7), (7,8)],
            "last_move": (7,8),
            "expected_top1": [(7,9)]
        },
        {
            "name": "高级进攻: 一子双活三 - 必胜",
            "my_pieces": [(7,6), (7,7), (5,8), (6,8)],
            "opp_pieces": [(6,6), (8,7)],
            "last_move": (8,7),
            "expected_top1": [(7,8)]
        },
        {
            "name": "极高难度: 反冲四防守 - 以攻代守反杀",
            "my_pieces": [(5,9), (6,9), (8,9), (9,9), (7,4)],
            "opp_pieces": [(7,5), (7,6), (7,7), (7,8), (4,9)],
            "last_move": (7,8),
            "expected_top1": [(7,9)]
        },
        {
            "name": "回归本源：无禁手五子棋黑方胜率",
            "my_pieces": [],
            "opp_pieces": [],
            "last_move": None,
            "expected_top1": [(7,7)]
        },
    ]

    with torch.no_grad():
        for case in test_cases:
            input_tensor = create_test_board(case["my_pieces"], case["opp_pieces"], case["last_move"])
            logits, value = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).view(15, 15).numpy()

            print(f"▶ 棋形: {case['name']}")
            print(visualize_board(case["my_pieces"], case["opp_pieces"], case["last_move"]))
            print(f"  Critic 评估: {value.item():+.4f} (正=我优, 负=敌优)")
            top1_r, top1_c = np.unravel_index(probs.argmax(), probs.shape)
            print(f"  Actor Top1 选择: ({top1_r}, {top1_c}) (概率: {probs.max():.2%})")

            is_correct = (top1_r, top1_c) in case["expected_top1"]
            print(f"  逻辑校验: {'✓ 符合预期' if is_correct else '✗ 不符合预期(网络尚未掌握此战术)'}\n")

            plot_policy_heatmap(probs, case["my_pieces"], case["opp_pieces"], case["last_move"],
                                f"{case['name']} | 胜率评估: {value.item():+.2f}")


if __name__ == '__main__':
    main()