# test.py
"""
五子棋 Actor-Critic 可视化测试脚本 (适配内置Mask新架构 + 多维度战术测试)

用法:
  python test.py
  python test.py --model_path checkpoints/joint_pretrain/best_model.pt
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from network import ActorCriticNet

# ====== 全局设置中文字体防乱码 ======
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
# ====================================

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
    
    # 修改4：将mask阈值从0.01降至0.001，防止开阔局面下所有合法位置被过滤导致热力图空白
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

def main():
    parser = argparse.ArgumentParser()
    # 修改1：修复文件路径反斜杠转义问题，改用正斜杠
    parser.add_argument('--model_path', type=str, default="checkpoints/az_train/best_model.pt")
    args = parser.parse_args()

    # 实例化模型 (可根据实际训练的参数调整 num_res_blocks)
    model = ActorCriticNet(num_res_blocks=4, channels=128)
    if args.model_path:
        print(f"加载模型: {args.model_path}")
        try:
            ckpt = torch.load(args.model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            print("✓ 权重加载成功！\n")
        except Exception as e:
            print(f"❌ 加载失败: {e}，使用随机权重\n")
    model.eval()

    # ==============================================
    #  多维度战术测试用例
    # ==============================================
    test_cases = [
        # --- 1. 基础攻防测试 ---
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

        # --- 2. 绝对强制防守测试 (漏防即死) ---
        {
            "name": "强制防守: 对方冲四 - 唯一解",
            "my_pieces": [(5,5), (6,6), (9,9)],        
            "opp_pieces": [(7,5), (7,6), (7,7), (7,8)], # 白棋横向冲四
            "last_move": (7,8),
            "expected_top1": [(7,9)]                    # 黑棋必须堵住这一头，别无选择
        },

        # --- 3. 高级进攻手筋测试 ---
        {
            "name": "高级进攻: 一子双活三 - 必胜",
            "my_pieces": [(7,6), (7,7), (5,8), (6,8)], # 横向(7,6-7)缺活三，纵向(5-6,8)缺活三
            "opp_pieces": [(6,6), (8,7)],              
            "last_move": (8,7),
            "expected_top1": [(7,8)]                    # 走(7,8)同时形成横向和纵向两个活三，对方无法同时防守
        },

        # --- 4. 极高难度：以攻代守（反先手） ---
        {
            "name": "极高难度: 反冲四防守 - 以攻代守反杀",
            # 修改3：修正了棋子摆放，使(7,9)落下后真正形成纵向五连绝杀
            "my_pieces": [(5,9), (6,9), (8,9), (9,9), (7,4)], # 我方纵向(5,6,8,9)有一缺口(7,9)
            "opp_pieces": [(7,5), (7,6), (7,7), (7,8), (4,9)], # 对方横向冲四(7,5-8)，且堵住了我方(4,9)
            "last_move": (7,8),                         # 对方刚走(7,8)冲四！
            "expected_top1": [(7,9)]                    # 我方必须走(7,9)防守，但这步恰好把我方纵向连成五！绝地反杀
        },

        # --- 5. 回归本源：无禁手五子棋黑方胜率 ---
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
            
            # 网络内部已经对非法位置进行了Mask，并输出形状为 (B, 225) 的 logits
            logits, value = model(input_tensor)
            
            # 修改2：规范维度操作，先squeeze(0)去除batch维度，再view(15, 15)
            probs = torch.softmax(logits, dim=1).squeeze(0).view(15, 15).numpy()

            print(f"▶ 棋形: {case['name']}")
            print(visualize_board(case["my_pieces"], case["opp_pieces"], case["last_move"]))
            print(f"  Critic 评估: {value.item():+.4f} (正=我优, 负=敌优)")
            top1_r, top1_c = np.unravel_index(probs.argmax(), probs.shape)
            print(f"  Actor Top1 选择: ({top1_r}, {top1_c}) (概率: {probs.max():.2%})")
            
            # 验证是否符合预期
            is_correct = (top1_r, top1_c) in case["expected_top1"]
            print(f"  逻辑校验: {'✓ 符合预期' if is_correct else '✗ 不符合预期(网络尚未掌握此战术)'}\n")
            
            plot_policy_heatmap(probs, case["my_pieces"], case["opp_pieces"], case["last_move"], 
                                f"{case['name']} | 胜率评估: {value.item():+.2f}")

if __name__ == '__main__':
    main()