# search/sampling.py
"""
核采样 (Nucleus / Top-p Sampling) 策略模块

从策略头输出的概率分布中选取累积概率达阈值的前 N 个动作，
按归一化概率采样。支持温度缩放控制探索程度。

与 MCTS 解耦：不依赖搜索树，直接使用神经网络原始策略输出。
"""

import numpy as np
from core.gamerules import GomokuRules

BOARD_SIZE = GomokuRules.BOARD_SIZE


def nucleus_sample_action(state, policy_probs, nucleus_p, temperature, candidate_radius):
    """
    核采样 (top-p sampling)：从策略头输出概率中选取累积概率达 nucleus_p 的前 N 个动作，
    按归一化概率采样。支持温度缩放控制探索程度。

    Args:
        state: 当前 GameState
        policy_probs: 神经网络策略头输出 (225,) float32
        nucleus_p: top-p 累积概率阈值 (0~1)
        temperature: 温度 (T>0: p^(1/T) 缩放; T<=0: 退化为 argmax)
        candidate_radius: 候选着法搜索半径

    Returns:
        选中的落子坐标 (r, c)
    """
    candidates = GomokuRules.get_candidates(state, radius=candidate_radius)
    if not candidates:
        return (BOARD_SIZE // 2, BOARD_SIZE // 2)

    move_probs = []
    for move in candidates:
        idx = move[0] * BOARD_SIZE + move[1]
        move_probs.append((move, policy_probs[idx]))

    total = sum(p for _, p in move_probs)
    if total > 1e-8:
        move_probs = [(m, p / total) for m, p in move_probs]
    else:
        n = len(move_probs)
        move_probs = [(m, 1.0 / n) for m, _ in move_probs]

    if temperature <= 0:
        best_move, _ = max(move_probs, key=lambda x: x[1])
        return best_move

    if temperature != 1.0:
        inv_temp = 1.0 / temperature
        move_probs = [(m, p ** inv_temp) for m, p in move_probs]
        total = sum(p for _, p in move_probs)
        if total > 1e-10:
            move_probs = [(m, p / total) for m, p in move_probs]

    move_probs.sort(key=lambda x: x[1], reverse=True)

    cumsum = 0.0
    nucleus = []
    for move, prob in move_probs:
        nucleus.append((move, prob))
        cumsum += prob
        if cumsum >= nucleus_p:
            break

    probs = np.array([p for _, p in nucleus], dtype=np.float32)
    probs /= probs.sum()
    idx = np.random.choice(len(nucleus), p=probs)
    return nucleus[idx][0]
