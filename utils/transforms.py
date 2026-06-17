# utils/transforms.py
"""
8向对称变换工具（D4二面体群）

五子棋棋盘具有 D4 对称性（旋转+翻转），训练时通过
8种变换增强数据，等价于将数据量扩大8倍。

搬自: utils.py 前半部分
"""

import numpy as np


def transform_2d(arr: np.ndarray, tid: int) -> np.ndarray:
    """
    对 2D 数组施加 D4 二面体群对称变换。
    
    D4 群包含 8 种对称操作:
      tid=0: 原样 (恒等变换)
      tid=1: 逆时针旋转 90°
      tid=2: 旋转 180°
      tid=3: 顺时针旋转 90°
      tid=4: 左右翻转 (水平镜像)
      tid=5: 上下翻转 (垂直镜像)
      tid=6: 转置 (主对角线镜像)
      tid=7: 反对角翻转 (副对角线镜像)
    
    Args:
        arr: 输入 2D 数组
        tid: 变换 ID (0-7)
        
    Returns:
        变换后的数组（C-contiguous）
        
    Raises:
        ValueError: tid 不在 0-7 范围内
    """
    if tid == 0:
        result = arr
    elif tid == 1:
        result = np.rot90(arr, k=3)  # 逆时针90° = 顺时针3次
    elif tid == 2:
        result = np.rot90(arr, k=2)
    elif tid == 3:
        result = np.rot90(arr, k=1)
    elif tid == 4:
        result = np.fliplr(arr)
    elif tid == 5:
        result = np.flipud(arr)
    elif tid == 6:
        result = arr.T
    elif tid == 7:
        result = arr.T[::-1, ::-1]
    else:
        raise ValueError(f"Invalid transform_id: {tid} (must be 0-7)")
    
    return np.ascontiguousarray(result)


def transform_state(state_3d: np.ndarray, tid: int) -> np.ndarray:
    """
    对 3 通道状态张量施加 D4 对称变换。
    
    输入形状: (3, 15, 15) — 3个通道各自独立变换。
    通道 0: 己方棋子, 通道 1: 对方棋子, 通道 2: 上一步标记。
    
    所有通道使用相同的 tid 变换，保证空间关系一致。
    
    Args:
        state_3d: (3, H, W) 状态张量
        tid: 变换 ID (0-7)
        
    Returns:
        (3, H, W) 变换后的状态张量
    """
    result = np.empty_like(state_3d)
    for ch in range(3):
        result[ch] = transform_2d(state_3d[ch], tid)
    return result