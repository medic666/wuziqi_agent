# utils.py
"""五子棋训练系统公共工具函数（8向对称变换 + 棋谱图片生成）"""

import os
import numpy as np


# ======================== 8向同性变换工具 ========================

def transform_2d(arr: np.ndarray, tid: int) -> np.ndarray:
    """
    对 2D 数组施加 D4 二面体群对称变换。
    tid: 0=原样, 1=逆时针90°, 2=180°, 3=顺时针90°, 4=左右翻转, 5=上下翻转, 6=转置, 7=反对角翻转
    """
    if tid == 0:   result = arr
    elif tid == 1: result = np.rot90(arr, k=3)
    elif tid == 2: result = np.rot90(arr, k=2)
    elif tid == 3: result = np.rot90(arr, k=1)
    elif tid == 4: result = np.fliplr(arr)
    elif tid == 5: result = np.flipud(arr)
    elif tid == 6: result = arr.T
    elif tid == 7: result = arr.T[::-1, ::-1]
    else: raise ValueError(f"Invalid transform_id: {tid}")
    return np.ascontiguousarray(result)


def transform_state(state_3d: np.ndarray, tid: int) -> np.ndarray:
    """对 3 通道状态施加 D4 对称变换"""
    result = np.empty_like(state_3d)
    for ch in range(3):
        result[ch] = transform_2d(state_3d[ch], tid)
    return result


# ======================== 棋谱图片生成 ========================

def save_board_image(image_dir: str, image_idx: int, history, winner: int):
    """
    将一局棋的落子过程保存为 PNG 图片。
    
    Args:
        image_dir: 保存目录
        image_idx: 图片编号（用于文件名）
        history: 落子序列 [(r1,c1), (r2,c2), ...]
        winner: 1=黑胜, 2=白胜, 0=平局
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    cell, margin = 28, 18
    size = 14 * cell + 2 * margin
    img = Image.new('RGB', (size, size), '#DEB887')
    draw = ImageDraw.Draw(img)

    # 画网格线
    for i in range(15):
        p = margin + i * cell
        draw.line([(p, margin), (p, margin + 14 * cell)], fill='#444', width=1)
        draw.line([(margin, p), (margin + 14 * cell, p)], fill='#444', width=1)

    # 画星位
    for r, c in [(7, 7), (3, 3), (3, 11), (11, 3), (11, 11)]:
        cx, cy = margin + c * cell, margin + r * cell
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill='#444')

    # 加载字体
    font = None
    for font_path in [
        "arialbd.ttf", "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            font = ImageFont.truetype(font_path, 10)
            break
        except (IOError, OSError):
            continue
    if font is None:
        font = ImageFont.load_default()

    # 画棋子
    radius = cell // 2 - 2
    for step_idx, (r, c) in enumerate(history):
        cx, cy = margin + c * cell, margin + r * cell
        fill_color = 'black' if step_idx % 2 == 0 else 'white'
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=fill_color, outline='black')

        step_num = str(step_idx + 1)
        text_color = 'white' if step_idx % 2 == 0 else 'black'
        try:
            draw.text((cx, cy), step_num, fill=text_color, font=font, anchor="mm")
        except TypeError:
            bbox = draw.textbbox((0, 0), step_num, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx - tw / 2, cy - th / 2), step_num, fill=text_color, font=font)

    # 标记最后一手
    if history:
        lr, lc = history[-1]
        cx, cy = margin + lc * cell, margin + lr * cell
        draw.ellipse([cx - radius - 2, cy - radius - 2, cx + radius + 2, cy + radius + 2],
                     outline='red', width=2)

    winner_str = {1: "Bwin", 2: "Wwin", 0: "Draw"}.get(winner, "?")
    os.makedirs(image_dir, exist_ok=True)
    img.save(os.path.join(image_dir, f"game_{image_idx:04d}_{winner_str}_{len(history)}s.png"))