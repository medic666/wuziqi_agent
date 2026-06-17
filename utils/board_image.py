# utils/board_image.py
"""
棋谱图片生成工具

将对局的落子过程导出为 PNG 图片，便于可视化分析。
包含棋盘网格、星位、棋子编号、最后一手标记、胜负标注。

需要: pip install pillow (PIL)

搬自: utils.py 后半部分
"""

import os
import numpy as np


def save_board_image(image_dir: str, image_idx: int, history, winner: int):
    """
    将一局棋的落子过程保存为 PNG 图片。
    
    功能:
      - 绘制 15×15 棋盘网格和 5 个星位
      - 按顺序绘制棋子，标注步数
      - 红色圆圈标记最后一手
      - 文件名含胜负信息和步数
    
    Args:
        image_dir: 保存目录（不存在则自动创建）
        image_idx: 图片编号（用于文件名）
        history: 落子序列 [(r1,c1), (r2,c2), ...]
        winner: 1=黑胜, 2=白胜, 0=平局
        
    文件名格式: game_{image_idx:04d}_{Bwin/Wwin/Draw}_{steps}s.png
    
    注意: 如果 PIL 未安装，函数静默跳过（不报错）。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # PIL 未安装，静默跳过
        return

    cell, margin = 28, 18
    size = 14 * cell + 2 * margin
    img = Image.new('RGB', (size, size), '#DEB887')
    draw = ImageDraw.Draw(img)

    # ── 画网格线 ──
    for i in range(15):
        p = margin + i * cell
        draw.line([(p, margin), (p, margin + 14 * cell)], fill='#444', width=1)
        draw.line([(margin, p), (margin + 14 * cell, p)], fill='#444', width=1)

    # ── 画星位 ──
    for r, c in [(7, 7), (3, 3), (3, 11), (11, 3), (11, 11)]:
        cx, cy = margin + c * cell, margin + r * cell
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill='#444')

    # ── 加载字体 ──
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

    # ── 画棋子 ──
    radius = cell // 2 - 2
    for step_idx, (r, c) in enumerate(history):
        cx, cy = margin + c * cell, margin + r * cell
        # 黑先白后交替
        fill_color = 'black' if step_idx % 2 == 0 else 'white'
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=fill_color, outline='black')

        # 标步数
        step_num = str(step_idx + 1)
        text_color = 'white' if step_idx % 2 == 0 else 'black'
        try:
            draw.text((cx, cy), step_num, fill=text_color, font=font, anchor="mm")
        except TypeError:
            # 兼容旧版 PIL（无 anchor 参数）
            bbox = draw.textbbox((0, 0), step_num, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((cx - tw / 2, cy - th / 2), step_num, fill=text_color, font=font)

    # ── 标记最后一手（红色圆圈） ──
    if history:
        lr, lc = history[-1]
        cx, cy = margin + lc * cell, margin + lr * cell
        draw.ellipse([cx - radius - 2, cy - radius - 2, cx + radius + 2, cy + radius + 2],
                     outline='red', width=2)

    # ── 保存 ──
    winner_str = {1: "Bwin", 2: "Wwin", 0: "Draw"}.get(winner, "?")
    os.makedirs(image_dir, exist_ok=True)
    img.save(os.path.join(image_dir, f"game_{image_idx:04d}_{winner_str}_{len(history)}s.png"))