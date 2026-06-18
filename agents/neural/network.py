# agents/neural/network.py
"""
五子棋神经网络模型定义

架构: 预激活残差块 + GAP价值头 + 纯净卷积策略头 (v9.2)

预激活范式 (He et al. 2016):
  Stem:     Conv (无BN/ReLU)
  ResBlock: BN → ReLU → Conv → BN → ReLU → Conv → (+x)
  尾部:     BN → ReLU → 送入头部

当前配置: 4个残差块 + 128通道，约124万参数

搬自: network.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    预激活残差块: BN → ReLU → Conv → BN → ReLU → Conv → (+x)
    
    预激活范式的优势:
      - 梯度可以直接通过 skip connection 传播，缓解梯度消失
      - BN 在卷积之前，使训练更稳定
    """
    def __init__(self, channels: int = 128):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: 预激活 → 卷积 → 预激活 → 卷积 → 残差加"""
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + x  # 残差连接


class ActorCriticNet(nn.Module):
    """
    Actor-Critic 双头神经网络。
    
    结构:
      Stem → N个预激活ResBlock → BN+ReLU → 分叉
                                              ├→ 策略头(Conv×3) → logits (225维)
                                              └→ 价值头(Conv×3 + GAP + FC×2) → tanh (-1~1)
    
    输入: (B, 3, 15, 15) - 3通道 (己方/对方/上一步标记)
    输出: policy_logits (B, 225), value (B,)
    
    Args:
        num_res_blocks: 残差块数量 (默认4)
        channels: 特征通道数 (默认128)
        board_size: 棋盘大小 (默认15)
    """
    def __init__(self, num_res_blocks: int = 4, channels: int = 128, board_size: int = 15):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.board_squares = board_size * board_size

        # ═══════════════ Stem: 纯卷积（预激活范式下不带BN） ═══════════════
        self.stem_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)

        # ═══════════════ 预激活残差塔 ═══════════════
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])

        # ═══════════════ 尾部归一化 ═══════════════
        # 预激活范式下，最后一层残差的输出没有归一化和激活
        # 必须补上 BN+ReLU，为头部提供干净的归一化特征
        self.final_bn = nn.BatchNorm2d(channels)

        # ═══════════════ 策略头 (Policy Head) ═══════════════
        # 纯净卷积策略头：不含BN，直接输出每个位置的 logit
        self.policy_conv1 = nn.Conv2d(channels, 64, kernel_size=1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(64)
        self.policy_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.policy_bn2 = nn.BatchNorm2d(32)
        self.policy_conv3 = nn.Conv2d(32, 1, kernel_size=1, bias=False)

        # ═══════════════ 价值头 (Value Head) ═══════════════
        # GAP (Global Average Pooling) 价值头，输出 tanh(-1~1)
        self.value_conv1 = nn.Conv2d(channels, 64, kernel_size=1, bias=False)
        self.value_bn1 = nn.BatchNorm2d(64)
        self.value_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.value_bn2 = nn.BatchNorm2d(32)
        self.value_conv3 = nn.Conv2d(32, 32, kernel_size=1, bias=True)

        self.value_fc1 = nn.Linear(32, 32)
        self.value_fc2 = nn.Linear(32, 1)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """
        He 初始化 + BN 初始化。
        
        - Conv2d: Kaiming Normal (fan_out, ReLU)
        - BatchNorm2d: weight=1, bias=0
        - Linear: Kaiming Normal
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, return_value_only: bool = False):
        """
        前向传播。
        
        Args:
            x: 输入张量 (B, 3, 15, 15)
            return_value_only: 仅返回价值（用于快速评估，跳过策略头计算）
            
        Returns:
            (policy_logits, value): 策略logits (B, 225) 和价值 (B,)
            若 return_value_only=True, policy_logits 为 None
        """
        # ── Stem: 纯卷积，不带BN/ReLU ──
        out = self.stem_conv(x)                # (B, C, 15, 15)

        # ── 残差塔 ──
        for block in self.res_blocks:
            out = block(out)                   # 每个block自带预激活BN

        # ── 尾部归一化 ──
        out = F.relu(self.final_bn(out))       # BN→ReLU，为头部提供干净输入

        # ═══════════ 价值头 ═══════════
        v = F.relu(self.value_bn1(self.value_conv1(out)))
        v = F.relu(self.value_bn2(self.value_conv2(v)))
        v = self.value_conv3(v)                                        # (B, 32, 15, 15)
        v = F.adaptive_avg_pool2d(v, (1, 1)).view(v.size(0), -1)      # (B, 32) GAP
        v = F.relu(self.value_fc1(v))                                  # (B, 32)
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)              # (B,) tanh → [-1, 1]

        # 仅返回价值（用于快速评估时跳过策略头）
        if return_value_only:
            return None, value

        # ═══════════ 策略头 ═══════════
        p = F.relu(self.policy_bn1(self.policy_conv1(out)))
        p = F.relu(self.policy_bn2(self.policy_conv2(p)))
        p = self.policy_conv3(p)                                       # (B, 1, 15, 15)

        policy_logits = p.view(p.size(0), -1)                          # (B, 225)

        # 非法落子掩码: 已经被占用的位置设为 -inf
        occupied = (x[:, 0, :, :] + x[:, 1, :, :]).view(x.size(0), -1) > 0
        policy_logits = policy_logits.masked_fill(occupied, -float('inf'))

        return policy_logits, value

    @property
    def arch_type(self) -> str:
        """返回架构标识，用于 AZAgent 自动推断网络类型。"""
        return "cnn"

    def get_config(self) -> dict:
        """返回网络配置字典，用于存档和恢复。"""
        return {
            'arch_type': 'cnn',
            'num_res_blocks': len(self.res_blocks),
            'channels': self.channels,
            'board_size': self.board_size,
        }
