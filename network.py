# network.py
"""五子棋架构：预激活残差块 + GAP价值头 + 纯净卷积策略头(v9.2)

预激活范式 (He et al. 2016):
  Stem:     Conv (无BN/ReLU)
  ResBlock: BN → ReLU → Conv → BN → ReLU → Conv → (+x)
  尾部:     BN → ReLU → 送入头部
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """预激活残差块: BN→ReLU→Conv→BN→ReLU→Conv→(+x)"""
    def __init__(self, channels: int = 128):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + x


class ActorCriticNet(nn.Module):
    def __init__(self, num_res_blocks: int = 4, channels: int = 128, board_size: int = 15):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.board_squares = board_size * board_size

        # ======================== Stem ========================
        # 预激活范式: stem 只做卷积，不带 BN/ReLU
        # 第一个 ResBlock 的 bn1 会负责归一化 stem 输出
        self.stem_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)

        # ======================== 预激活残差塔 ========================
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])

        # ======================== 残差塔尾部归一化 ========================
        # 预激活范式: 最后一层残差加法输出无归一化无激活
        # 必须补上 BN+ReLU，为头部提供干净的归一化特征
        self.final_bn = nn.BatchNorm2d(channels)

        # ======================== 策略头 ========================
        self.policy_conv1 = nn.Conv2d(channels, 64, kernel_size=1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(64)
        self.policy_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False)
        self.policy_bn2 = nn.BatchNorm2d(32)
        self.policy_conv3 = nn.Conv2d(32, 1, kernel_size=1, bias=False)

        # ======================== 价值头 ========================
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
        # ── Stem: 纯卷积 ──
        out = self.stem_conv(x)                # (B, C, 15, 15) 无BN/ReLU

        # ── 残差塔 ──
        for block in self.res_blocks:
            out = block(out)                   # 每个block自带预激活BN

        # ── 尾部归一化 ──
        out = F.relu(self.final_bn(out))       # BN→ReLU，为头部提供干净输入

        # ======================== 价值头 ========================
        v = F.relu(self.value_bn1(self.value_conv1(out)))
        v = F.relu(self.value_bn2(self.value_conv2(v)))
        v = self.value_conv3(v)                                        # (B, 32, 15, 15)
        v = F.adaptive_avg_pool2d(v, (1, 1)).view(v.size(0), -1)      # (B, 32) GAP
        v = F.relu(self.value_fc1(v))                                  # (B, 32)
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)              # (B,)

        if return_value_only:
            return None, value

        # ======================== 策略头 ========================
        p = F.relu(self.policy_bn1(self.policy_conv1(out)))
        p = F.relu(self.policy_bn2(self.policy_conv2(p)))
        p = self.policy_conv3(p)                                       # (B, 1, 15, 15)

        policy_logits = p.view(p.size(0), -1)                          # (B, 225)

        # 非法落子掩码
        occupied = (x[:, 0, :, :] + x[:, 1, :, :]).view(x.size(0), -1) > 0
        policy_logits = policy_logits.masked_fill(occupied, -float('inf'))

        return policy_logits, value