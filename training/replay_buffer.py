# training/replay_buffer.py
"""
经验回放缓冲区 (ReplayBuffer)

用于 AlphaZero 训练中存储自对弈产生的 (状态, 策略, 价值, 优势) 元组。
支持循环覆盖、随机采样、线性化存档/恢复。

合并来源: az_train.py L106-180 + pretrain_vs_agent.py L113-171
"""

import numpy as np

# 棋盘常量（与 gamerules 保持一致）
BOARD_SIZE = 15
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE


class ReplayBuffer:
    """
    循环缓冲区，存储自对弈数据。
    
    存储内容:
      - states: (N, 3, 15, 15) float32 输入状态
      - policies: (N, 225) float32 MCTS 目标策略
      - values: (N,) float32 对局结果 (-1/0/+1)
      - advantages: (N, 225) float32 MCTS 优势值
    
    特性:
      - 循环覆盖：容量满后自动覆盖最旧数据
      - 随机采样：sample() 用于训练批次
      - 线性化存取：get_linearized_data() / restore_from_linearized()
        用于存档到磁盘（处理回绕情况）
    """
    
    def __init__(self, capacity: int):
        """
        初始化回放缓冲区。
        
        Args:
            capacity: 最大存储样本数
        """
        self.capacity = capacity
        self.states = np.zeros((capacity, 3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.policies = np.zeros((capacity, BOARD_SQUARES), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros((capacity, BOARD_SQUARES), dtype=np.float32)
        self.size = 0       # 当前有效样本数
        self.cursor = 0     # 下一个写入位置

    def __len__(self):
        """返回当前有效样本数。"""
        return self.size

    def add(self, states, policies, values, advantages):
        """
        批量添加样本到缓冲区。
        
        自动处理循环覆盖：游标超过 capacity 后从开头覆盖。
        
        Args:
            states: (N, 3, 15, 15) 输入状态数组
            policies: (N, 225) 目标策略数组
            values: (N,) 对局结果数组
            advantages: (N, 225) 优势值数组
        """
        n = len(states)
        if n == 0:
            return
        
        start = self.cursor % self.capacity
        end = start + n
        
        if end <= self.capacity:
            # 不跨越边界，直接写入
            self.states[start:end] = states
            self.policies[start:end] = policies
            self.values[start:end] = values
            self.advantages[start:end] = advantages
        else:
            # 跨越边界，分两段写入
            split = self.capacity - start
            self.states[start:] = states[:split]
            self.policies[start:] = policies[:split]
            self.values[start:] = values[:split]
            self.advantages[start:] = advantages[:split]
            rest = n - split
            self.states[:rest] = states[split:]
            self.policies[:rest] = policies[split:]
            self.values[:rest] = values[split:]
            self.advantages[:rest] = advantages[split:]
        
        self.cursor += n
        self.size = min(self.cursor, self.capacity)

    def sample(self, batch_size):
        """
        随机采样一个训练批次。
        
        Args:
            batch_size: 批次大小
            
        Returns:
            (states, policies, values, advantages) 各为 numpy 数组
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return (self.states[indices], self.policies[indices],
                self.values[indices], self.advantages[indices])

    def get_linearized_data(self):
        """
        获取线性化数据（用于存档）。
        
        处理循环缓冲区回绕情况，返回从旧到新排列的连续数据。
        
        Returns:
            (states, policies, values, advantages) 或 (None, None, None, None)
        """
        if self.size == 0:
            return None, None, None, None
        
        # 缓冲区未回绕时，数据从0开始连续
        if self.cursor <= self.capacity:
            return (self.states[:self.size], self.policies[:self.size],
                    self.values[:self.size], self.advantages[:self.size])
        
        # 缓冲区已回绕，需要拼接
        start = self.cursor % self.capacity
        if start == 0:
            # 刚好整除，数据连续
            return (self.states[:self.capacity], self.policies[:self.capacity],
                    self.values[:self.capacity], self.advantages[:self.capacity])
        
        # 跨边界拼接
        first = self.capacity - start
        states = np.concatenate([self.states[start:], self.states[:first]], axis=0)
        policies = np.concatenate([self.policies[start:], self.policies[:first]], axis=0)
        values = np.concatenate([self.values[start:], self.values[:first]], axis=0)
        advantages = np.concatenate([self.advantages[start:], self.advantages[:first]], axis=0)
        return states, policies, values, advantages

    def restore_from_linearized(self, states, policies, values, cursor, advantages=None):
        """
        从线性化数据恢复缓冲区（用于从存档加载）。
        
        Args:
            states: (N, 3, 15, 15) 状态数组
            policies: (N, 225) 策略数组
            values: (N,) 价值数组
            cursor: 写入游标位置
            advantages: (N, 225) 优势数组（可选，默认填充 1.0）
        """
        n = len(states)
        if n > self.capacity:
            raise ValueError(f"数据量 {n} 超出缓冲区容量 {self.capacity}")
        
        self.states[:n] = states
        self.policies[:n] = policies
        self.values[:n] = values
        
        if advantages is not None and len(advantages) == n:
            self.advantages[:n] = advantages
        else:
            self.advantages[:n] = 1.0  # 兼容旧存档（无 advantages 字段）
        
        self.size = n
        self.cursor = cursor