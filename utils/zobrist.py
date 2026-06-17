# utils/zobrist.py
"""
Zobrist 哈希工具函数

Zobrist 哈希是一种用于棋盘游戏的高效哈希方法，
通过随机数异或实现增量更新（走子/悔子时只需 O(1) 更新哈希值）。

用于:
  - AgentAD 置换表索引
  - 置换表持久化（跨进程继承）

合并来源: data_collector.py L164-201 + pretrain_vs_agent.py L175-203
"""

import hashlib
import struct
import os
import pickle


def compute_zobrist_fingerprint(zobrist_table):
    """
    计算 Zobrist 表的 MD5 指纹。
    
    用于验证跨进程/跨文件加载的置换表是否基于相同的 Zobrist 表。
    如果 fingerint 不匹配，说明随机种子不同，置换表不应加载。
    
    Args:
        zobrist_table: 三维列表 [15][15][3] 的随机 64 位整数
        
    Returns:
        32 字符 MD5 十六进制字符串
    """
    data = b''
    for row in zobrist_table:
        for col in row:
            for val in col:
                data += struct.pack('Q', val)
    return hashlib.md5(data).hexdigest()


def save_trans_table(agent, tt_path):
    """
    保存置换表到磁盘（原子写入）。
    
    写入临时文件后原子 rename，防止写入过程中进程被强杀导致文件损坏。
    
    Args:
        agent: 包含 trans_table 属性和 ZOBRIST_TABLE 的 Agent 实例
        tt_path: 目标文件路径
    """
    tt_data = {
        'zobrist_fingerprint': compute_zobrist_fingerprint(agent.ZOBRIST_TABLE),
        'trans_table': dict(agent.trans_table),
    }
    tmp_path = tt_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(tt_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, tt_path)


def load_trans_table(agent, tt_path, source=None, logger=None):
    """
    从磁盘加载置换表（带指纹校验）。
    
    只会加载深度 ≥ 本地已有深度的条目（只取更深或同级）。
    
    Args:
        agent: 包含 trans_table 属性和 ZOBRIST_TABLE 的 Agent 实例
        tt_path: 源文件路径
        source: 来源标识（用于日志输出）
        logger: 可选的 logging.Logger 实例
        
    Returns:
        bool: 是否成功加载
    """
    if not os.path.exists(tt_path):
        return False
    
    try:
        with open(tt_path, 'rb') as f:
            tt_data = pickle.load(f)
        
        # 指纹校验
        saved_fp = tt_data.get('zobrist_fingerprint', '')
        current_fp = compute_zobrist_fingerprint(agent.ZOBRIST_TABLE)
        if saved_fp and saved_fp != current_fp:
            src = source or 'W'
            if logger:
                logger.warning(f"  [{src}] ⚠ Zobrist指纹不匹配，跳过TT加载")
            else:
                print(f"  [{src}] ⚠ Zobrist指纹不匹配，跳过TT加载")
            return False
        
        loaded = 0
        for key, value in tt_data.get('trans_table', {}).items():
            if key not in agent.trans_table or agent.trans_table[key][0] <= value[0]:
                agent.trans_table[key] = tuple(value)
                loaded += 1
        
        src = source or 'W'
        if logger:
            logger.info(f"  [{src}] ✓ 置换表已加载: {loaded} 条目 (总{len(agent.trans_table)})")
        else:
            print(f"  [{src}] ✓ 置换表已加载: {loaded} 条目 (总{len(agent.trans_table)})")
        
        return True
    except Exception as e:
        src = source or 'W'
        if logger:
            logger.warning(f"  [{src}] 置换表加载失败: {e}")
        else:
            print(f"  [{src}] 置换表加载失败: {e}")
        return False