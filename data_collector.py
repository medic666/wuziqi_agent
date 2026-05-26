# data_collector.py
"""
五子棋对弈数据收集模块 (v9.2)

v9.2 改动 (配置重构 — 参数逻辑理清):
  1. 所有可配置参数集中到顶部配置区，醒目分类标注
  2. 明确优先级: 命令行参数 > 配置区默认值 (CLI 始终优先)
  3. 新增配置项: SEARCH_DEPTH, VCT_DEPTH, MAX_CANDIDATES,
     USE_QUIESCENCE, QUIESCENCE_DEPTH (原硬编码于 worker_init 中)
  4. argparse 所有 default=None，由 main() 统一回退到配置区
  5. 布尔开关支持 --flag / --no-flag 双向覆盖

v9.1 改动 (修复退出卡死 & TT丢失):
  1. 修复退出卡死: 达标后立即 pool.terminate() 强杀子进程
  2. 修复TT丢失: TT保存间隔降至10局
  3. 修复默认参数: SAVE_IMAGES 默认设为 False
  4. 缩小提交buffer: 从 5% 降至 3倍workers
"""

import numpy as np
import multiprocessing as mp
import time
import os
import hashlib
import json
import argparse

from utils import transform_2d, transform_state, save_board_image


# ╔════════════════════════════════════════════════════════════════════╗
# ║                                                                    ║
# ║              ★★★  默 认 配 置  ★★★                               ║
# ║                                                                    ║
# ║   ▸ 在此修改所有默认参数，保存后直接运行即可生效                     ║
# ║   ▸ 命令行传参会覆盖此处的值                                       ║
# ║   ▸ 优先级: 命令行参数 > 本配置区 > 模块内硬编码                    ║
# ║                                                                    ║
# ╚════════════════════════════════════════════════════════════════════╝

# ─────────────── 对弈搜索参数 ───────────────
TARGET_GAMES       = 10000    # 目标对局数
SEARCH_DEPTH       = 4        # Alpha-Beta 搜索深度 (降低→提速, 升高→提质量)
VCT_DEPTH          = 8        # VCT 连续冲四搜索深度 (降低→显著提速)
MAX_CANDIDATES     = 10       # 每步候选着法数
USE_QUIESCENCE     = True     # 是否启用静态搜索
QUIESCENCE_DEPTH   = 2        # 静态搜索深度

# ─────────────── 并行参数 ───────────────
NUM_WORKERS        = 10       # 工作进程数

# ─────────────── 保存间隔 ───────────────
CHECKPOINT_INTERVAL   = 20   # 每N局保存一次断点 (批次数据 + 元信息)
SAVE_TT_INTERVAL      = 10   # 每N局保存一次置换表 (防强杀丢失)
SAVE_IMAGE_INTERVAL   = 100  # 每N局保存一张棋谱图片

# ─────────────── 开关 ───────────────
SAVE_IMAGES        = False   # 是否保存棋谱图片 (关闭可提升收集速度)

# ─────────────── 路径配置 ───────────────
DATA_DIR           = "collected_data"
# 以下路径基于 DATA_DIR 自动生成，通常无需修改
IMAGE_DIR          = os.path.join(DATA_DIR, "game_images")
BATCH_DIR          = os.path.join(DATA_DIR, "batches")
TT_DIR             = os.path.join(DATA_DIR, "trans_tables")
META_FILE          = os.path.join(DATA_DIR, "meta.json")
FINAL_FILE         = os.path.join(DATA_DIR, "training_data.npz")


# ======================== 命令行参数 ========================
def parse_args():
    """
    命令行参数解析。
    所有可覆盖参数的 default=None，表示"未指定时使用配置区默认值"。
    优先级: 命令行参数 > 配置区默认值
    """
    parser = argparse.ArgumentParser(
        description='五子棋对弈数据收集 v9.2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
优先级: 命令行参数 > 文件顶部配置区默认值

示例:
  python data_collector.py --target 5000 --depth 3 --no-save-images
  python data_collector.py --resume --workers 20 --vct-depth 6
        """)

    # ── 功能开关 ──
    parser.add_argument('--resume', action='store_true',
                        help='从断点恢复收集（默认从头开始）')

    # ── 数值参数 (未指定 → None → main() 回退到配置区) ──
    parser.add_argument('--target', type=int, default=None,
                        help=f'目标对局数 (默认: {TARGET_GAMES})')
    parser.add_argument('--workers', type=int, default=None,
                        help=f'工作进程数 (默认: {NUM_WORKERS})')
    parser.add_argument('--depth', type=int, default=None,
                        help=f'搜索深度 (默认: {SEARCH_DEPTH})')
    parser.add_argument('--vct-depth', type=int, default=None,
                        help=f'VCT搜索深度 (默认: {VCT_DEPTH})')

    # ── 布尔开关 (支持 --flag / --no-flag 双向覆盖) ──
    parser.add_argument('--save-images', action='store_true', default=None,
                        dest='save_images',
                        help=f'保存棋谱图片 (默认: {SAVE_IMAGES})')
    parser.add_argument('--no-save-images', action='store_false', default=None,
                        dest='save_images',
                        help=f'不保存棋谱图片 (默认: {SAVE_IMAGES})')

    return parser.parse_args()


# ======================== 工作进程：初始化 ========================
def worker_init(counter, data_dir, depth, vct_depth):
    """
    工作进程初始化。

    参数优先级链:
      depth, vct_depth  ← 来自 CLI 覆盖后的值 (通过 initargs 传入)
      MAX_CANDIDATES 等  ← 模块级配置区 (spawn 模式下 worker 重新导入模块，直接读取)
    """
    global _worker_id, _data_dir
    global _agent1, _agent2, _rules_cls, _gamestate_cls
    global _local_game_count

    _data_dir = data_dir
    _local_game_count = 0

    with counter.get_lock():
        _worker_id = counter.value
        counter.value += 1

    from agent_ad import Agent as ADAgent
    from gamerules import GameState, GomokuRules

    # ★ depth/vct_depth: 来自 CLI 覆盖后的值 (通过 initargs 传入)
    # ★ MAX_CANDIDATES/USE_QUIESCENCE/QUIESCENCE_DEPTH: 来自配置区模块全局变量
    _agent1 = ADAgent(depth=depth, max_candidates=MAX_CANDIDATES,
                      use_quiescence=USE_QUIESCENCE,
                      quiescence_depth=QUIESCENCE_DEPTH,
                      vct_depth=vct_depth, name="ADAgent1")
    _agent2 = ADAgent(depth=depth, max_candidates=MAX_CANDIDATES,
                      use_quiescence=USE_QUIESCENCE,
                      quiescence_depth=QUIESCENCE_DEPTH,
                      vct_depth=vct_depth, name="ADAgent2")
    _rules_cls = GomokuRules
    _gamestate_cls = GameState

    _zobrist_fingerprint = _compute_zobrist_fingerprint(_agent1)

    os.makedirs(os.path.join(_data_dir, "trans_tables"), exist_ok=True)
    tt_path = os.path.join(_data_dir, "trans_tables", f"tt_worker_{_worker_id}.pkl")
    loaded = False

    if os.path.exists(tt_path):
        loaded = _load_trans_table(tt_path, _zobrist_fingerprint)

    if not loaded and _worker_id > 0:
        tt_path_0 = os.path.join(_data_dir, "trans_tables", "tt_worker_0.pkl")
        if os.path.exists(tt_path_0):
            _load_trans_table(tt_path_0, _zobrist_fingerprint, source="Worker_0继承")


def _compute_zobrist_fingerprint(agent):
    import struct
    data = b''
    for row in agent.ZOBRIST_TABLE:
        for col in row:
            for val in col:
                data += struct.pack('Q', val)
    return hashlib.md5(data).hexdigest()


def _load_trans_table(tt_path, expected_fingerprint, source=None):
    global _agent1, _agent2
    import pickle
    try:
        with open(tt_path, 'rb') as f:
            tt_data = pickle.load(f)

        saved_fp = tt_data.get('zobrist_fingerprint', '')
        if saved_fp and saved_fp != expected_fingerprint:
            src = source or f"Worker_{_worker_id}"
            print(f"  [{src}] ⚠ ZOBRIST指纹不匹配，跳过加载")
            return False

        for key, value in tt_data.get('black', {}).items():
            if key not in _agent1.trans_table or _agent1.trans_table[key][0] <= value[0]:
                _agent1.trans_table[key] = tuple(value)

        for key, value in tt_data.get('white', {}).items():
            if key not in _agent2.trans_table or _agent2.trans_table[key][0] <= value[0]:
                _agent2.trans_table[key] = tuple(value)

        src = source or f"Worker_{_worker_id}"
        print(f"  [{src}] ✓ 置换表已加载: {_agent1.name}{len(_agent1.trans_table)} {_agent2.name}{len(_agent2.trans_table)}")
        return True
    except Exception as e:
        src = source or f"Worker_{_worker_id}"
        print(f"  [{src}] 置换表加载失败: {e}")
        return False


def _save_my_trans_table():
    global _worker_id, _data_dir, _agent1, _agent2
    import pickle

    os.makedirs(os.path.join(_data_dir, "trans_tables"), exist_ok=True)
    tt_path = os.path.join(_data_dir, "trans_tables", f"tt_worker_{_worker_id}.pkl")

    tt_data = {
        'zobrist_fingerprint': _compute_zobrist_fingerprint(_agent1),
        'black': dict(_agent1.trans_table),
        'white': dict(_agent2.trans_table),
    }

    tmp_path = tt_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(tt_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, tt_path)


# ======================== 工作进程：单局对弈 ========================
def play_single_game(game_id):
    global _local_game_count
    try:
        if game_id % 2 == 0:
            black_agent, white_agent = _agent1, _agent2
        else:
            black_agent, white_agent = _agent2, _agent1

        _agent1._chosen_opening = None; _agent1._opening_step = 0
        _agent2._chosen_opening = None; _agent2._opening_step = 0
        _agent1.reset_incremental_cache()
        _agent2.reset_incremental_cache()

        state = _gamestate_cls(board=bytearray(225), current_player=1, history=[], last_move=None)
        steps_raw = []

        while True:
            current_player = state.current_player
            agent = black_agent if current_player == 1 else white_agent

            board_snapshot = bytes(state.board)
            last_move_before = state.last_move
            history_len_before = len(state.history)

            move = agent.get_move(state)

            if not (0 <= move[0] < 15 and 0 <= move[1] < 15 and state.board[move[0] * 15 + move[1]] == 0):
                winner = 3 - current_player
                total_steps = len(state.history)
                break

            steps_raw.append({
                'board_snapshot': board_snapshot,
                'current_player': current_player,
                'last_move': last_move_before,
                'move': move,
                'step_num': history_len_before,
            })

            state.board[move[0] * 15 + move[1]] = current_player
            state.history.append(move)
            state.last_move = move
            state.current_player = 3 - current_player

            winner = _rules_cls.check_winner(state)
            if winner is not None:
                total_steps = len(state.history)
                break

        data_list = []
        if winner != 0:
            for step in steps_raw:
                board_flat = np.frombuffer(step['board_snapshot'], dtype=np.int8).reshape(15, 15)
                board_np = np.zeros((3, 15, 15), dtype=np.float32)
                board_np[0] = (board_flat == step['current_player'])
                board_np[1] = (board_flat == 3 - step['current_player'])
                if step['last_move'] is not None:
                    board_np[2, step['last_move'][0], step['last_move'][1]] = 1.0

                target_action = np.zeros((15, 15), dtype=np.float32)
                target_action[step['move'][0], step['move'][1]] = 1.0

                target_result = 1 if winner == step['current_player'] else -1

                data_list.append({
                    'state': board_np,
                    'target_action': target_action,
                    'target_result': target_result,
                    'curr_step': step['step_num'],
                    'total_step': total_steps,
                })

        _local_game_count += 1
        # SAVE_TT_INTERVAL 来自配置区模块全局变量 (spawn 模式下 worker 重新导入模块)
        if _local_game_count % SAVE_TT_INTERVAL == 0:
            try:
                _save_my_trans_table()
            except Exception:
                pass

        return game_id, {
            'data': data_list,
            'total_steps': total_steps,
            'winner': winner,
            'history': list(state.history),
            'agent1_tt_size': len(_agent1.trans_table),
            'agent2_tt_size': len(_agent2.trans_table),
            'black_side_name': black_agent.name,
            'white_side_name': white_agent.name,
        }, None

    except Exception as e:
        import traceback
        return game_id, None, f"Game {game_id} 异常: {e}\n{traceback.format_exc()}"


# ======================== 断点机制 ========================
def load_checkpoint():
    completed = 0
    total_samples = 0

    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            meta = json.load(f)
        completed = meta.get('completed_games', 0)
        total_samples = meta.get('total_samples', 0)

    if completed > 0:
        print(f"✓ 断点恢复: 已完成 {completed} 局, {total_samples} 条样本")

    return completed, total_samples


def save_batch(batch_id, states_list, actions_list, results_list, curr_steps_list, total_steps_list):
    if not states_list:
        return
    os.makedirs(BATCH_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(BATCH_DIR, f"batch_{batch_id:06d}.npz"),
                        states=np.array(states_list, dtype=np.float32),
                        actions=np.array(actions_list, dtype=np.float32),
                        results=np.array(results_list, dtype=np.int8),
                        curr_steps=np.array(curr_steps_list, dtype=np.int16),
                        total_steps=np.array(total_steps_list, dtype=np.int16))


def save_meta(completed, total_samples):
    tmp = META_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump({'completed_games': completed, 'total_samples': total_samples}, f)
    if os.path.exists(META_FILE):
        os.replace(tmp, META_FILE)
    else:
        os.rename(tmp, META_FILE)


# ======================== 胜率格式化工具 ========================
def _format_win_rates(agent_wins, draws, total_games):
    parts = []
    for name in sorted(agent_wins.keys()):
        w = agent_wins[name]
        rate = w / total_games * 100 if total_games > 0 else 0
        parts.append(f"{name} {w}胜({rate:.1f}%)")
    parts.append(f"平{draws}")
    return ' │ '.join(parts)


# ======================== 主流程 ========================
def main():
    args = parse_args()

    # ──────────────────────────────────────────────────────────────
    #  参数解析 — 统一优先级: 命令行参数 > 配置区默认值
    #  规则: CLI 传了非 None 值就用 CLI，否则回退到配置区
    # ──────────────────────────────────────────────────────────────
    target_games = args.target        if args.target is not None        else TARGET_GAMES
    num_workers  = args.workers       if args.workers is not None       else NUM_WORKERS
    depth        = args.depth         if args.depth is not None         else SEARCH_DEPTH
    vct_depth    = args.vct_depth     if args.vct_depth is not None     else VCT_DEPTH
    save_images  = args.save_images   if args.save_images is not None   else SAVE_IMAGES
    # 以下参数仅配置区可设，不支持 CLI 覆盖 (需修改文件顶部配置区)
    max_candidates     = MAX_CANDIDATES
    use_quiescence     = USE_QUIESCENCE
    quiescence_depth   = QUIESCENCE_DEPTH
    checkpoint_interval = CHECKPOINT_INTERVAL
    save_tt_interval    = SAVE_TT_INTERVAL
    save_image_interval = SAVE_IMAGE_INTERVAL

    os.makedirs(DATA_DIR, exist_ok=True)
    if save_images:
        os.makedirs(IMAGE_DIR, exist_ok=True)

    # ========== 仅 --resume 时从断点恢复，否则从头开始 ==========
    if args.resume:
        completed_games, total_samples = load_checkpoint()
    else:
        completed_games, total_samples = 0, 0
        print("✓ 全新开始（未使用 --resume）")

    existing_images = (len([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])
                       if save_images and os.path.exists(IMAGE_DIR) else 0)
    image_counter = existing_images
    existing_batches = (len([f for f in os.listdir(BATCH_DIR)
                             if f.startswith("batch_") and f.endswith(".npz")])
                        if os.path.exists(BATCH_DIR) else 0)
    batch_counter = existing_batches

    remaining = target_games - completed_games
    if remaining <= 0:
        _merge_and_save(completed_games=completed_games, total_samples=total_samples)
        return

    # ── 打印配置摘要，标明每个值的来源 ──
    def _src(cli_val, config_name):
        """判断值来自 CLI 还是配置区"""
        return "CLI" if cli_val is not None else "配置区"

    print("=" * 70)
    print("  五子棋数据收集 v9.2 — 配置重构 + 平局剔除 + 修复退出卡死")
    print("=" * 70)
    print(f"  目标对局:       {target_games:>6}   ({_src(args.target, 'TARGET_GAMES')})")
    print(f"  工作进程:       {num_workers:>6}   ({_src(args.workers, 'NUM_WORKERS')})")
    print(f"  搜索深度:       {depth:>6}   ({_src(args.depth, 'SEARCH_DEPTH')})")
    print(f"  VCT深度:        {vct_depth:>6}   ({_src(args.vct_depth, 'VCT_DEPTH')})")
    print(f"  候选着法:       {max_candidates:>6}   (配置区)")
    print(f"  静态搜索:    {'开启':>6}   (配置区: use_quiescence={use_quiescence}, depth={quiescence_depth})")
    print(f"  保存图片:    {'开启' if save_images else '关闭':>6}   ({_src(args.save_images, 'SAVE_IMAGES')})")
    print("-" * 70)
    print(f"  已完成: {completed_games}   剩余: {remaining}")
    print(f"  平局处理: 剔除（不记录训练数据）")
    print(f"  数据保存: 仅存原始视角(1x)，8向增强由训练阶段动态完成")
    print(f"  Resume: {'开启（从断点恢复）' if args.resume else '关闭（从头开始）'}")
    if save_images:
        print(f"  图片间隔: 每 {save_image_interval} 局")
    print(f"  断点间隔: 每 {checkpoint_interval} 局   TT间隔: 每 {save_tt_interval} 局")
    print("=" * 70)

    start_time = time.time()
    session_completed = 0
    agent_wins = {}
    draws = 0
    errors = 0
    max_tt_agent1 = max_tt_agent2 = 0

    batch_states, batch_actions, batch_results, batch_curr_steps, batch_total_steps = [], [], [], [], []
    worker_id_counter = mp.Value('i', 0)

    # ★ depth/vct_depth 通过 initargs 传入 (支持 CLI 覆盖)
    # ★ MAX_CANDIDATES/USE_QUIESCENCE/QUIESCENCE_DEPTH 由 worker 导入模块时读取配置区
    pool = mp.Pool(processes=num_workers, initializer=worker_init,
                   initargs=(worker_id_counter, DATA_DIR, depth, vct_depth))

    try:
        buffer = max(num_workers * 3, 20)
        total_to_submit = remaining + buffer
        all_game_ids = range(completed_games, completed_games + total_to_submit)

        for result in pool.imap_unordered(play_single_game, all_game_ids):
            game_id, game_data, error = result

            if error is not None:
                errors += 1
                completed_games += 1
                print(f"  [!] {error.strip()}")
                if completed_games >= target_games:
                    break
                continue

            completed_games += 1
            session_completed += 1
            winner = game_data['winner']

            black_side_name = game_data['black_side_name']
            white_side_name = game_data['white_side_name']
            for name in [black_side_name, white_side_name]:
                if name not in agent_wins:
                    agent_wins[name] = 0
            if winner == 1:
                agent_wins[black_side_name] += 1
            elif winner == 2:
                agent_wins[white_side_name] += 1
            else:
                draws += 1

            max_tt_agent1 = max(max_tt_agent1, game_data['agent1_tt_size'])
            max_tt_agent2 = max(max_tt_agent2, game_data['agent2_tt_size'])

            raw_data_count = len(game_data['data'])
            for dp in game_data['data']:
                batch_states.append(dp['state'])
                batch_actions.append(dp['target_action'])
                batch_results.append(dp['target_result'])
                batch_curr_steps.append(dp['curr_step'])
                batch_total_steps.append(dp['total_step'])

            total_samples += raw_data_count

            if save_images and session_completed % save_image_interval == 0:
                save_board_image(IMAGE_DIR, image_counter, game_data['history'], winner)
                image_counter += 1

            elapsed = time.time() - start_time
            rate = session_completed / elapsed if session_completed > 0 else 0
            current_remaining = target_games - completed_games
            eta = current_remaining / rate if rate > 0 else 0

            w_str = '●胜' if winner == 1 else '○胜' if winner == 2 else '平局✗'
            win_rate_info = _format_win_rates(agent_wins, draws, session_completed)

            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"{completed_games}/{target_games} ({completed_games / target_games * 100:5.1f}%) │ "
                  f"{rate:.2f}局/s │ ETA {eta / 60:.1f}m │ "
                  f"{game_data['total_steps']}步 {w_str} │ {win_rate_info} │ "
                  f"有效{total_samples}样本")

            if session_completed % checkpoint_interval == 0:
                batch_counter += 1
                save_batch(batch_counter, batch_states, batch_actions, batch_results,
                           batch_curr_steps, batch_total_steps)
                save_meta(completed_games, total_samples)
                print(f"  → 断点已保存 (批次 {batch_counter}, {len(batch_states)} 样本)")
                batch_states.clear()
                batch_actions.clear()
                batch_results.clear()
                batch_curr_steps.clear()
                batch_total_steps.clear()

            if completed_games >= target_games:
                break

    finally:
        print("\n✓ 目标达成，正在强制终止子进程（不等待多余任务）...")
        pool.terminate()
        pool.join()
        print("✓ 子进程已全部清理")

    if batch_states:
        batch_counter += 1
        save_batch(batch_counter, batch_states, batch_actions, batch_results,
                   batch_curr_steps, batch_total_steps)
        save_meta(completed_games, total_samples)

    _merge_and_save(completed_games=completed_games, total_samples=total_samples,
                    start_time=start_time, session_completed=session_completed,
                    agent_wins=agent_wins, draws=draws, errors=errors,
                    max_tt_agent1=max_tt_agent1, max_tt_agent2=max_tt_agent2)


def _merge_and_save(completed_games=0, total_samples=0, start_time=None, session_completed=0,
                    agent_wins=None, draws=0, errors=0, max_tt_agent1=0, max_tt_agent2=0):
    print("\n正在合并批次数据并打包最终文件...")
    all_states, all_actions, all_results, all_curr_steps, all_total_steps = [], [], [], [], []

    if os.path.exists(BATCH_DIR):
        batch_files = sorted(f for f in os.listdir(BATCH_DIR) if f.startswith("batch_") and f.endswith(".npz"))
        for bf in batch_files:
            data = np.load(os.path.join(BATCH_DIR, bf))
            all_states.append(data['states'])
            all_actions.append(data['actions'])
            all_results.append(data['results'])
            all_curr_steps.append(data['curr_steps'])
            all_total_steps.append(data['total_steps'])

    if not all_states:
        print("没有收集到任何数据！")
        return

    final_states = np.concatenate(all_states, axis=0)
    final_actions = np.concatenate(all_actions, axis=0)
    final_results = np.concatenate(all_results, axis=0)
    final_curr_steps = np.concatenate(all_curr_steps, axis=0)
    final_total_steps = np.concatenate(all_total_steps, axis=0)

    np.savez_compressed(FINAL_FILE, states=final_states, target_actions=final_actions,
                        target_results=final_results, curr_steps=final_curr_steps,
                        total_steps=final_total_steps)

    file_size_mb = os.path.getsize(FINAL_FILE) / (1024 * 1024)
    total_time = time.time() - start_time if start_time else 0
    if agent_wins is None:
        agent_wins = {}
    win_rate_summary = _format_win_rates(agent_wins, draws, session_completed)
    equiv_samples = len(final_states) * 8
    draw_rate = draws / session_completed * 100 if session_completed > 0 else 0

    print("\n" + "=" * 70)
    print("  ✓ 数据收集完成！")
    print("=" * 70)
    print(f"  总用时:         {total_time:.0f}s ({total_time / 60:.1f} 分钟)")
    print(f"  完成对局:       {completed_games}   (错误 {errors})")
    print(f"  平局数:         {draws} ({draw_rate:.1f}%) — 已剔除，不计入训练数据")
    print(f"  保存策略:       仅原始视角 (1x)，平局已剔除")
    print(f"  原始样本数:     {len(final_states)}")
    print(f"  等效增强样本:   {equiv_samples}  (训练时8向增强)")
    print(f"  双方胜率:       {win_rate_summary}")
    print(f"  TT峰值:         Agent1={max_tt_agent1} / Agent2={max_tt_agent2}")
    print(f"  数据文件:       {FINAL_FILE}  ({file_size_mb:.1f} MB)")
    print(f"  states 形状:    {final_states.shape}")
    print(f"  actions 形状:   {final_actions.shape}")
    print(f"  results 形状:   {final_results.shape}")
    print("=" * 70)

    print("\n✓ 临时文件已保留 (batches/trans_tables/meta.json)")
    print("  如需清理，可手动删除: collected_data/batches/ 和 collected_data/trans_tables/")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
