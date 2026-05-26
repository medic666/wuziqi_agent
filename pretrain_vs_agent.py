# pretrain_vs_agent.py
"""
五子棋神经网络预训练模块：与 AgentAD 对弈 (成熟版)v9.3

融合 data_collector / pre_train / az_train 三模块精华：
  ✅ AgentAD 置换表持久化 + Zobrist指纹校验 + Worker0继承 (来自 data_collector)
  ✅ Top-K精度 / MAE / 早停 / Cosine LR+Warmup (来自 pre_train)
  ✅ MCTS+GPU推理服务器 / 两阶段断点续训 / 优势裁剪+HuberLoss (来自 az_train)
  ✅ 修复：MCTS树复用（手动两步root推进）、开局库/增量缓存逐局重置
  ✅ 图片保存：对弈/评估阶段独立开关与间隔配置
"""

import os
import sys
import time
import math
import argparse
import logging
import queue
import hashlib
import pickle
import shutil
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch.multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from gamerules import GameState, GomokuRules
from network import ActorCriticNet
from mcts import MCTS, state_to_tensor, create_local_eval_fn
from inference_server import InferenceServer
from agent_ad import Agent as AgentAD
from utils import transform_2d, transform_state, save_board_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

BOARD_SIZE = GomokuRules.BOARD_SIZE
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE


# ═══════════════════════ 配置 ═══════════════════════

@dataclass
class PretrainConfig:
    # 网络
    num_res_blocks: int = 4
    channels: int = 128
    board_size: int = 15

    # 预训练轮次
    num_iterations: int = 15
    games_per_iteration: int = 200

    # MCTS
    num_sims: int = 200
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.2
    dirichlet_epsilon: float = 0.25
    temp_threshold: int = 20
    candidate_radius: int = 3
    advantage_clip: float = 1.0

    # AgentAD 对手
    agent_depth: int = 2
    agent_max_candidates: int = 8
    agent_use_quiescence: bool = True
    agent_vct_depth: int = 6

    # 训练
    replay_buffer_size: int = 500000
    min_replay_size: int = 2000
    batch_size: int = 128
    train_steps_per_iteration: int = 200
    learning_rate: float = 1e-4
    lr_warmup_iterations: int = 3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    value_loss_delta: float = 0.5
    early_stop_patience: int = 5       # 评估胜率连续N轮不提升则早停
    early_stop_min_delta: float = 0.02  # 胜率提升至少2%才算改善

    # 多进程
    num_workers: int = 8
    max_batch_size: int = 128

    # 存档 & 评估
    checkpoint_dir: str = "checkpoints/pretrain_vs_agent"
    initial_model: Optional[str] = "checkpoints/joint_pretrain/best_model.pt"
    eval_interval: int = 2
    eval_games: int = 10

    # ✅ 图片保存配置
    save_images: bool = True                       # 是否保存对弈阶段的棋谱图片
    save_image_every_n_games: int = 50             # 对弈阶段每N局保存一次图片
    save_eval_images: bool = True                  # 是否保存评估阶段的棋谱图片

    # 置换表
    tt_save_interval: int = 10          # 每N局保存一次TT
    tt_inherit_from_worker0: bool = True

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = cls().__dict__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ═══════════════════════ 回放缓冲区 ═══════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, 3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.policies = np.zeros((capacity, BOARD_SQUARES), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros((capacity, BOARD_SQUARES), dtype=np.float32)
        self.size = 0
        self.cursor = 0

    def __len__(self):
        return self.size

    def add(self, states, policies, values, advantages):
        n = len(states)
        if n == 0: return
        start = self.cursor % self.capacity
        end = start + n
        if end <= self.capacity:
            self.states[start:end] = states
            self.policies[start:end] = policies
            self.values[start:end] = values
            self.advantages[start:end] = advantages
        else:
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
        indices = np.random.randint(0, self.size, size=batch_size)
        return (self.states[indices], self.policies[indices],
                self.values[indices], self.advantages[indices])

    def get_linearized_data(self):
        if self.size == 0: return None, None, None, None
        start = self.cursor % self.capacity
        if start + self.size <= self.capacity:
            return (self.states[start:start+self.size], self.policies[start:start+self.size],
                    self.values[start:start+self.size], self.advantages[start:start+self.size])
        first = self.capacity - start
        s = np.concatenate([self.states[start:], self.states[:self.size-first]], axis=0)
        p = np.concatenate([self.policies[start:], self.policies[:self.size-first]], axis=0)
        v = np.concatenate([self.values[start:], self.values[:self.size-first]], axis=0)
        a = np.concatenate([self.advantages[start:], self.advantages[:self.size-first]], axis=0)
        return s, p, v, a

    def restore_from_linearized(self, states, policies, values, cursor, advantages=None):
        n = len(states)
        if n > self.capacity: raise ValueError("数据超容量")
        self.states[:n] = states
        self.policies[:n] = policies
        self.values[:n] = values
        if advantages is not None and len(advantages) == n:
            self.advantages[:n] = advantages
        else:
            self.advantages[:n] = 1.0
        self.size = n
        self.cursor = cursor % self.capacity


# ═══════════════════════ 置换表持久化 ═══════════════════════

def _compute_zobrist_fingerprint(agent):
    """计算 Zobrist 表的 MD5 指纹，防止加载不匹配的置换表"""
    import struct
    data = b''
    for row in agent.ZOBRIST_TABLE:
        for col in row:
            for val in col:
                data += struct.pack('Q', val)
    return hashlib.md5(data).hexdigest()


def _save_trans_table(agent, tt_path):
    """保存 AgentAD 的置换表到磁盘"""
    tt_data = {
        'zobrist_fingerprint': _compute_zobrist_fingerprint(agent),
        'trans_table': dict(agent.trans_table),
    }
    tmp_path = tt_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(tt_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, tt_path)


def _load_trans_table(agent, tt_path, source=None):
    """加载置换表到 AgentAD，指纹不匹配则跳过"""
    if not os.path.exists(tt_path): return False
    try:
        with open(tt_path, 'rb') as f:
            tt_data = pickle.load(f)
        saved_fp = tt_data.get('zobrist_fingerprint', '')
        if saved_fp and saved_fp != _compute_zobrist_fingerprint(agent):
            src = source or "Worker"
            logger.warning(f"  [{src}] ⚠ Zobrist指纹不匹配，跳过TT加载")
            return False
        loaded = 0
        for key, value in tt_data.get('trans_table', {}).items():
            if key not in agent.trans_table or agent.trans_table[key][0] <= value[0]:
                agent.trans_table[key] = tuple(value)
                loaded += 1
        src = source or "Worker"
        logger.info(f"  [{src}] ✓ 置换表已加载: {loaded} 条目 (总{len(agent.trans_table)})")
        return True
    except Exception as e:
        src = source or "Worker"
        logger.warning(f"  [{src}] 置换表加载失败: {e}")
        return False


# ═══════════════════════ Worker ═══════════════════════

def worker_loop_vs_agent(
    worker_id, request_queue, result_queue, task_queue, output_queue,
    temp_threshold, num_sims, c_puct, dirichlet_alpha, dirichlet_epsilon,
    candidate_radius, advantage_clip,
    agent_depth, agent_max_candidates, agent_use_quiescence, agent_vct_depth,
    tt_dir, tt_save_interval, tt_inherit_from_worker0
):
    """子进程：网络 (MCTS+GPU) vs AgentAD，仅收集网络走子样本"""

    # ---- 初始化 AgentAD ----
    agent = AgentAD(
        depth=agent_depth, max_candidates=agent_max_candidates,
        use_quiescence=agent_use_quiescence, vct_depth=agent_vct_depth,
        name="AgentAD"
    )

    # ---- 置换表加载 ----
    os.makedirs(tt_dir, exist_ok=True)
    tt_path = os.path.join(tt_dir, f"tt_worker_{worker_id}.pkl")
    loaded = _load_trans_table(agent, tt_path, source=f"W{worker_id}")

    if not loaded and tt_inherit_from_worker0 and worker_id > 0:
        tt_path_0 = os.path.join(tt_dir, "tt_worker_0.pkl")
        if os.path.exists(tt_path_0):
            _load_trans_table(agent, tt_path_0, source=f"W{worker_id}←W0")

    # ---- MCTS 初始化 ----
    def server_eval_fn(state_np):
        request_queue.put((worker_id, state_np))
        try:
            policy, value = result_queue.get(timeout=60)
        except queue.Empty:
            raise RuntimeError("推理服务器超时(60s)")
        if policy is None:
            raise RuntimeError("推理服务器返回异常")
        return policy, value

    mcts = MCTS(
        eval_fn=server_eval_fn, c_puct=c_puct, num_simulations=num_sims,
        dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon,
        candidate_radius=candidate_radius, advantage_clip=advantage_clip,
    )

    consecutive_failures = 0
    MAX_FAILURES = 5
    local_game_count = 0

    while True:
        try:
            task = task_queue.get(timeout=5)
        except queue.Empty:
            break
        if task is None:
            break

        game_idx = task

        try:
            state = GameState(board=bytearray(BOARD_SQUARES), current_player=1,
                              history=[], last_move=None)
            states_list, policies_list, advantages_list = [], [], []
            move_count = 0

            # 网络执黑执白交替
            net_is_black = (game_idx % 2 == 0)
            net_player = 1 if net_is_black else 2

            # ✅ 重置 AgentAD 逐局状态
            agent._chosen_opening = None
            agent._opening_step = 0
            agent.reset_incremental_cache()

            mcts.root = None

            while move_count < BOARD_SQUARES:
                is_net_turn = (state.current_player == net_player)

                if is_net_turn:
                    # ---- 网络走子 ----
                    temperature = 1.0 if move_count < temp_threshold else 1e-3
                    mcts_policy, action, advantages = mcts.search(
                        state, temperature=temperature, last_action=None
                    )
                    states_list.append(state_to_tensor(state))
                    policies_list.append(mcts_policy)
                    advantages_list.append(advantages)

                    # ✅ 手动推进 MCTS root：网络走子 → 对手走子（两步）
                    if mcts.root is not None and action in mcts.root.children:
                        child_after_net = mcts.root.children[action]
                        child_after_net.parent = None
                        mcts.root = child_after_net
                    else:
                        mcts.root = None
                else:
                    # ---- AgentAD 走子 ----
                    action = agent.get_move(state)

                    # ✅ 手动推进 MCTS root：对手走子后（第二步）
                    if mcts.root is not None and action in mcts.root.children:
                        child_after_agent = mcts.root.children[action]
                        child_after_agent.parent = None
                        mcts.root = child_after_agent
                    else:
                        mcts.root = None

                GomokuRules.apply_move_fast(state, action)
                move_count += 1

                winner = GomokuRules.check_winner(state)
                if winner is not None:
                    break

            output_queue.put((states_list, policies_list, winner,
                              list(state.history), advantages_list, net_player))
            consecutive_failures = 0
            local_game_count += 1

            # ✅ 定期保存置换表
            if local_game_count % tt_save_interval == 0:
                try:
                    _save_trans_table(agent, tt_path)
                except Exception:
                    pass

        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Worker {worker_id} 游戏 {game_idx} 出错: {e}")
            import traceback; traceback.print_exc()
            output_queue.put((None, None, None, None, None, None))
            if consecutive_failures >= MAX_FAILURES:
                logger.error(f"Worker {worker_id} 连续失败{MAX_FAILURES}次，退出")
                output_queue.put(("FATAL", worker_id))
                break

    # 退出前保存置换表
    try:
        _save_trans_table(agent, tt_path)
    except Exception:
        pass
    output_queue.put(("DONE", worker_id))


# ═══════════════════════ 并行对弈调度 ═══════════════════════

def agent_play_phase_parallel(model_path, device_str, num_games, temp_threshold,
                              num_workers, max_batch_size, config, iteration_idx):
    logger.info(f"启动 GPU 推理服务器 (模型: {os.path.basename(model_path)})...")
    server = InferenceServer(model_path, device_str, num_workers, max_batch_size)
    server.ready_event.wait()
    logger.info("推理服务器已就绪")

    task_queue = mp.Queue()
    for i in range(num_games):
        task_queue.put(i)
    for _ in range(num_workers):
        task_queue.put(None)

    eval_queues = [server.get_queues(i) for i in range(num_workers)]
    output_queue = mp.Queue()
    tt_dir = os.path.join(config.checkpoint_dir, "trans_tables")

    processes = []
    for i in range(num_workers):
        req_q, res_q = eval_queues[i]
        p = mp.Process(target=worker_loop_vs_agent, args=(
            i, req_q, res_q, task_queue, output_queue,
            temp_threshold, config.num_sims, config.c_puct,
            config.dirichlet_alpha, config.dirichlet_epsilon,
            config.candidate_radius, config.advantage_clip,
            config.agent_depth, config.agent_max_candidates,
            config.agent_use_quiescence, config.agent_vct_depth,
            tt_dir, config.tt_save_interval, config.tt_inherit_from_worker0
        ), daemon=True)
        p.start()
        processes.append(p)

    all_samples = []
    games_completed = 0
    net_wins, agent_wins, draws = 0, 0, 0
    workers_done = 0
    fatal_error = False
    image_dir = os.path.join(config.checkpoint_dir, "pretrain_images", f"iter_{iteration_idx+1:03d}")
    if config.save_images:
        os.makedirs(image_dir, exist_ok=True)

    pbar = tqdm(total=num_games, desc="对弈(vs AgentAD)")

    while workers_done < num_workers:
        try:
            result = output_queue.get(timeout=120)
        except queue.Empty:
            logger.warning("等待游戏结果超时(120s)")
            continue

        if isinstance(result, tuple) and len(result) >= 2:
            if result[0] == "DONE":
                workers_done += 1
                continue
            elif result[0] == "FATAL":
                logger.error(f"Worker {result[1]} 致命错误，提前终止")
                fatal_error = True
                workers_done = num_workers
                continue

        states, policies, winner, history, advantages, net_player = result
        if states is None:
            games_completed += 1
            pbar.update(1)
            continue

        # ✅ 计算价值标签（网络视角）
        for s, p, adv in zip(states, policies, advantages):
            value = 0.0 if winner == 0 else (1.0 if winner == net_player else -1.0)
            all_samples.append((s, p, value, adv))

        # 胜率统计
        if winner == 0: draws += 1
        elif winner == net_player: net_wins += 1
        else: agent_wins += 1

        games_completed += 1
        
        # ✅ 图片保存逻辑
        if config.save_images and games_completed % config.save_image_every_n_games == 0:
            save_board_image(image_dir, games_completed, history, winner)
            
        pbar.update(1)

    pbar.close()
    for p in processes:
        p.join(timeout=5)
        if p.is_alive(): p.terminate()
    server.shutdown()

    total = net_wins + agent_wins + draws
    win_rate = net_wins / total if total > 0 else 0
    logger.info(f"  对弈完毕: {games_completed}局 | 网络{net_wins}胜 / Agent{agent_wins}胜 / 平{draws} | 胜率{win_rate:.1%}")
    logger.info(f"  收集样本: {len(all_samples)} (仅网络走子)")
    return all_samples, win_rate


# ═══════════════════════ 预训练器 ═══════════════════════

class AgentPreTrainer:
    def __init__(self, config: PretrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCriticNet(
            config.num_res_blocks, config.channels, config.board_size
        ).to(self.device)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.value_loss_fn = nn.HuberLoss(delta=config.value_loss_delta)

        # 优化器（区分衰减/不衰减参数）
        self._create_optimizer()

        self.current_iteration = 0
        self.current_phase = 0          # 0=对弈, 1=训练, 2=完成
        self.best_win_rate = 0.0
        self.es_counter = 0
        self.iteration_stats = []
        self._should_stop = False

        loaded = self._load_checkpoint()
        if not loaded:
            self._load_initial_model()

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self._print_header()

    def _create_optimizer(self):
        decay = [p for n, p in self.model.named_parameters() if not ('bn' in n or 'bias' in n)]
        no_decay = [p for n, p in self.model.named_parameters() if 'bn' in n or 'bias' in n]
        self.optimizer = torch.optim.AdamW([
            {'params': decay, 'weight_decay': self.config.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ], lr=self.config.learning_rate)

    def _get_lr(self, iteration: int) -> float:
        lr = self.config.learning_rate
        warmup = self.config.lr_warmup_iterations
        total = self.config.num_iterations
        if iteration < warmup:
            return lr * (iteration + 1) / max(1, warmup)
        progress = (iteration - warmup) / max(1, total - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _load_initial_model(self):
        path = self.config.initial_model
        if path and os.path.exists(path):
            logger.info(f"加载初始模型: {path}")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            sd = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(sd)
            logger.info("✓ 初始模型加载成功")
        else:
            logger.info("未指定初始模型，从随机权重开始")

    # -------------------- 训练 --------------------

    def _train_step(self, states, policies, values, advantages):
        self.model.train()
        B = states.shape[0]

        # ✅ 8向数据增强
        tids = np.random.randint(0, 8, size=B)
        for i in range(B):
            states[i] = transform_state(states[i], tids[i])
            policies[i] = transform_2d(policies[i].reshape(BOARD_SIZE, BOARD_SIZE), tids[i]).reshape(-1)
            advantages[i] = transform_2d(advantages[i].reshape(BOARD_SIZE, BOARD_SIZE), tids[i]).reshape(-1)

        states_t = torch.from_numpy(states).to(self.device)
        policies_t = torch.from_numpy(policies).to(self.device)
        values_t = torch.from_numpy(values).to(self.device)
        advantages_t = torch.from_numpy(advantages).to(self.device)
        advantages_t = torch.clamp(advantages_t, -self.config.advantage_clip, self.config.advantage_clip)

        self.optimizer.zero_grad(set_to_none=True)
        logits, pred_vals = self.model(states_t)
        logits_flat = logits.view(B, -1)

        # 策略损失（带优势加权）
        log_policy = F.log_softmax(logits_flat, dim=1)
        log_policy_safe = torch.where(policies_t > 0, log_policy, torch.zeros_like(log_policy))
        policy_loss = -(policies_t * log_policy_safe * advantages_t).sum(dim=1).mean()

        # 价值损失（Huber）
        value_loss = self.value_loss_fn(pred_vals, values_t)
        loss = policy_loss + value_loss

        loss.backward()
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("⚠️ NaN/Inf 损失，跳过此步")
            self.optimizer.zero_grad(set_to_none=True)
            return None

        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        # ✅ 计算 Top-K 精度和 MAE（来自 pre_train.py）
        with torch.no_grad():
            _, topk_indices = logits_flat.topk(5, dim=1)
            actions_1d = policies_t.argmax(dim=1).unsqueeze(1)
            top1 = (topk_indices[:, :1] == actions_1d).float().mean().item()
            top3 = (topk_indices[:, :3] == actions_1d).any(dim=1).float().mean().item()
            top5 = (topk_indices[:, :5] == actions_1d).any(dim=1).float().mean().item()
            mae = torch.abs(pred_vals - values_t).mean().item()

        return {
            'loss': loss.item(), 'p_loss': policy_loss.item(), 'v_loss': value_loss.item(),
            'top1': top1, 'top3': top3, 'top5': top5, 'mae': mae
        }

    def _train_phase(self, iteration: int):
        if len(self.replay_buffer) < self.config.min_replay_size:
            logger.info(f"  缓冲区不足({len(self.replay_buffer)}), 跳过训练")
            return {}

        new_lr = self._get_lr(iteration)
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr

        buffer_samples = len(self.replay_buffer)
        max_steps = buffer_samples // self.config.batch_size
        steps = min(self.config.train_steps_per_iteration, max_steps)
        steps = max(steps, 20)

        metrics_sum = {}
        valid_steps = 0
        pbar = tqdm(range(steps), desc="  训练", leave=False)
        for _ in pbar:
            s, p, v, a = self.replay_buffer.sample(self.config.batch_size)
            m = self._train_step(s, p, v, a)
            if m is not None:
                for k, val in m.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + val
                valid_steps += 1
                pbar.set_postfix(L=f"{m['loss']:.3f}", T1=f"{m['top1']:.1%}", MAE=f"{m['mae']:.3f}")

        if valid_steps == 0:
            logger.warning("⚠️ 本轮训练全部NaN")
            return {}

        avg = {k: v / valid_steps for k, v in metrics_sum.items()}
        avg['lr'] = new_lr
        logger.info(f"  训练完成: Loss={avg['loss']:.4f} | P={avg['p_loss']:.4f} | V={avg['v_loss']:.4f} | "
                    f"Top1={avg['top1']:.1%} | Top3={avg['top3']:.1%} | MAE={avg['mae']:.3f} | LR={avg['lr']:.2e}")
        return avg

    # -------------------- 评估 --------------------

    def _evaluate_vs_agent(self):
        logger.info(f"[评估] 当前模型 vs AgentAD (depth={self.config.agent_depth})...")
        agent_ad = AgentAD(
            depth=self.config.agent_depth, max_candidates=self.config.agent_max_candidates,
            use_quiescence=self.config.agent_use_quiescence, vct_depth=self.config.agent_vct_depth,
            name="AgentAD_Eval"
        )
        mcts_eval = MCTS(
            eval_fn=create_local_eval_fn(self.model, self.device),
            c_puct=self.config.c_puct, num_simulations=200,
            dirichlet_epsilon=0.0, candidate_radius=self.config.candidate_radius,
            advantage_clip=self.config.advantage_clip,
        )

        az_wins, agent_wins, draws = 0, 0, 0
        num_games = self.config.eval_games
        total_moves = 0

        # ✅ 评估阶段图片保存
        eval_image_dir = os.path.join(self.config.checkpoint_dir, "eval_images")
        if self.config.save_eval_images:
            os.makedirs(eval_image_dir, exist_ok=True)

        for game_idx in range(num_games):
            state = GameState(board=bytearray(BOARD_SQUARES), current_player=1,
                              history=[], last_move=None)
            az_is_black = (game_idx % 2 == 0)

            mcts_eval.root = None
            agent_ad._chosen_opening = None
            agent_ad._opening_step = 0
            agent_ad.reset_incremental_cache()

            while True:
                if (state.current_player == 1) == az_is_black:
                    mcts_eval.root = None
                    _, action, _ = mcts_eval.search(state, temperature=1e-3, last_action=None)
                else:
                    action = agent_ad.get_move(state)

                GomokuRules.apply_move(state, action)
                total_moves += 1
                winner = GomokuRules.check_winner(state)
                if winner is not None:
                    break

            if winner == 0: draws += 1
            elif (winner == 1 and az_is_black) or (winner == 2 and not az_is_black): az_wins += 1
            else: agent_wins += 1

            # ✅ 保存评估图片
            if self.config.save_eval_images:
                save_board_image(eval_image_dir, game_idx + 1, list(state.history), winner)

        win_rate = az_wins / num_games
        avg_moves = total_moves / num_games
        logger.info(f"  评估: 模型{az_wins}胜 / Agent{agent_wins}胜 / 平{draws} | "
                    f"胜率{win_rate:.1%} | 平均{avg_moves:.0f}步")
        return win_rate

    # -------------------- 存档 --------------------

    def _save_checkpoint(self, is_best=False, save_replay=False, phase=0):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        state = {
            'iteration': self.current_iteration,
            'current_phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_win_rate': self.best_win_rate,
            'es_counter': self.es_counter,
            'config': self.config.to_dict(),
            'iteration_stats': self.iteration_stats,
            'replay_buffer_cursor': self.replay_buffer.cursor,
        }
        torch.save(state, os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt'))

        if is_best:
            torch.save({'model_state_dict': self.model.state_dict()},
                       os.path.join(self.config.checkpoint_dir, 'best_model.pt'))
            logger.info(f"  ★ 新最佳模型 → best_model.pt (胜率 {self.best_win_rate:.1%})")

        if save_replay:
            s, p, v, a = self.replay_buffer.get_linearized_data()
            if s is not None:
                np.savez_compressed(
                    os.path.join(self.config.checkpoint_dir, 'replay_buffer.npz'),
                    states=s, policies=p, values=v,
                    cursor=np.array([self.replay_buffer.cursor]), advantages=a
                )

    def _load_checkpoint(self) -> bool:
        path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt')
        if not os.path.exists(path):
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            for s in self.optimizer.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor): s[k] = v.to(self.device)
            self.current_iteration = ckpt.get('iteration', 0)
            self.current_phase = ckpt.get('current_phase', 0)
            self.best_win_rate = ckpt.get('best_win_rate', 0.0)
            self.es_counter = ckpt.get('es_counter', 0)
            self.iteration_stats = ckpt.get('iteration_stats', [])

            # 恢复回放缓冲区
            rb_path = os.path.join(self.config.checkpoint_dir, 'replay_buffer.npz')
            if os.path.exists(rb_path):
                data = np.load(rb_path, allow_pickle=True)
                cursor = int(data['cursor'][0])
                adv = data['advantages'] if 'advantages' in data else None
                self.replay_buffer.restore_from_linearized(
                    data['states'], data['policies'], data['values'], cursor, advantages=adv
                )
                logger.info(f"  回放缓冲区恢复: {len(data['states']):,} 样本")

            phase_msg = "完整迭代" if self.current_phase == 0 else f"阶段{self.current_phase}"
            logger.info(f"  ✓ 恢复至迭代 {self.current_iteration+1} | 阶段: {phase_msg} | "
                       f"最佳胜率: {self.best_win_rate:.1%} | 缓冲区: {self.replay_buffer.size:,}")
            return True
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return False

    # -------------------- 打印 --------------------

    def _print_header(self):
        c = self.config
        total_p = sum(p.numel() for p in self.model.parameters())
        logger.info("=" * 70)
        logger.info("  五子棋预训练: 神经网络 vs AgentAD (成熟版)")
        logger.info("=" * 70)
        logger.info(f"  网络: {c.num_res_blocks} Blocks × {c.channels} Ch | 参数量: {total_p:,}")
        logger.info(f"  对手: AgentAD (depth={c.agent_depth}, cand={c.agent_max_candidates}, "
                   f"vct={c.agent_vct_depth}, quiesce={c.agent_use_quiescence})")
        logger.info(f"  轮次: {c.num_iterations} × {c.games_per_iteration}局/轮 | MCTS: {c.num_sims}次模拟")
        logger.info(f"  LR: {c.learning_rate:.1e} Cosine + {c.lr_warmup_iterations}轮Warmup")
        logger.info(f"  早停: patience={c.early_stop_patience}, min_delta={c.early_stop_min_delta:.0%}")
        logger.info(f"  置换表: 每{c.tt_save_interval}局保存 | Worker0继承: {'ON' if c.tt_inherit_from_worker0 else 'OFF'}")
        # ✅ 图片保存信息打印
        logger.info(f"  对弈图片: {'ON (每'+str(c.save_image_every_n_games)+'局)' if c.save_images else 'OFF'}")
        logger.info(f"  评估图片: {'ON' if c.save_eval_images else 'OFF'}")
        logger.info(f"  评估: 每{c.eval_interval}轮 | {c.eval_games}局")
        logger.info("=" * 70)

    # -------------------- 主循环 --------------------

    def run(self):
        model_path = os.path.join(self.config.checkpoint_dir, 'self_play_model.pt')
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        if not os.path.exists(model_path):
            torch.save({'model_state_dict': self.model.state_dict()}, model_path)

        for iteration in range(self.current_iteration, self.config.num_iterations):
            if self._should_stop: break
            self.current_iteration = iteration
            iter_start = time.time()
            logger.info(f"\n{'='*60}\n  预训练迭代 {iteration+1} / {self.config.num_iterations}\n{'='*60}")

            # ==================== 阶段 1: 对弈 ====================
            if self.current_phase < 1:
                logger.info("[阶段1] 与 AgentAD 对弈...")
                samples, play_win_rate = agent_play_phase_parallel(
                    model_path=model_path, device_str=str(self.device),
                    num_games=self.config.games_per_iteration,
                    temp_threshold=self.config.temp_threshold,
                    num_workers=self.config.num_workers,
                    max_batch_size=self.config.max_batch_size,
                    config=self.config, iteration_idx=iteration,
                )

                if samples:
                    states = np.array([s[0] for s in samples], dtype=np.float32)
                    policies = np.array([s[1] for s in samples], dtype=np.float32)
                    values = np.array([s[2] for s in samples], dtype=np.float32)
                    advantages = np.array([s[3] for s in samples], dtype=np.float32)
                    self.replay_buffer.add(states, policies, values, advantages)
                logger.info(f"  缓冲区: {len(self.replay_buffer):,} 样本")

                self.current_phase = 1
                self._save_checkpoint(phase=1)
            else:
                logger.info("[阶段1] 跳过(已在上次完成)")
                play_win_rate = 0.0

            # ==================== 阶段 2: 训练 ====================
            if self.current_phase < 2:
                logger.info("[阶段2] 网络训练...")
                train_metrics = self._train_phase(iteration)

                # 训练后保存最新模型供下一轮使用
                torch.save({'model_state_dict': self.model.state_dict()}, model_path)

                self.current_phase = 2
                should_save_replay = (iteration + 1) % 3 == 0
                self._save_checkpoint(phase=2, save_replay=should_save_replay)
                logger.info("  ★ 阶段1&2已安全保存")
            else:
                logger.info("[阶段2] 跳过(已在上次完成)")
                train_metrics = {}

            # ==================== 阶段 3: 评估 & 更新 ====================
            is_best = False
            should_eval = (iteration + 1) % self.config.eval_interval == 0 or iteration == 0

            if should_eval:
                eval_win_rate = self._evaluate_vs_agent()
                self.iteration_stats.append({
                    'type': 'eval', 'iteration': iteration,
                    'win_rate': eval_win_rate, 'play_win_rate': play_win_rate,
                })

                if eval_win_rate > self.best_win_rate + self.config.early_stop_min_delta:
                    self.best_win_rate = eval_win_rate
                    self.es_counter = 0
                    is_best = True
                else:
                    self.es_counter += 1
                    logger.info(f"  胜率未提升 (ES {self.es_counter}/{self.config.early_stop_patience})")

                if self.es_counter >= self.config.early_stop_patience:
                    logger.info(f"  ⚠ 早停触发！胜率连续{self.config.early_stop_patience}轮未改善")
                    self.current_phase = 0          # ✅ 早停退出前也要重置
                    self._save_checkpoint(is_best=is_best, phase=0)
                    break
            else:
                is_best = True

            # ✅ 关键修复：迭代结束前重置 current_phase
            self.current_phase = 0
            self._save_checkpoint(is_best=is_best, phase=0)

            iter_time = time.time() - iter_start
            logger.info(f"  迭代总结: {iter_time:.0f}s | 历史最佳胜率: {self.best_win_rate:.1%}")

        # 最终保存
        torch.save({'model_state_dict': self.model.state_dict()},
                   os.path.join(self.config.checkpoint_dir, 'final_model.pt'))
        logger.info(f"\n✓ 预训练完成！最佳胜率: {self.best_win_rate:.1%}")
        logger.info(f"  可用 best_model.pt 作为 az_train.py 的 --initial_model")


# ═══════════════════════ 入口 ═══════════════════════

def main():
    parser = argparse.ArgumentParser(description="五子棋预训练：网络 vs AgentAD")
    parser.add_argument('--initial_model', type=str, default=None, help="初始模型路径")
    parser.add_argument('--agent_depth', type=int, default=None, help="AgentAD搜索深度(2弱/3中/4强)")
    parser.add_argument('--iterations', type=int, default=None, help="预训练轮数")
    parser.add_argument('--games', type=int, default=None, help="每轮对弈局数")
    parser.add_argument('--sims', type=int, default=None, help="MCTS模拟次数")
    parser.add_argument('--workers', type=int, default=None, help="工作进程数")
    # ✅ 新增图片保存控制命令行
    parser.add_argument('--no-save-images', action='store_true', default=False, help="关闭对弈阶段棋谱图片保存")
    parser.add_argument('--save-image-every', type=int, default=None, help="每N局保存一次对弈图片")
    parser.add_argument('--no-save-eval-images', action='store_true', default=False, help="关闭评估阶段棋谱图片保存")
    
    parser.add_argument('--resume', action='store_true', default=False, help="从断点续训")
    args = parser.parse_args()

    config = PretrainConfig()
    if args.initial_model: config.initial_model = args.initial_model
    if args.agent_depth is not None: config.agent_depth = args.agent_depth
    if args.iterations is not None: config.num_iterations = args.iterations
    if args.games is not None: config.games_per_iteration = args.games
    if args.sims is not None: config.num_sims = args.sims
    if args.workers is not None: config.num_workers = args.workers
    # ✅ 图片配置覆盖
    if args.no_save_images: config.save_images = False
    if args.save_image_every is not None: config.save_image_every_n_games = args.save_image_every
    if args.no_save_eval_images: config.save_eval_images = False
    
    if args.resume: config.initial_model = None  # 续训时不覆盖已训练的权重

    trainer = AgentPreTrainer(config)
    trainer.run()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
