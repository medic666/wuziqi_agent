# az_train.py
"""
五子棋 AlphaZero 训练主循环
(改进版v9.5: 优势裁剪 + HuberLoss + Cosine LR/Warmup + 两阶段安全存档 + 竞技场/基准评估完美树复用 + 【竞技场双模型并发推理】)
"""

import os
import sys
import signal
import time
import math
import argparse
import logging
import queue
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
from utils import transform_2d, transform_state, save_board_image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

BOARD_SIZE = GomokuRules.BOARD_SIZE
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE


class AlphaZeroConfig:
    def __init__(
        self,
        num_res_blocks: int = 4,
        channels: int = 128,
        board_size: int = 15,
        num_iterations: int = 200,
        games_per_iteration: int = 200,
        train_steps_per_iteration: int = 80,
        baseline_eval_games: int = 40,
        arena_games: int = 40,
        num_sims: int = 400,
        c_puct: float = 2.5,
        dirichlet_alpha: float = 0.2,
        dirichlet_epsilon: float = 0.25,
        temp_threshold: int = 8,
        candidate_radius: int = 2,
        advantage_clip: float = 1.0,
        arena_win_threshold: float = 0.6,
        arena_num_sims: int = 400,
        arena_c_puct: float = 2.5,
        arena_dirichlet_alpha: float = 0.2,
        arena_dirichlet_epsilon: float = 0.0,
        arena_temperature: float = 1e-3,
        arena_temp_threshold: int = 8,
        arena_collapse_threshold: float = 0.35,
        arena_save_image_every_n_games: int = 5,
        arena_data_to_buffer: bool = True,
        baseline_num_sims: int = 400,
        baseline_agent_depth: int = 4,
        baseline_agent_max_candidates: int = 10,
        replay_buffer_size: int = 500000,
        min_replay_size: int = 5000,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        lr_warmup_iterations: int = 5,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        value_loss_delta: float = 0.5,
        num_workers: int = 16,
        max_batch_size: int = 128,
        checkpoint_dir: str = "checkpoints/az_train",
        save_interval: int = 1,
        save_replay_interval: int = 1,
        save_image_every_n_games: int = 50,
        device: str = "auto",
        initial_model: Optional[str] = "checkpoints/joint_pretrain/best_model.pt",
        resume: bool = False,
    ):
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = cls().__dict__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


class ReplayBuffer:
    # ... (保持不变) ...
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
        return self.states[indices], self.policies[indices], self.values[indices], self.advantages[indices]

    def get_linearized_data(self):
        if self.size == 0: return None, None, None, None
        start = self.cursor % self.capacity
        if start + self.size <= self.capacity:
            return (self.states[start:start+self.size], self.policies[start:start+self.size], 
                    self.values[start:start+self.size], self.advantages[start:start+self.size])
        else:
            first = self.capacity - start
            states = np.concatenate([self.states[start:], self.states[:self.size-first]], axis=0)
            policies = np.concatenate([self.policies[start:], self.policies[:self.size-first]], axis=0)
            values = np.concatenate([self.values[start:], self.values[:self.size-first]], axis=0)
            advantages = np.concatenate([self.advantages[start:], self.advantages[:self.size-first]], axis=0)
            return states, policies, values, advantages

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


# =====================================================================
# ★ 新增：双模型并发推理服务器
# =====================================================================
class DualInferenceServer:
    def __init__(self, best_model_path: str, new_model_path: str, device_str: str, num_workers: int, max_batch_size: int = 32):
        self.best_model_path = best_model_path
        self.new_model_path = new_model_path
        self.device_str = device_str
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        
        self.request_queue = mp.Queue()
        self.result_queues = [mp.Queue() for _ in range(num_workers)]
        
        self.ready_event = mp.Event()
        self.shutdown_event = mp.Event()
        
        self.process = mp.Process(
            target=DualInferenceServer._server_loop_static, 
            args=(self.best_model_path, self.new_model_path, self.device_str, self.max_batch_size, 
                  self.num_workers, self.request_queue, self.result_queues, 
                  self.shutdown_event, self.ready_event),
            daemon=True
        )
        self.process.start()

    @staticmethod
    def _server_loop_static(best_model_path, new_model_path, device_str, max_batch_size, num_workers,
                            request_queue, result_queues, shutdown_event, ready_event):
        device = torch.device(device_str)
        try:
            # 加载双模型
            best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
            best_sd = best_ckpt.get('model_state_dict', best_ckpt)
            channels = best_sd['stem_conv.weight'].shape[0]
            res_block_indices = [int(k.split('.')[1]) for k in best_sd if k.startswith('res_blocks.')]
            num_blocks = max(res_block_indices) + 1 if res_block_indices else 4
            
            best_model = ActorCriticNet(num_res_blocks=num_blocks, channels=channels).to(device)
            best_model.load_state_dict(best_sd)
            best_model.eval()
            
            new_ckpt = torch.load(new_model_path, map_location=device, weights_only=False)
            new_sd = new_ckpt.get('model_state_dict', new_ckpt)
            new_model = ActorCriticNet(num_res_blocks=num_blocks, channels=channels).to(device)
            new_model.load_state_dict(new_sd)
            new_model.eval()
            
            if device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                
            print(f"[DualInferenceServer] 双模型并发推理启动 (Max Batch: {max_batch_size})")
        except Exception as e:
            print(f"[DualInferenceServer] 模型加载失败: {e}")
            for q in result_queues: q.put((None, None))
        finally:
            ready_event.set()

        while not shutdown_event.is_set():
            batch_data = []
            try:
                item = request_queue.get(timeout=0.1) 
                batch_data.append(item)
                
                while len(batch_data) < max_batch_size:
                    try:
                        item = request_queue.get_nowait()
                        batch_data.append(item)
                    except queue.Empty:
                        break
            except queue.Empty:
                continue

            if not batch_data:
                continue

            try:
                # ★ 核心改进：按 model_id 分组凑批
                # 0: best_model, 1: new_model
                best_items = [d for d in batch_data if d[1] == 0]
                new_items = [d for d in batch_data if d[1] == 1]
                
                # 处理 Best Model 批次
                if best_items:
                    wids_b = [d[0] for d in best_items]
                    states_np_b = np.stack([d[2] for d in best_items], axis=0)
                    states_t_b = torch.from_numpy(states_np_b).to(device)
                    with torch.no_grad():
                        p_t_b, v_t_b = best_model(states_t_b)
                        p_np_b = torch.softmax(p_t_b.view(states_t_b.size(0), -1), dim=1).cpu().numpy()
                        v_np_b = v_t_b.view(-1).cpu().numpy()
                    for i, wid in enumerate(wids_b):
                        result_queues[wid].put((p_np_b[i], v_np_b[i].item()))

                # 处理 New Model 批次
                if new_items:
                    wids_n = [d[0] for d in new_items]
                    states_np_n = np.stack([d[2] for d in new_items], axis=0)
                    states_t_n = torch.from_numpy(states_np_n).to(device)
                    with torch.no_grad():
                        p_t_n, v_t_n = new_model(states_t_n)
                        p_np_n = torch.softmax(p_t_n.view(states_t_n.size(0), -1), dim=1).cpu().numpy()
                        v_np_n = v_t_n.view(-1).cpu().numpy()
                    for i, wid in enumerate(wids_n):
                        result_queues[wid].put((p_np_n[i], v_np_n[i].item()))
                        
            except Exception as e:
                print(f"\n[DualInferenceServer] 推理出错: {e}")
                for d in batch_data:
                    result_queues[d[0]].put((None, None))

    def get_queues(self, worker_id: int):
        return self.request_queue, self.result_queues[worker_id]

    def shutdown(self):
        self.shutdown_event.set()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


# =====================================================================
# Worker 进程
# =====================================================================

def worker_loop(
    worker_id, request_queue, result_queue, task_queue, output_queue,
    temp_threshold, num_sims, c_puct, dirichlet_alpha, dirichlet_epsilon,
    candidate_radius, advantage_clip
):
    """自弈 Worker"""
    def server_eval_fn(state_np):
        request_queue.put((worker_id, state_np))
        try:
            policy, value = result_queue.get(timeout=30)
        except queue.Empty:
            raise RuntimeError("推理服务器超时(30s)未响应")
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
    games_done_by_me = 0

    while True:
        try:
            task = task_queue.get(timeout=5)
        except queue.Empty:
            break
        if task is None:
            break

        game_idx = task
        try:
            state = GameState(board=bytearray(BOARD_SQUARES), current_player=1, history=[], last_move=None)
            states_list, policies_list, advantages_list, move_count = [], [], [], 0
            mcts.root = None
            last_action = None 
            
            while move_count < BOARD_SQUARES:
                temperature = 1.0 if move_count < temp_threshold else 1e-3
                mcts_policy, action, advantages = mcts.search(state, temperature=temperature, last_action=last_action)
                states_list.append(state_to_tensor(state))
                policies_list.append(mcts_policy)
                advantages_list.append(advantages)
                GomokuRules.apply_move_fast(state, action)
                move_count += 1
                last_action = action
                winner = GomokuRules.check_winner(state)
                if winner is not None:
                    break

            output_queue.put((states_list, policies_list, winner, list(state.history), advantages_list))
            consecutive_failures = 0
            games_done_by_me += 1
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"Worker {worker_id} 游戏 {game_idx} 出错: {e}")
            output_queue.put((None, None, None, None, None))
            if consecutive_failures >= MAX_FAILURES:
                output_queue.put(("FATAL", worker_id))
                break

    output_queue.put(("DONE", worker_id, games_done_by_me))


def arena_worker_loop(
    worker_id, request_queue, result_queue, task_queue, output_queue, config
):
    """★ 竞技场并发 Worker (自带完美树复用)"""
    # 绑定 model_id 的评估函数
    def make_eval_fn(model_id):
        def eval_fn(state_np):
            # 格式: (worker_id, model_id, state_np)
            request_queue.put((worker_id, model_id, state_np))
            try:
                policy, value = result_queue.get(timeout=30)
            except queue.Empty:
                raise RuntimeError("竞技场推理服务器超时")
            if policy is None:
                raise RuntimeError("竞技场推理服务器返回异常")
            return policy, value
        return eval_fn

    new_mcts = MCTS(
        eval_fn=make_eval_fn(1), c_puct=config.arena_c_puct, num_simulations=config.arena_num_sims,
        dirichlet_alpha=config.arena_dirichlet_alpha, dirichlet_epsilon=config.arena_dirichlet_epsilon,
        candidate_radius=config.candidate_radius, advantage_clip=config.advantage_clip,
    )
    best_mcts = MCTS(
        eval_fn=make_eval_fn(0), c_puct=config.arena_c_puct, num_simulations=config.arena_num_sims,
        dirichlet_alpha=config.arena_dirichlet_alpha, dirichlet_epsilon=config.arena_dirichlet_epsilon,
        candidate_radius=config.candidate_radius, advantage_clip=config.advantage_clip,
    )

    while True:
        try:
            task = task_queue.get(timeout=5)
        except queue.Empty:
            break
        if task is None:
            break
            
        game_idx = task
        try:
            state = GameState(board=bytearray(BOARD_SQUARES), current_player=1, history=[], last_move=None)
            new_is_black = (game_idx % 2 == 0)
            
            new_mcts.root = None
            best_mcts.root = None
            last_action_for_new = None
            last_action_for_best = None
            
            game_data = []
            
            while True:
                is_new_turn = (state.current_player == 1) == new_is_black
                current_mcts = new_mcts if is_new_turn else best_mcts
                current_last_action = last_action_for_new if is_new_turn else last_action_for_best

                move_count = len(state.history)
                temperature = 1.0 if move_count < config.arena_temp_threshold else config.arena_temperature
                
                mcts_policy, action, advantages = current_mcts.search(
                    state, temperature=temperature, last_action=current_last_action
                )
                game_data.append((state_to_tensor(state), mcts_policy, advantages, state.current_player))

                # 追踪对手动作
                if is_new_turn:
                    last_action_for_best = action
                else:
                    last_action_for_new = action

                # 手动推进当前MCTS的root，为两步后的复用做准备
                if current_mcts.root is not None and action in current_mcts.root.children:
                    child = current_mcts.root.children[action]
                    child.parent = None
                    current_mcts.root = child
                else:
                    current_mcts.root = None

                GomokuRules.apply_move_fast(state, action)
                winner = GomokuRules.check_winner(state)
                if winner is not None: break
            
            # 格式: (winner, new_is_black, game_data, history)
            output_queue.put((winner, new_is_black, game_data, list(state.history)))
            
        except Exception as e:
            logger.error(f"Arena Worker {worker_id} 游戏 {game_idx} 出错: {e}")
            output_queue.put(None)

    output_queue.put(("DONE", worker_id))


def baseline_eval_worker(
    worker_id, request_queue, result_queue, task_queue, output_queue,
    baseline_num_sims, arena_c_puct, arena_dirichlet_alpha, arena_dirichlet_epsilon,
    candidate_radius, advantage_clip, arena_temperature,
    agent_depth, agent_max_candidates
):
    """基准评估 Worker (单模型树复用)"""
    try:
        from agent_ad import Agent as AgentAD
    except ImportError:
        output_queue.put(("FATAL", worker_id, "ImportError"))
        return

    agent = AgentAD(depth=agent_depth, max_candidates=agent_max_candidates, name="RuleBaseline")
    
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
        eval_fn=server_eval_fn, c_puct=arena_c_puct, num_simulations=baseline_num_sims,
        dirichlet_alpha=arena_dirichlet_alpha, dirichlet_epsilon=arena_dirichlet_epsilon,
        candidate_radius=candidate_radius, advantage_clip=advantage_clip,
    )

    while True:
        try:
            task = task_queue.get(timeout=5)
        except queue.Empty:
            break
        if task is None:
            break
            
        game_idx = task
        try:
            state = GameState(board=bytearray(BOARD_SQUARES), current_player=1, history=[], last_move=None)
            az_is_black = (game_idx % 2 == 0)
            az_player = 1 if az_is_black else 2
            
            if hasattr(agent, '_chosen_opening'): agent._chosen_opening = None
            if hasattr(agent, '_opening_step'): agent._opening_step = 0
            if hasattr(agent, 'reset_incremental_cache'): agent.reset_incremental_cache()
            
            mcts.root = None
            last_az_action = None
            last_opp_action = None
            
            while True:
                is_az_turn = (state.current_player == az_player)
                if is_az_turn:
                    mcts_policy, action, advantages = mcts.search(state, temperature=arena_temperature, last_action=last_opp_action)
                    last_az_action = action
                    if mcts.root is not None and action in mcts.root.children:
                        child = mcts.root.children[action]
                        child.parent = None
                        mcts.root = child
                    else:
                        mcts.root = None
                else:
                    action = agent.get_move(state)
                    last_opp_action = action
                    if mcts.root is not None and action in mcts.root.children:
                        child = mcts.root.children[action]
                        child.parent = None
                        mcts.root = child
                    else:
                        mcts.root = None
                        
                GomokuRules.apply_move_fast(state, action)
                winner = GomokuRules.check_winner(state)
                if winner is not None:
                    break
            
            output_queue.put((winner, az_player))
            
        except Exception as e:
            logger.error(f"Eval Worker {worker_id} 游戏 {game_idx} 出错: {e}")
            output_queue.put(None)

    output_queue.put(("DONE", worker_id))


def self_play_phase_parallel(
    model_path, device_str, num_games, temp_threshold, num_workers,
    max_batch_size, config, iteration_idx
):
    # ... (与之前版本完全一致) ...
    logger.info(f"启动 GPU 推理服务器 (自弈模型: {os.path.basename(model_path)})...")
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

    processes = []
    for i in range(num_workers):
        req_q, res_q = eval_queues[i]
        p = mp.Process(target=worker_loop, args=(
            i, req_q, res_q, task_queue, output_queue, temp_threshold, config.num_sims,
            config.c_puct, config.dirichlet_alpha, config.dirichlet_epsilon,
            config.candidate_radius, config.advantage_clip
        ), daemon=True)
        p.start()
        processes.append(p)

    all_samples = []
    games_completed = 0
    workers_done = 0
    fatal_error = False
    worker_stats = {}
    image_dir = os.path.join(config.checkpoint_dir, "game_images", f"iter_{iteration_idx+1:03d}")
    pbar = tqdm(total=num_games, desc="自弈进度")

    while workers_done < num_workers:
        try:
            result = output_queue.get(timeout=60)
        except queue.Empty:
            logger.warning("等待游戏结果超时(60s)")
            continue

        if isinstance(result, tuple) and len(result) >= 2:
            if result[0] == "DONE":
                wid = result[1]
                wcount = result[2] if len(result) > 2 else -1
                worker_stats[wid] = wcount
                workers_done += 1
                continue
            elif result[0] == "FATAL":
                logger.error(f"收到 Worker {result[1]} 的致命错误，提前终止自弈")
                fatal_error = True
                workers_done = num_workers
                continue

        states, policies, winner, history, advantages = result
        if states is None:
            games_completed += 1
            pbar.update(1)
            continue
        
        current_player = 1
        for s, p, adv in zip(states, policies, advantages):
            value = 0.0 if winner == 0 else (1.0 if winner == current_player else -1.0)
            all_samples.append((s, p, value, adv))
            current_player = 3 - current_player
            
        games_completed += 1
        if games_completed % config.save_image_every_n_games == 0:
            save_board_image(image_dir, games_completed, history, winner)
        pbar.update(1)

    pbar.close()
    for p in processes:
        p.join(timeout=5)
        if p.is_alive(): p.terminate()
    server.shutdown()

    if worker_stats:
        counts = [worker_stats.get(i, 0) for i in range(num_workers)]
        active = [c for c in counts if c >= 0]
        if active:
            logger.info(f"  负载均衡: 最多 {max(active)}局 / 最少 {min(active)}局 / "
                       f"均值 {sum(active)/len(active):.1f}局 (共{num_workers}个Worker)")

    if fatal_error:
        logger.error("自弈因致命错误中断，本迭代数据可能不完整")
        
    logger.info(f"  自弈收集完毕: {games_completed} 局, {len(all_samples)} 样本")
    return all_samples


class AlphaZeroTrainer:
    def __init__(self, config: AlphaZeroConfig):
        self.config = config
        self.device = self._get_device()
        self.current_iteration = 0
        self.current_phase = 0
        self._should_stop = False

        self.best_model = ActorCriticNet(config.num_res_blocks, config.channels, config.board_size).to(self.device)
        self.new_model = ActorCriticNet(config.num_res_blocks, config.channels, config.board_size).to(self.device)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        self.value_loss_fn = nn.HuberLoss(delta=config.value_loss_delta)

        self._create_optimizer()

        loaded = False
        if config.resume:
            loaded = self._load_checkpoint()
            if not loaded: logger.warning("续训失败，将从头开始训练")
        if not loaded and config.initial_model:
            self._load_initial_model(config.initial_model)
            loaded = True
        if not loaded:
            logger.info("从随机初始化开始训练")
            self.new_model.load_state_dict(self.best_model.state_dict())

        self.iteration_stats = []
        self._print_header()

    def _create_optimizer(self):
        decay_params = [p for n, p in self.new_model.named_parameters() if not ('bn' in n or 'bias' in n)]
        no_decay_params = [p for n, p in self.new_model.named_parameters() if 'bn' in n or 'bias' in n]
        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.config.learning_rate)

    def _reset_optimizer(self):
        current_lr = self.optimizer.param_groups[0]['lr']
        self._create_optimizer()
        for pg in self.optimizer.param_groups:
            pg['lr'] = current_lr
        logger.info(f"  权重已回退，优化器已重建（清除错位动量），学习率保持 {current_lr:.2e}")

    def _get_lr(self, iteration: int) -> float:
        lr = self.config.learning_rate
        warmup = self.config.lr_warmup_iterations
        total = self.config.num_iterations
        if iteration < warmup:
            return lr * (iteration + 1) / max(1, warmup)
        progress = (iteration - warmup) / max(1, total - warmup)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _get_device(self):
        cfg = self.config.device.lower()
        if cfg in ("auto", "cuda"): return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(cfg)

    def _load_initial_model(self, path):
        logger.info(f"加载预训练模型: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        self.best_model.load_state_dict(state_dict)
        self.new_model.load_state_dict(state_dict)
        logger.info("✓ 预训练模型加载成功 (best_model + new_model 同步)")

    def _load_checkpoint(self) -> bool:
        # ... (与之前修复版本一致) ...
        path = os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt')
        if not os.path.exists(path):
            logger.info("未找到检查点文件")
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.best_model.load_state_dict(ckpt['best_model_state_dict'])
            self.new_model.load_state_dict(ckpt['new_model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            for s in self.optimizer.state.values():
                for k, v in s.items():
                    if isinstance(v, torch.Tensor): s[k] = v.to(self.device)
            self.current_iteration = ckpt.get('iteration', 0)
            self.current_phase = ckpt.get('current_phase', 0)
            self.iteration_stats = ckpt.get('iteration_stats', [])

            rb_path = os.path.join(self.config.checkpoint_dir, 'replay_buffer.npz')
            if os.path.exists(rb_path):
                data = np.load(rb_path, allow_pickle=True)
                cursor = int(data['cursor'][0])
                adv = data['advantages'] if 'advantages' in data else None
                self.replay_buffer.restore_from_linearized(
                    data['states'], data['policies'], data['values'], cursor, advantages=adv
                )
                logger.info(f"  回放缓冲区恢复: {len(data['states']):,} 样本")

            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            torch.save({'model_state_dict': self.best_model.state_dict()}, best_path)

            phase_msg = "完整迭代" if self.current_phase == 0 else f"阶段{self.current_phase}"
            logger.info(f"  ✓ 恢复至迭代 {self.current_iteration + 1} | 阶段: {phase_msg} | 缓冲区 {self.replay_buffer.size:,} 样本")
            return True
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return False

    def _print_header(self):
        c = self.config
        total_p = sum(p.numel() for p in self.best_model.parameters())
        logger.info("=" * 70)
        logger.info("  五子棋 AlphaZero 训练系统 (双模型并发推理版)")
        logger.info("=" * 70)
        logger.info(f"  网络: {c.num_res_blocks} Blocks × {c.channels} Ch | 参数量: {total_p:,}")
        logger.info(f"  自弈: {c.games_per_iteration}局/轮 × {c.num_sims}次模拟")
        logger.info(f"  竞技场: {c.arena_games}局(★并发) × {c.arena_num_sims}次模拟 | 树复用: ON")
        logger.info("=" * 70)

    def _train_step(self, states, policies, values, advantages):
        # ... (与之前版本一致) ...
        self.new_model.train()
        B = states.shape[0]
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
        logits, pred_vals = self.new_model(states_t)
        logits_flat = logits.view(logits.size(0), -1)
        log_policy = F.log_softmax(logits_flat, dim=1)
        log_policy_safe = torch.where(policies_t > 0, log_policy, torch.zeros_like(log_policy))
        
        policy_loss = -(policies_t * log_policy_safe).sum(dim=1).mean()
        value_loss = self.value_loss_fn(pred_vals, values_t)
        loss = self.config.policy_loss_weight * policy_loss + self.config.value_loss_weight * value_loss
        
        loss.backward()
        if torch.isnan(loss) or torch.isinf(loss):
            self.optimizer.zero_grad(set_to_none=True)
            return float('nan'), float('nan'), float('nan')
        
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.new_model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return loss.item(), policy_loss.item(), value_loss.item()

    def _train_phase(self, iteration: int):
        if len(self.replay_buffer) < self.config.min_replay_size:
            logger.info(f"  缓冲区不足，跳过训练")
            return {}

        new_lr = self._get_lr(iteration)
        for pg in self.optimizer.param_groups:
            pg['lr'] = new_lr

        buffer_samples = len(self.replay_buffer)
        max_steps = buffer_samples // self.config.batch_size // 2
        steps = min(self.config.train_steps_per_iteration, max_steps)
        steps = max(steps, 20)

        total_loss = total_policy = total_value = 0.0
        valid_steps = 0
        pbar = tqdm(range(steps), desc="  训练", leave=False)
        for _ in pbar:
            s, p, v, a = self.replay_buffer.sample(self.config.batch_size)
            loss, p_loss, v_loss = self._train_step(s, p, v, a)
            if not (math.isnan(loss) or math.isinf(loss)):
                total_loss += loss; total_policy += p_loss; total_value += v_loss
                valid_steps += 1
            pbar.set_postfix(L=f"{loss:.4f}", P=f"{p_loss:.4f}", V=f"{v_loss:.4f}")
        
        if valid_steps == 0:
            return {'train_loss': float('nan'), 'train_policy_loss': float('nan'), 
                    'train_value_loss': float('nan'), 'lr': new_lr, 'train_steps': steps, 'valid_steps': 0}
        
        return {
            'train_loss': total_loss / valid_steps,
            'train_policy_loss': total_policy / valid_steps,
            'train_value_loss': total_value / valid_steps,
            'lr': new_lr,
            'train_steps': steps,
            'valid_steps': valid_steps,
        }

    def _arena_phase(self, iteration: int) -> Tuple[bool, list]:
        logger.info(f"  竞技场: 新模型 vs 最佳模型 ({self.config.arena_games} 局 ★并发)")
        
        # ★ 保存 new_model 到磁盘，供 DualInferenceServer 加载
        new_model_path = os.path.join(self.config.checkpoint_dir, 'new_model_arena.pt')
        torch.save({'model_state_dict': self.new_model.state_dict()}, new_model_path)
        best_model_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
        
        # ★ 启动双模型并发推理服务器
        server = DualInferenceServer(
            best_model_path=best_model_path, 
            new_model_path=new_model_path, 
            device_str=str(self.device), 
            num_workers=self.config.num_workers, 
            max_batch_size=self.config.max_batch_size
        )
        server.ready_event.wait()
        logger.info("  双模型推理服务器已就绪")

        task_queue = mp.Queue()
        for i in range(self.config.arena_games):
            task_queue.put(i)
        for _ in range(self.config.num_workers):
            task_queue.put(None)

        eval_queues = [server.get_queues(i) for i in range(self.config.num_workers)]
        output_queue = mp.Queue()

        processes = []
        for i in range(self.config.num_workers):
            req_q, res_q = eval_queues[i]
            p = mp.Process(target=arena_worker_loop, args=(
                i, req_q, res_q, task_queue, output_queue, self.config
            ), daemon=True)
            p.start()
            processes.append(p)

        new_wins, best_wins, draws = 0, 0, 0
        all_arena_samples = []
        games_completed = 0
        workers_done = 0
        
        arena_image_dir = os.path.join(self.config.checkpoint_dir, "arena_images", f"iter_{iteration+1:03d}")
        os.makedirs(arena_image_dir, exist_ok=True)
        
        pbar = tqdm(total=self.config.arena_games, desc="  竞技场", leave=False)
        
        while workers_done < self.config.num_workers:
            try:
                result = output_queue.get(timeout=120)
            except queue.Empty:
                continue

            if isinstance(result, tuple) and len(result) >= 2:
                if result[0] == "DONE":
                    workers_done += 1
                    continue
                elif result[0] == "FATAL":
                    workers_done = self.config.num_workers
                    continue
            
            if result is None:
                games_completed += 1
                pbar.update(1)
                continue
                
            winner, new_is_black, game_data, history = result
            
            # 统计胜负
            if winner == 0: 
                draws += 1
            elif (winner == 1 and new_is_black) or (winner == 2 and not new_is_black): 
                new_wins += 1
            else: 
                best_wins += 1
                
            # 收集训练数据
            for s, p, adv, player in game_data:
                value = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
                all_arena_samples.append((s, p, value, adv))

            games_completed += 1
            if games_completed % self.config.arena_save_image_every_n_games == 0:
                try:
                    save_board_image(arena_image_dir, games_completed, history, winner)
                except Exception:
                    pass
            pbar.update(1)

        pbar.close()
        
        for p in processes:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()
        server.shutdown()
        
        # 清理临时文件
        if os.path.exists(new_model_path):
            os.remove(new_model_path)

        total = new_wins + best_wins + draws
        win_rate = new_wins / total if total > 0 else 0
        logger.info(f"  竞技场结果: 新模型 {new_wins}胜 / 最佳 {best_wins}胜 / 平 {draws} | 胜率 {win_rate:.1%}")
        logger.info(f"  竞技场数据: 收集 {len(all_arena_samples)} 样本")
        
        self.iteration_stats.append({
            'type': 'arena', 'iteration': iteration,
            'new_wins': new_wins, 'best_wins': best_wins, 'draws': draws, 'win_rate': win_rate
        })
        return win_rate >= self.config.arena_win_threshold, all_arena_samples

    def _evaluate_baseline(self, iteration: int):
        # ... (与之前版本一致，使用 InferenceServer 单模型并发) ...
        try:
            from agent_ad import Agent as AgentAD
            logger.info(f"[基准评估] 最佳模型 vs 规则引擎 ({self.config.baseline_eval_games}局, 并行)...")
            best_model_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
            if not os.path.exists(best_model_path):
                logger.error("找不到 best_model.pt，跳过基准评估")
                return

            server = InferenceServer(best_model_path, str(self.device), self.config.num_workers, self.config.max_batch_size)
            server.ready_event.wait()
            
            task_queue = mp.Queue()
            for i in range(self.config.baseline_eval_games):
                task_queue.put(i)
            for _ in range(self.config.num_workers):
                task_queue.put(None)
                
            eval_queues = [server.get_queues(i) for i in range(self.config.num_workers)]
            output_queue = mp.Queue()
            
            processes = []
            for i in range(self.config.num_workers):
                req_q, res_q = eval_queues[i]
                p = mp.Process(target=baseline_eval_worker, args=(
                    i, req_q, res_q, task_queue, output_queue,
                    self.config.baseline_num_sims, self.config.arena_c_puct, 
                    self.config.arena_dirichlet_alpha, self.config.arena_dirichlet_epsilon,
                    self.config.candidate_radius, self.config.advantage_clip, self.config.arena_temperature,
                    self.config.baseline_agent_depth, self.config.baseline_agent_max_candidates
                ), daemon=True)
                p.start()
                processes.append(p)
                
            az_wins, base_wins, draws = 0, 0, 0
            games_completed = 0
            workers_done = 0
            
            pbar = tqdm(total=self.config.baseline_eval_games, desc="基准评估", leave=False)
            while workers_done < self.config.num_workers:
                try:
                    result = output_queue.get(timeout=120)
                except queue.Empty:
                    continue
                if isinstance(result, tuple) and len(result) >= 2:
                    if result[0] == "DONE": workers_done += 1; continue
                    elif result[0] == "FATAL": workers_done = self.config.num_workers; continue
                if result is None: games_completed += 1; pbar.update(1); continue
                winner, az_player = result
                if winner == 0: draws += 1
                elif winner == az_player: az_wins += 1
                else: base_wins += 1
                games_completed += 1
                pbar.update(1)
                
            pbar.close()
            for p in processes:
                p.join(timeout=5)
                if p.is_alive(): p.terminate()
            server.shutdown()
            
            win_rate = az_wins / self.config.baseline_eval_games if self.config.baseline_eval_games > 0 else 0
            logger.info(f"  基准评估结果: AZ {az_wins}胜 / 规则 {base_wins}胜 / 平 {draws} | 胜率 {win_rate:.1%}")
            self.iteration_stats.append({'type': 'baseline', 'iteration': iteration, 'az_wins': az_wins, 'base_wins': base_wins, 'win_rate': win_rate})
        except Exception as e:
            logger.error(f"基准评估执行失败(可忽略): {e}")

    def _save_checkpoint(self, iteration, is_best=False, save_replay=False, phase=0):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        state = {
            'iteration': iteration,
            'best_model_state_dict': self.best_model.state_dict(),
            'new_model_state_dict': self.new_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_cursor': self.replay_buffer.cursor,
            'config': self.config.to_dict(),
            'iteration_stats': self.iteration_stats,
            'current_phase': phase,
        }
        torch.save(state, os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt'))
        torch.save({'model_state_dict': self.best_model.state_dict()}, 
                  os.path.join(self.config.checkpoint_dir, 'best_model.pt'))
        if save_replay: self._save_replay_buffer()

    def _save_replay_buffer(self):
        if self.replay_buffer.size == 0: return
        s, p, v, a = self.replay_buffer.get_linearized_data()
        if s is None: return
        np.savez_compressed(os.path.join(self.config.checkpoint_dir, 'replay_buffer.npz'),
                            states=s, policies=p, values=v,
                            cursor=np.array([self.replay_buffer.cursor]), advantages=a)

    def _signal_handler(self, sig, frame):
        logger.info("\n收到中断信号，正在优雅关闭...")
        self._should_stop = True

    def run(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        best_model_path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        if not os.path.exists(best_model_path):
            torch.save({'model_state_dict': self.best_model.state_dict()}, best_model_path)

        for iteration in range(self.current_iteration, self.config.num_iterations):
            if self._should_stop: break
            self.current_iteration = iteration
            iter_start = time.time()
            logger.info(f"\n{'='*60}\n  迭代 {iteration+1} / {self.config.num_iterations}\n{'='*60}")

            if self.current_phase < 1:
                logger.info("[阶段1] 自我对弈...")
                samples = self_play_phase_parallel(
                    model_path=best_model_path, device_str=str(self.device),
                    num_games=self.config.games_per_iteration, temp_threshold=self.config.temp_threshold,
                    num_workers=self.config.num_workers, max_batch_size=self.config.max_batch_size,
                    config=self.config, iteration_idx=iteration,
                )
                if samples:
                    states = np.array([s[0] for s in samples], dtype=np.float32)
                    policies = np.array([s[1] for s in samples], dtype=np.float32)
                    values = np.array([s[2] for s in samples], dtype=np.float32)
                    advantages = np.array([s[3] for s in samples], dtype=np.float32)
                    self.replay_buffer.add(states, policies, values, advantages)
                logger.info(f"  自弈完成: {len(samples)} 新样本 | 缓冲区 {len(self.replay_buffer):,}")
            else:
                logger.info("[阶段1] 自我对弈... (跳过，已在上次崩溃前完成)")

            if self.current_phase < 2:
                logger.info("[阶段2] 网络训练...")
                train_metrics = self._train_phase(iteration)
                if train_metrics:
                    logger.info(f"  训练完成: Loss={train_metrics['train_loss']:.4f} | "
                               f"LR={train_metrics['lr']:.2e} | "
                               f"Steps={train_metrics['train_steps']}(有效={train_metrics['valid_steps']})")
                self.current_phase = 2
                self._save_checkpoint(iteration, is_best=False, save_replay=True, phase=2)
                logger.info("  ★ 阶段1&2已完成，进度已安全保存！接下来进入耗时的竞技场评估。")
            else:
                logger.info("[阶段2] 网络训练... (跳过，已在上次崩溃前完成)")

            is_best = False
            if self.current_phase < 3:
                if len(self.replay_buffer) >= self.config.min_replay_size:
                    logger.info("[阶段3] 竞技场评估...")
                    is_best, arena_samples = self._arena_phase(iteration)

                    if self.config.arena_data_to_buffer and arena_samples:
                        a_states = np.array([s[0] for s in arena_samples], dtype=np.float32)
                        a_policies = np.array([s[1] for s in arena_samples], dtype=np.float32)
                        a_values = np.array([s[2] for s in arena_samples], dtype=np.float32)
                        a_advantages = np.array([s[3] for s in arena_samples], dtype=np.float32)
                        self.replay_buffer.add(a_states, a_policies, a_values, a_advantages)
                        logger.info(f"  ✓ 竞技场数据已加入缓冲区: +{len(arena_samples)} 样本 | 缓冲区总计 {len(self.replay_buffer):,}")

                    arena_win_rate = 0.0
                    for stat in reversed(self.iteration_stats):
                        if stat.get('type') == 'arena' and stat.get('iteration') == iteration:
                            arena_win_rate = stat.get('win_rate', 0.0)
                            break

                    if is_best:
                        self.best_model.load_state_dict(self.new_model.state_dict())
                        history_dir = os.path.join(self.config.checkpoint_dir, "history_best_models")
                        os.makedirs(history_dir, exist_ok=True)
                        history_best_path = os.path.join(history_dir, f'best_model_iter_{iteration+1}.pt')
                        torch.save({'model_state_dict': self.best_model.state_dict()}, history_best_path)
                        logger.info(f"  ★ 新模型胜出，已更新为最佳模型 (历史快照已保存: {os.path.basename(history_best_path)})")
                    elif arena_win_rate < self.config.arena_collapse_threshold:
                        logger.warning(f"  ⚠️ 竞技场胜率极低({arena_win_rate:.1%} < {self.config.arena_collapse_threshold:.0%})，紧急回退到最佳模型！")
                        self.new_model.load_state_dict(self.best_model.state_dict())
                        self._reset_optimizer()
                    else:
                        logger.info(f"  新模型未胜出(胜率{arena_win_rate:.1%})，保留当前权重继续训练")
                else:
                    logger.info("  初期热身: 保持当前最佳模型, 缓冲区不足")
            else:
                logger.info("[阶段3] 竞技场评估... (跳过，已在上次崩溃前完成)")

            self.current_phase = 0
            self._save_checkpoint(iteration + 1, is_best=is_best, save_replay=True, phase=0)

            if is_best and len(self.replay_buffer) >= self.config.min_replay_size:
                self._evaluate_baseline(iteration)

            iter_time = time.time() - iter_start
            logger.info(f"  迭代总结: {iter_time:.0f}s | 最佳 {'★' if is_best else '✗'}")

        logger.info("\n✓ AlphaZero 训练完成！")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_model', type=str, default=None)
    parser.add_argument('--resume', action='store_true', default=False)
    args = parser.parse_args()

    config = AlphaZeroConfig()
    if args.initial_model: config.initial_model = args.initial_model
    if args.resume: config.resume = True

    trainer = AlphaZeroTrainer(config)
    trainer.run()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()