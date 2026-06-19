# training/inference_server.py
"""
GPU 推理服务器模块

提供两种推理模式:
  1. InferenceServer: 单模型推理服务器（自对弈/基准评估场景）
  2. DualInferenceServer: 双模型并发推理服务器（竞技场场景）

核心设计:
  - 批量推理：多个 worker 的请求自动凑批，提升 GPU 利用率
  - 极速凑批策略：尽量清空队列凑满 batch，比逐条推理快 5-20x
  - 进程隔离：推理在独立进程中运行，与 worker 完全解耦

合并来源: inference_server.py + az_train.py L182-303 (DualInferenceServer)
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import queue
from agents.neural.registry import get_network_class, get_defaults, resolve_arch, infer_arch_from_state_dict


class InferenceServer:
    """
    单模型 GPU 推理服务器。
    
    用于自对弈和基准评估场景，worker 通过 get_queues() 获取通信队列。
    
    通信协议:
      - 请求: (worker_id, state_np) → request_queue
      - 响应: (policy_probs, value) → result_queues[worker_id]
    """
    
    def __init__(self, model_path: str, device_str: str, num_workers: int, max_batch_size: int = 32):
        """
        初始化单模型推理服务器。
        
        Args:
            model_path: 模型权重文件路径
            device_str: 设备字符串（如 "cuda", "cpu"）
            num_workers: Worker 数量
            max_batch_size: 最大批处理大小
        """
        self.model_path = model_path
        self.device_str = device_str
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        
        self.request_queue = mp.Queue()
        self.result_queues = [mp.Queue() for _ in range(num_workers)]
        
        self.ready_event = mp.Event()
        self.shutdown_event = mp.Event()
        
        self.process = mp.Process(
            target=InferenceServer._server_loop_static, 
            args=(self.model_path, self.device_str, self.max_batch_size, 
                  self.num_workers, self.request_queue, self.result_queues, 
                  self.shutdown_event, self.ready_event),
            daemon=True
        )
        self.process.start()

    @staticmethod
    def _server_loop_static(model_path, device_str, max_batch_size, num_workers,
                            request_queue, result_queues, shutdown_event, ready_event):
        """
        推理服务器主循环（静态方法，在子进程中运行）。
        
        核心逻辑:
          1. 加载模型并自动推断架构参数
          2. 阻塞等待第一个请求 → 极速清空队列凑批 → GPU 批量推理 → 分发结果
        """
        device = torch.device(device_str)
        model = None
        try:
            # 加载模型（通过单一真理源 build_model_from_checkpoint）
            from agents.neural.registry import build_model_from_checkpoint
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            model, _, _ = build_model_from_checkpoint(ckpt, device=device)
            
            if device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                
            print(f"[InferenceServer] 极速凑批模式启动 (Max Batch: {max_batch_size})")
        except Exception as e:
            print(f"[InferenceServer] 模型加载失败: {e}")
            for q in result_queues:
                q.put((None, None))
        finally:
            ready_event.set()
        
        if model is None:
            return

        # 主循环：凑批 → 推理 → 分发
        while not shutdown_event.is_set():
            batch_data = []
            try:
                # 阻塞等待第一个请求
                item = request_queue.get(timeout=0.1) 
                batch_data.append(item)
                
                # 极速清空队列：持续尝试直到凑满 max_batch_size 或队列为空
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
                worker_ids = [d[0] for d in batch_data]
                
                # 极致优化：np.stack + torch.from_numpy
                # 先在 CPU 上用 C 级别的速度 stack，然后零拷贝转 Tensor 上传 GPU
                states_np = np.stack([d[1] for d in batch_data], axis=0)
                states_t = torch.from_numpy(states_np).to(device)
                
                with torch.no_grad():
                    policy_logits, values = model(states_t)
                    
                    # GPU 上计算 Softmax，比传回 CPU 用 Numpy 算快一倍
                    policies_t = torch.softmax(policy_logits.view(states_t.size(0), -1), dim=1)
                    
                    # 一次性传回 CPU
                    policies_np = policies_t.cpu().numpy()
                    values_np = values.view(-1).cpu().numpy()

                # 分发结果
                for i, wid in enumerate(worker_ids):
                    self_result = (policies_np[i], values_np[i].item())
                    result_queues[wid].put(self_result)
                    
            except Exception as e:
                print(f"\n[InferenceServer] 推理过程出错: {e}")
                for i, wid in enumerate(worker_ids):
                    result_queues[wid].put((None, None))
        
        print("\n[InferenceServer] 推理服务器已优雅退出")

    def get_queues(self, worker_id: int):
        """获取 worker 的请求/响应队列对。"""
        return self.request_queue, self.result_queues[worker_id]

    def shutdown(self):
        """关闭推理服务器。"""
        self.shutdown_event.set()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()


class DualInferenceServer:
    """
    双模型并发推理服务器（用于竞技场场景）。
    
    同时加载 best_model 和 new_model，根据请求中的 model_id 选择模型推理。
    model_id: 0 = best_model, 1 = new_model
    
    通信协议:
      - 请求: (worker_id, model_id, state_np) → request_queue
      - 响应: (policy_probs, value) → result_queues[worker_id]
    """
    
    def __init__(self, best_model_path: str, new_model_path: str, device_str: str, 
                 num_workers: int, max_batch_size: int = 32):
        self.best_model_path = best_model_path
        self.new_model_path = new_model_path
        self.device_str = device_str
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        
        self.request_queue = mp.Queue()
        self.result_queues = [mp.Queue() for _ in range(num_workers)]
        
        self.ready_event = mp.Event()
        self.shutdown_event = mp.Event()
        self.init_error = mp.Value('b', False)  # 共享标志：初始化是否失败
        
        self.process = mp.Process(
            target=DualInferenceServer._server_loop_static, 
            args=(self.best_model_path, self.new_model_path, self.device_str, self.max_batch_size, 
                  self.num_workers, self.request_queue, self.result_queues, 
                  self.shutdown_event, self.ready_event, self.init_error),
            daemon=True
        )
        self.process.start()

    @staticmethod
    def _server_loop_static(best_model_path, new_model_path, device_str, max_batch_size, num_workers,
                            request_queue, result_queues, shutdown_event, ready_event, init_error):
        """
        双模型服务器主循环（静态方法，在子进程中运行）。
        
        每次凑批后按 model_id 分组，分别用对应模型推理。
        """
        device = torch.device(device_str)
        best_model, new_model = None, None
        
        try:
            from agents.neural.registry import build_model_from_checkpoint

            # ── 加载 best_model ──
            best_ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
            best_model, _, _ = build_model_from_checkpoint(best_ckpt, device=device)

            # ── 加载 new_model ──
            new_ckpt = torch.load(new_model_path, map_location=device, weights_only=False)
            new_model, _, _ = build_model_from_checkpoint(new_ckpt, device=device)
            
            if device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                
            print(f"[DualInferenceServer] 双模型并发推理启动 (Max Batch: {max_batch_size})")
        except Exception as e:
            print(f"[DualInferenceServer] 模型加载失败: {e}")
            init_error.value = True
            for q in result_queues:
                q.put(("FATAL_INIT", None, None))
        finally:
            ready_event.set()  # 无论成功失败都 set，防止主进程死锁

        if best_model is None or new_model is None:
            return

        while not shutdown_event.is_set():
            batch_data = []
            try:
                # 阻塞等待第一个请求
                item = request_queue.get(timeout=0.1) 
                batch_data.append(item)
                
                # 极速凑批
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
                # 按 model_id 分组
                best_items = [d for d in batch_data if d[1] == 0]
                new_items = [d for d in batch_data if d[1] == 1]
                
                # best_model 批量推理
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

                # new_model 批量推理
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
        """获取 worker 的请求/响应队列对。"""
        return self.request_queue, self.result_queues[worker_id]

    def shutdown(self):
        """关闭推理服务器。"""
        self.shutdown_event.set()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()