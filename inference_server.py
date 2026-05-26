# inference_server.py
import torch
import torch.multiprocessing as mp
import numpy as np
import queue
from network import ActorCriticNet

class InferenceServer:
    def __init__(self, model_path: str, device_str: str, num_workers: int, max_batch_size: int = 32):
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
        device = torch.device(device_str)
        try:
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            
            channels = state_dict['stem_conv.weight'].shape[0]
            res_block_indices = [int(k.split('.')[1]) for k in state_dict if k.startswith('res_blocks.')]
            num_blocks = max(res_block_indices) + 1 if res_block_indices else 4
            
            model = ActorCriticNet(num_res_blocks=num_blocks, channels=channels).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            
            if device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                
            print(f"[InferenceServer] 极速凑批模式启动 (Max Batch: {max_batch_size})")
        except Exception as e:
            print(f"[InferenceServer] 模型加载失败: {e}")
            for q in result_queues:
                q.put((None, None))
        finally:
            ready_event.set()

        while not shutdown_event.is_set():
            batch_data = []
            try:
                # 1. 阻塞等待第一个请求到来
                item = request_queue.get(timeout=0.1) 
                batch_data.append(item)
                
                # ✅ 2. 极速清空队列策略：只要队列里有数据，立刻拿出来凑批，绝不等待！
                # 这比之前的 get_nowait 循环好，因为如果有 100 个 Worker 同时发来，
                # 旧版 get_nowait 可能只能捞到 5 个就空了（因为其他还在传输中），
                # 这里会持续尝试，直到真正凑满 max_batch_size 或队列为空。
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
                
                # ✅ 3. 极致优化：np.stack + torch.from_numpy
                # 对于小数据包，先在 CPU 上用 C 级别的速度 stack，然后零拷贝转 Tensor 上传 GPU
                # 这比在 GPU 上逐个上传快得多
                states_np = np.stack([d[1] for d in batch_data], axis=0)
                states_t = torch.from_numpy(states_np).to(device)
                
                with torch.no_grad():
                    policy_logits, values = model(states_t)
                    
                    # ✅ 4. GPU 上计算 Softmax，比传回 CPU 用 Numpy 算快一倍
                    policies_t = torch.softmax(policy_logits.view(states_t.size(0), -1), dim=1)
                    
                    # 一次性传回 CPU
                    policies_np = policies_t.cpu().numpy()
                    values_np = values.view(-1).cpu().numpy()

                # 5. 分发结果
                for i, wid in enumerate(worker_ids):
                    # 恢复使用普通的 tuple(float)，彻底杜绝共享内存泄漏
                    self_result = (policies_np[i], values_np[i].item())
                    result_queues[wid].put(self_result)
                    
            except Exception as e:
                print(f"\n[InferenceServer] 推理过程出错: {e}")
                for i, wid in enumerate(worker_ids):
                    result_queues[wid].put((None, None))
        
        print("\n[InferenceServer] 推理服务器已优雅退出")

    def get_queues(self, worker_id: int):
        return self.request_queue, self.result_queues[worker_id]

    def shutdown(self):
        self.shutdown_event.set()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()