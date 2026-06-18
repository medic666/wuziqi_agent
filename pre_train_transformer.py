# pre_train_transformer.py
"""
五子棋 GoBangTransformer_v2 联合预训练模块
(从 pre_train.py 改造为 Transformer 版本)

支持: 8向动态增强 + 精准Action索引 + HuberLoss
使用 training_data.npz 进行 Behavior Cloning + Value Regression 预训练

用法:
    python pre_train_transformer.py
    python pre_train_transformer.py --resume --max_epochs 100
"""

import math
import os
import sys
import time
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from agents.neural.transformer_network import GoBangTransformer_v2
from utils.transforms import transform_state, transform_2d


# ═══════════════════════════════════════════════════════════════
#  Transformer 预训练配置
# ═══════════════════════════════════════════════════════════════

class TransformerPretrainConfig:
    """GoBangTransformer_v2 联合预训练配置。"""

    def __init__(
        self,
        data_path: str = "collected_data/training_data.npz",
        val_ratio: float = 0.1,
        max_samples: int = 0,
        # ── Transformer 网络架构 ──
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 5,
        ff_expand: int = 4,
        dropout: float = 0.1,
        board_size: int = 15,
        # ── 训练参数 ──
        batch_size: int = 128,
        max_epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        actor_loss_weight: float = 1.0,
        critic_loss_weight: float = 1.0,
        loss_type: str = "huber",
        grad_clip: float = 1.0,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        patience: int = 15,
        min_delta: float = 1e-5,
        checkpoint_dir: str = "checkpoints/transformer_pretrain",
        save_interval: int = 5,
        device: str = "auto",
        num_workers: int = 8,
        pin_memory: bool = True,
        resume: bool = False,
        resume_path: Optional[str] = None,
        load_weights: Optional[str] = None,
    ):
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = cls().__dict__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def get_model_config(self) -> dict:
        return {
            'arch_type': 'transformer',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_expand': self.ff_expand,
            'dropout': self.dropout,
            'board_size': self.board_size,
        }


# ═══════════════════════════════════════════════════════════════
#  数据集 (支持8向动态增强)
# ═══════════════════════════════════════════════════════════════

class GomokuDataset(Dataset):
    def __init__(self, states: np.ndarray, actions: np.ndarray, results: np.ndarray):
        self.states = torch.from_numpy(states).float()
        self.actions_2d = torch.from_numpy(actions.reshape(-1, 15, 15)).float()
        self.results = torch.from_numpy(results).float()

    def __len__(self):
        return len(self.states) * 8  # 8向增强

    def __getitem__(self, idx):
        orig_idx = idx // 8
        tid = idx % 8

        state_np = self.states[orig_idx].numpy()
        action_2d_np = self.actions_2d[orig_idx].numpy()

        if tid != 0:
            state_np = transform_state(state_np, tid)
            action_2d_np = transform_2d(action_2d_np, tid)

        action_idx = np.argmax(action_2d_np.reshape(-1))
        state_tensor = torch.from_numpy(np.ascontiguousarray(state_np))
        action_tensor = torch.tensor(action_idx, dtype=torch.long)
        result_tensor = self.results[orig_idx]

        return state_tensor, action_tensor, result_tensor


# ═══════════════════════════════════════════════════════════════
#  Transformer 联合训练器
# ═══════════════════════════════════════════════════════════════

class JointTransformerTrainer:
    """基于 Transformer 的联合预训练器。"""

    def __init__(self, config: TransformerPretrainConfig):
        self.config = config
        self.device = self._get_device()

        model_cfg = config.get_model_config()
        self.model = GoBangTransformer_v2(
            **{k: v for k, v in model_cfg.items() if k != 'arch_type'}
        ).to(self.device)

        self.train_loader, self.val_loader = self._build_dataloaders()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = self._build_scheduler()

        self.critic_loss_fn = nn.HuberLoss(delta=0.5) if config.loss_type == "huber" else nn.MSELoss()

        # AMP 混合精度（仅 CUDA 生效，CPU 自动走 fp32）
        self._amp_enabled = (self.device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.es_counter = 0
        self.history = []
        self.global_start_time = None

        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        if config.resume:
            self._load_checkpoint()
        if config.load_weights:
            self._load_weights_only(config.load_weights)
        self._print_header()

    def _get_device(self):
        cfg = self.config.device.lower()
        if cfg in ("auto", "cuda"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(cfg)

    def _build_dataloaders(self):
        print(f"正在加载数据: {self.config.data_path}")
        data = np.load(self.config.data_path, allow_pickle=False)
        states = data['states'].astype(np.float32)
        actions = data['target_actions'].astype(np.float32)
        results_key = 'target_results' if 'target_results' in data else 'results'
        results = data[results_key].astype(np.float32)

        n_total = len(states)
        if self.config.max_samples > 0:
            n_total = min(n_total, self.config.max_samples)
            states, actions, results = states[:n_total], actions[:n_total], results[:n_total]

        rng = np.random.RandomState(42)
        indices = rng.permutation(n_total)
        n_val = max(1, int(n_total * self.config.val_ratio))

        train_ds = GomokuDataset(states[indices[n_val:]], actions[indices[n_val:]], results[indices[n_val:]])
        val_ds = GomokuDataset(states[indices[:n_val]], actions[indices[:n_val]], results[indices[:n_val]])

        print(f"✓ 数据加载完成: 训练集 {len(train_ds):,} (含8向增强), 验证集 {len(val_ds):,} (含8向增强)")

        nw = self.config.num_workers
        pm = self.config.pin_memory and self.device.type == 'cuda'
        pw = nw > 0

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True,
            num_workers=nw, pin_memory=pm, persistent_workers=pw,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size * 2, shuffle=False,
            num_workers=nw, pin_memory=pm, persistent_workers=pw,
        )
        return train_loader, val_loader

    def _build_scheduler(self, last_epoch=-1):
        if self.config.scheduler_type == "cosine":
            warmup, total = self.config.warmup_epochs, self.config.max_epochs
            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / max(1, warmup)
                return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup) / max(1, total - warmup)))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda, last_epoch=last_epoch)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

    def _print_header(self):
        c = self.config
        total_p = sum(p.numel() for p in self.model.parameters())
        print("\n" + "=" * 100)
        print("  五子棋 GoBangTransformer_v2 联合预训练 (8向动态增强 + HuberLoss)")
        print("=" * 100)
        print(f"  网络: Transformer d={c.d_model} H={c.num_heads} L={c.num_layers} | 参数量: {total_p:,}")
        print(f"  Dropout: {c.dropout} | FFN扩展: {c.ff_expand}x")
        print(f"  设备: {self.device}" + (f" ({torch.cuda.get_device_name()})" if self.device.type == 'cuda' else ""))
        print(f"  数据: 8向动态增强 | DataLoader Workers: {c.num_workers}")
        print(f"  Loss: Actor(CE)×{c.actor_loss_weight} + Critic({c.loss_type.upper()})×{c.critic_loss_weight}")
        print(f"  优化: LR={c.learning_rate:.1e}, BS={c.batch_size}, WD={c.weight_decay}")
        if c.resume:
            print(f"  续训: Epoch {self.current_epoch} | Best Val Loss: {self.best_val_loss:.6f}")
        print("=" * 100)

    def train(self):
        self.global_start_time = time.time()
        try:
            for epoch in range(self.current_epoch, self.config.max_epochs):
                self.current_epoch = epoch
                t0 = time.time()

                t_loss, t_actor, t_critic, t_top1, t_top3, t_top5, t_mae = self._train_epoch()
                v_loss, v_actor, v_critic, v_top1, v_top3, v_top5, v_mae = self._validate()

                lr = self.optimizer.param_groups[0]['lr']
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(v_loss)
                else:
                    self.scheduler.step()

                improved = v_loss < self.best_val_loss - self.config.min_delta
                if improved:
                    self.best_val_loss = v_loss
                    self.es_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.es_counter += 1

                self._log_epoch(t_loss, t_actor, t_critic, t_top1, t_top3, t_top5, t_mae,
                                v_loss, v_actor, v_critic, v_top1, v_top3, v_top5, v_mae, lr, time.time() - t0, improved)

                if (epoch + 1) % self.config.save_interval == 0 and not improved:
                    self._save_checkpoint()
                if self.es_counter >= self.config.patience:
                    print(f"\n⚠ 早停触发！验证损失连续 {self.config.patience} 轮未改善")
                    break
        except KeyboardInterrupt:
            print("\n⚠ 训练中断，保存检查点...")
            self._save_checkpoint()
        self._save_checkpoint()
        print("\n✓ 训练结束！")

    def _calculate_topk(self, logits, actions, ks=(1, 3, 5)):
        max_k = max(ks)
        _, topk_indices = logits.topk(max_k, dim=1)
        correct = topk_indices == actions.unsqueeze(1)
        accs = []
        for k in ks:
            accs.append(correct[:, :k].any(dim=1).float().sum().item())
        return accs[0], accs[1], accs[2]

    def _train_epoch(self):
        self.model.train()
        total_loss, total_actor, total_critic = 0, 0, 0
        correct_top1, correct_top3, correct_top5, mae_sum, count = 0, 0, 0, 0, 0

        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch+1:03d} [Train]", leave=False)

        for states, actions, results in pbar:
            states, actions, results = states.to(self.device), actions.to(self.device), results.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # AMP 混合精度前向：autocast 自动将计算转为 fp16（仅 CUDA 生效）
            with torch.cuda.amp.autocast(enabled=self._amp_enabled):
                logits, values = self.model(states)
                actor_loss = F.cross_entropy(logits, actions)
                critic_loss = self.critic_loss_fn(values, results)
                loss = self.config.actor_loss_weight * actor_loss + self.config.critic_loss_weight * critic_loss

            # GradScaler 缩放 loss → backward → unscale → clip → step
            self.scaler.scale(loss).backward()
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = states.size(0)
            total_loss += loss.item() * bs
            total_actor += actor_loss.item() * bs
            total_critic += critic_loss.item() * bs

            b_top1, b_top3, b_top5 = self._calculate_topk(logits, actions)
            correct_top1 += b_top1
            correct_top3 += b_top3
            correct_top5 += b_top5
            mae_sum += torch.abs(values - results).sum().item()
            count += bs

            pbar.set_postfix(Loss=f"{loss.item():.3f}", A=f"{actor_loss.item():.3f}", C=f"{critic_loss.item():.3f}")

        n = len(self.train_loader.dataset)
        return total_loss / n, total_actor / n, total_critic / n, correct_top1 / n, correct_top3 / n, correct_top5 / n, mae_sum / n

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total_loss, total_actor, total_critic = 0, 0, 0
        correct_top1, correct_top3, correct_top5, mae_sum, count = 0, 0, 0, 0, 0

        pbar = tqdm(self.val_loader, desc=f"Ep {self.current_epoch+1:03d} [Val  ]", leave=False)

        for states, actions, results in pbar:
            states, actions, results = states.to(self.device), actions.to(self.device), results.to(self.device)
            logits, values = self.model(states)

            actor_loss = F.cross_entropy(logits, actions)
            critic_loss = self.critic_loss_fn(values, results)
            loss = self.config.actor_loss_weight * actor_loss + self.config.critic_loss_weight * critic_loss

            bs = states.size(0)
            total_loss += loss.item() * bs
            total_actor += actor_loss.item() * bs
            total_critic += critic_loss.item() * bs

            b_top1, b_top3, b_top5 = self._calculate_topk(logits, actions)
            correct_top1 += b_top1
            correct_top3 += b_top3
            correct_top5 += b_top5
            mae_sum += torch.abs(values - results).sum().item()
            count += bs

            pbar.set_postfix(Loss=f"{loss.item():.3f}", A=f"{actor_loss.item():.3f}", C=f"{critic_loss.item():.3f}")

        n = len(self.val_loader.dataset)
        return total_loss / n, total_actor / n, total_critic / n, correct_top1 / n, correct_top3 / n, correct_top5 / n, mae_sum / n

    def _log_epoch(self, t_loss, t_actor, t_critic, t_top1, t_top3, t_top5, t_mae,
                   v_loss, v_actor, v_critic, v_top1, v_top3, v_top5, v_mae, lr, time_elapsed, improved):
        elapsed = time.time() - self.global_start_time
        eta = (elapsed / max(1, self.current_epoch + 1)) * (self.config.max_epochs - self.current_epoch - 1)
        eta_str = f"{eta / 3600:.1f}h" if eta > 3600 else f"{eta / 60:.1f}m"
        best_mark = " ★" if improved else ""
        es_info = f" │ ES {self.es_counter}/{self.config.patience}" if self.es_counter > 0 else ""

        print(f"\n[Ep {self.current_epoch+1:03d}/{self.config.max_epochs} Summary] "
              f"Train Loss: {t_loss:.4f} (Actor:{t_actor:.4f} | Critic:{t_critic:.4f}) "
              f"Acc T1/T3/T5: {t_top1:.1%}/{t_top3:.1%}/{t_top5:.1%} MAE:{t_mae:.3f}")
        print(f"{' ' * 27} "
              f"Val   Loss: {v_loss:.4f} (Actor:{v_actor:.4f} | Critic:{v_critic:.4f}) "
              f"Acc T1/T3/T5: {v_top1:.1%}/{v_top3:.1%}/{v_top5:.1%} MAE:{v_mae:.3f} "
              f"│ LR:{lr:.2e} │ {time_elapsed:.1f}s │ ETA {eta_str}{es_info}{best_mark}")

    def _save_checkpoint(self, is_best=False):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'es_counter': self.es_counter,
            'config': self.config.to_dict(),
            'model_config': self.config.get_model_config(),
        }
        if not is_best:
            torch.save(state, os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt'))
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': self.config.get_model_config(),
            }, os.path.join(self.config.checkpoint_dir, 'best_model.pt'))
            torch.save(state, os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt'))

    def _load_checkpoint(self):
        path = self.config.resume_path or os.path.join(self.config.checkpoint_dir, 'latest_checkpoint.pt')
        if not os.path.exists(path):
            return
        print(f"正在加载检查点: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        saved_cfg = TransformerPretrainConfig.from_dict(ckpt.get('config', {}))
        # 恢复网络架构参数
        for p in ['d_model', 'num_heads', 'num_layers', 'ff_expand', 'dropout', 'board_size']:
            if hasattr(saved_cfg, p):
                setattr(self.config, p, getattr(saved_cfg, p))
        model_cfg = self.config.get_model_config()
        self.model = GoBangTransformer_v2(
            **{k: v for k, v in model_cfg.items() if k != 'arch_type'}
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for s in self.optimizer.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(self.device)
        self.current_epoch = ckpt['epoch'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.es_counter = ckpt.get('es_counter', 0)
        if self.config.scheduler_type == "cosine":
            self.scheduler = self._build_scheduler(last_epoch=self.current_epoch - 1)
        else:
            self.scheduler.load_state_dict(ckpt.get('scheduler_state_dict', {}))
        print(f"✓ 恢复至 Epoch {self.current_epoch} | Best Val: {self.best_val_loss:.4f}")

    def _load_weights_only(self, path):
        if not os.path.exists(path):
            return
        print(f"★ 干净续训: 加载权重 {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GoBangTransformer_v2 联合预训练")
    parser.add_argument('--resume', action='store_true', default=False, help="从检查点续训")
    parser.add_argument('--max_epochs', type=int, default=None, help="最大训练轮数")
    parser.add_argument('--load_weights', type=str, default=None, help="仅加载权重（不恢复优化器）")
    args = parser.parse_args()

    config = TransformerPretrainConfig()
    if args.max_epochs:
        config.max_epochs = args.max_epochs
    if args.resume:
        config.resume = True
    if args.load_weights:
        config.load_weights = args.load_weights

    trainer = JointTransformerTrainer(config)
    trainer.train()