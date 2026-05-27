# wuziqi_agent
基于python的、从数据收集、到预训练、到强化学习的五子棋agent训练流程

## Python版本 3.10.11

## pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

## 人工评估的agent_ad.py:
depth: int = 4, 
max_candidates: int = 10, 
use_quiescence: bool = True,
vct_depth: int = 8, 
quiescence_depth: int = 2, 
算杀比较强，但是防杀比较傻。

---
### 数据生成data_collector.py:
这是一个强cpu占用模块，NUM_WORKERS = 10，设置为高于cpu核心数低于超线程数。运行agent_ad自对弈时，速度为0.16局/秒。

---
### 神经网络network.py:
stem+n个残差块+双头。
当前致力于训练4resblock+128通道，一共124万参数。

---
### 预训练pre_train.py:
预训练比较简单，但是之前因为样本过少所以一训练就过拟合。batch_size:128配合lr: 1e-4，是可行的。

---
### pretrain_vs_agent.py
还没运行过，不过用的都是pretrain和az_train的参数

---
### 自对弈 az_train.py
num_workers: int = 16, 算存之间的通道堵塞，cpu和gpu都吃不满，16个workers往gpu塞数据，效率比较折中。
续训系统会在 checkpoints/az_train/ 目录下寻找 latest_checkpoint.pt 文件，replay_buffer.npz 文件，self_play_model.pt文件，best_model.pt文件，请全部保留