# wuziqi_agent
基于python的、从数据收集、到预训练、到强化学习的五子棋agent训练流程

## Python版本 3.10.11
## pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
## pip install "numpy<2"

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
用的都是pretrain和az_train的参数。实践证明神经网络自对弈比较有效，很快就能对机械agent胜率100%了。这一步可以忽略

---
### 自对弈 az_train.py
num_workers: int = 16, 算存之间的通道堵塞，cpu和gpu都吃不满，16个workers往gpu塞数据，效率比较折中。
续训系统会在 checkpoints/az_train/ 目录下寻找 latest_checkpoint.pt 文件，replay_buffer.npz 文件，self_play_model.pt文件，best_model.pt文件，请全部保留

---
## 结论
当前实际训练下来的问题就是，可能因为历史通道只有上一步的棋子。所以ai被调度后，大概率遗忘之前的威胁点。不过强势的地方是ai进攻很好，有一点学习到了机械agent的vct搜杀。限于计算机性能和时间，不能再往上提高ai水平了(可能的方向是增加历史通道，增加残差块数量，增加自对弈盘数)。不过当前的无禁手五子棋实验中，神经网络通过自对弈明显胜过了机械AI老师，并且价值网络判断黑棋的胜率为90%以上，包括竞技场的自对弈中也基本上是黑棋胜，已经证明了神经网络的学习能力。项目在此结案。
实际游玩可以打开py human_vs_ai.py  --model "checkpoints/az_train/best_model_iter_59.pt"