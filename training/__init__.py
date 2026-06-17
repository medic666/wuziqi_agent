"""
training - 训练基础设施

提供 AlphaZero 训练流程中的通用组件，与具体网络架构解耦。

组件:
  - replay_buffer.py: 经验回放缓冲区 (ReplayBuffer)
  - inference_server.py: GPU 推理服务器 (单模型 + 双模型并发)
  - config.py: 训练配置类 (AlphaZeroConfig, PretrainConfig, TrainConfig)

合并来源:
  - ReplayBuffer: az_train.py + pretrain_vs_agent.py (去重)
  - InferenceServer: inference_server.py + az_train.py DualInferenceServer
  - Config: az_train.py + pretrain_vs_agent.py + pre_train.py
"""