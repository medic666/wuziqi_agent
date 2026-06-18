# run_arena.py
"""
五子棋 AI 竞技场入口 — 多种 Agent 自动对弈观战

对战模式:
  1. 规则引擎 vs CNN AlphaZero (默认)
  2. CNN AlphaZero vs Transformer
  3. 规则引擎 vs Transformer

使用方式:
  python run_arena.py                                 # 模式1: 规则 vs CNN
  python run_arena.py --mode cnn_vs_transformer       # 模式2: CNN vs Transformer
  python run_arena.py --mode rule_vs_transformer      # 模式3: 规则 vs Transformer

对战双方配置也可直接修改本文件的 if __name__ == '__main__' 块。
"""
import argparse
from agents.rule_based import ADAgent
from agents.neural.az_agent import AZAgent
from agents.neural.network import ActorCriticNet
from agents.neural.transformer_network import GoBangTransformer_v2
from arena.visual import Arena

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="五子棋 AI 竞技场")
    parser.add_argument(
        '--mode', type=str, default='rule_vs_cnn',
        choices=['rule_vs_cnn', 'cnn_vs_transformer', 'rule_vs_transformer'],
        help="对战模式 (默认: rule_vs_cnn)"
    )
    args = parser.parse_args()

    if args.mode == 'rule_vs_cnn':
        # ── 模式1: 规则引擎 vs CNN AlphaZero ──
        agent1 = ADAgent(
            depth=4, max_candidates=10,
            use_quiescence=True, quiescence_depth=2,
            vct_depth=8, name="ADAgent"
        )
        agent2 = AZAgent(
            model_path="checkpoints/az_train/best_model.pt",
            num_sims=400,
            temperature=0.0,
            dirichlet_epsilon=0.0,
            name="AlphaZero_CNN",
        )
        Arena(agent_black=agent1, agent_white=agent2)

    elif args.mode == 'cnn_vs_transformer':
        # ── 模式2: CNN AlphaZero vs Transformer ──
        cnn_agent = AZAgent(
            model_path="checkpoints/az_train/best_model.pt",
            num_sims=400,
            temperature=0.0,
            dirichlet_epsilon=0.0,
            name="CNN_AlphaZero",
        )
        tfm_agent = AZAgent(
            model_path="checkpoints/transformer_train/best_model.pt",
            num_sims=400,
            temperature=0.0,
            dirichlet_epsilon=0.0,
            name="Transformer_v2",
        )
        Arena(agent_black=cnn_agent, agent_white=tfm_agent)

    elif args.mode == 'rule_vs_transformer':
        # ── 模式3: 规则引擎 vs Transformer ──
        agent1 = ADAgent(
            depth=4, max_candidates=10,
            use_quiescence=True, quiescence_depth=2,
            vct_depth=8, name="ADAgent"
        )
        agent2 = AZAgent(
            model_path="checkpoints/transformer_train/best_model.pt",
            num_sims=400,
            temperature=0.0,
            dirichlet_epsilon=0.0,
            name="Transformer_v2",
        )
        Arena(agent_black=agent1, agent_white=agent2)
