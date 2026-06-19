# run_arena.py
"""
五子棋 AI 竞技场入口 — 灵活的架构赛马

用法:
    python run_arena.py --agent1 rule_based --agent2 cnn_v2
    python run_arena.py --agent1 cnn_v2 --agent2 cnn_v3
    python run_arena.py --agent1 cnn_v3 --agent2 transformer
    python run_arena.py --agent1 rule_based --agent2 /path/to/custom_model.pt

支持的 --agent1/--agent2 值:
    rule_based           → ADAgent (规则引擎)
    cnn_v2               → checkpoints/az_train/cnn_v2/best_model.pt
    cnn_v3               → checkpoints/az_train/cnn_v3/best_model.pt
    transformer          → checkpoints/az_train/transformer/best_model.pt
    /path/to/model.pt    → 显式模型路径 (AZAgent 自动推断架构)
"""

import argparse
import os
from agents.rule_based import ADAgent
from agents.neural.az_agent import AZAgent
from arena.visual import Arena

# 架构名 → checkpoint 路径映射
ARCH_TO_PATH = {
    'cnn_v2': 'checkpoints/az_train/cnn_v2/best_model.pt',
    'cnn_v3': 'checkpoints/az_train/cnn_v3/best_model.pt',
    'transformer': 'checkpoints/az_train/transformer/best_model.pt',
}


def create_agent(spec: str, default_sims: int = 400):
    """根据 spec 创建 Agent。

    Args:
        spec: 'rule_based' | 架构名 | 显式模型路径
        default_sims: MCTS 模拟次数 (仅用于 AZAgent)
    """
    if spec == 'rule_based':
        return ADAgent(
            depth=4, max_candidates=10,
            use_quiescence=True, quiescence_depth=2,
            vct_depth=8, name="ADAgent",
        )

    # 判断是架构名还是显式路径
    if spec in ARCH_TO_PATH:
        model_path = ARCH_TO_PATH[spec]
        agent_name = spec
    else:
        model_path = spec
        agent_name = os.path.splitext(os.path.basename(spec))[0]

    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        print(f"   可用架构名: {list(ARCH_TO_PATH.keys())}, rule_based")
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"  加载 {agent_name}: {model_path}")
    return AZAgent(
        model_path=model_path,
        num_sims=default_sims,
        temperature=0.0,
        dirichlet_epsilon=0.0,
        name=agent_name,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="五子棋 AI 竞技场",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_arena.py --agent1 rule_based --agent2 cnn_v2
  python run_arena.py --agent1 cnn_v2 --agent2 cnn_v3
  python run_arena.py --agent1 cnn_v3 --agent2 transformer
  python run_arena.py --agent1 rule_based --agent2 checkpoints/az_train/cnn_v3/best_model.pt
        """,
    )
    parser.add_argument(
        '--agent1', type=str, default='rule_based',
        help="黑方 Agent: rule_based | cnn_v2 | cnn_v3 | transformer | /path/to/model.pt",
    )
    parser.add_argument(
        '--agent2', type=str, default='cnn_v2',
        help="白方 Agent: rule_based | cnn_v2 | cnn_v3 | transformer | /path/to/model.pt",
    )
    parser.add_argument(
        '--sims', type=int, default=400,
        help="MCTS 模拟次数 (默认: 400)",
    )
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  竞技场: {args.agent1} (黑) vs {args.agent2} (白)")
    print(f"  MCTS: {args.sims} 次模拟")
    print(f"{'='*50}\n")

    agent_black = create_agent(args.agent1, args.sims)
    agent_white = create_agent(args.agent2, args.sims)

    Arena(agent_black=agent_black, agent_white=agent_white)