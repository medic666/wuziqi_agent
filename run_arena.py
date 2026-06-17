# run_arena.py
"""
五子棋 AI 竞技场入口 — 两种 Agent 自动对弈观战

使用方式:
  python run_arena.py
  python -m arena.visual

对战双方配置请在 arena/visual.py 的 if __name__ == '__main__' 中修改。
"""
from agents.rule_based import ADAgent
from agents.neural.az_agent import AZAgent
from arena.visual import Arena

if __name__ == '__main__':
    agent1 = ADAgent(
        depth=4, max_candidates=10,
        use_quiescence=True, quiescence_depth=2,
        vct_depth=8, name="ADAgent"
    )
    az_agent2 = AZAgent(
        model_path="checkpoints/az_train/best_model.pt",
        num_sims=400,
        temperature=0.0,
        dirichlet_epsilon=0.0,
        name="AlphaCurr",
    )
    Arena(agent_black=agent1, agent_white=az_agent2)