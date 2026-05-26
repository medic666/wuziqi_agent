# agent_az.py
import torch
from typing import Tuple, Optional
from agent_base import Agent as BaseAgent
from gamerules import GameState
from network import ActorCriticNet
from mcts import MCTS, create_local_eval_fn

class AZAgent(BaseAgent):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[ActorCriticNet] = None,
        num_sims: int = 400,
        c_puct: float = 1.5,
        temperature: float = 0.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.0,
        candidate_radius: int = 2,
        advantage_clip: float = 1.0,
        device_str: str = "auto",
        name: str = "AZAgent",
    ):
        self.name = name
        self.num_sims = num_sims
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.candidate_radius = candidate_radius
        self.advantage_clip = advantage_clip

        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("必须提供 model_path 或 model")

        self.model.eval()
        self.mcts = MCTS(
            eval_fn=create_local_eval_fn(self.model, self.device),
            c_puct=self.c_puct,
            num_simulations=self.num_sims,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            candidate_radius=self.candidate_radius,
            advantage_clip=self.advantage_clip,
        )

    def _load_model(self, path: str) -> ActorCriticNet:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        stem_key = 'stem_conv.weight'
        if stem_key in state_dict:
            channels = state_dict[stem_key].shape[0]
        else:
            channels = 128
        num_blocks = 0
        for key in state_dict:
            if key.startswith('res_blocks.'):
                block_idx = int(key.split('.')[1])
                num_blocks = max(num_blocks, block_idx + 1)
        model = ActorCriticNet(num_res_blocks=max(num_blocks, 4), channels=channels)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def get_move(self, state: GameState) -> Tuple[int, int]:
        last_action = state.last_move
        if not state.history:
            self.mcts.root = None
            last_action = None
        if self.mcts.root is not None and last_action is not None:
            if last_action not in self.mcts.root.children:
                self.mcts.root = None
                
        _, action, _ = self.mcts.search(state, temperature=self.temperature, last_action=last_action)
        return action

    def get_move_with_policy(self, state: GameState, temperature: float = 1.0):
        last_action = state.last_move
        if not state.history:
            self.mcts.root = None
            last_action = None
        if self.mcts.root is not None and last_action is not None:
            if last_action not in self.mcts.root.children:
                self.mcts.root = None
                
        policy, action, _ = self.mcts.search(state, temperature=temperature, last_action=last_action)
        return action, policy

    def update_model(self, model: ActorCriticNet):
        self.model = model.to(self.device)
        self.model.eval()
        self.mcts = MCTS(
            eval_fn=create_local_eval_fn(self.model, self.device),
            c_puct=self.c_puct,
            num_simulations=self.num_sims,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            candidate_radius=self.candidate_radius,
            advantage_clip=self.advantage_clip,
        )