# AGENTS.md

## Project identity
- Python 五子棋 AlphaZero training — flat scripts, no package install (`pip install torch numpy matplotlib tqdm`)
- Always run from repo root; no `setup.py`/`pyproject.toml`

## Architecture registry
- `agents/neural/registry.py` — single source of truth for network architectures
- `@register('name', param_names, defaults)` decorator on each network class
- To add architecture: (1) create file with `@register`, (2) import in `agents/neural/__init__.py`, (3) done
- Loading checkpoint: always use `registry.build_model_from_checkpoint()` — infers arch, resolves params from weights, constructs model, loads state_dict
- Alias: `'cnn'` → `'cnn_v2'`

## Checkpoint format
- **Must** include `model_config` with `arch_type` key; all save points use `model.get_config()`
- Keys: `{'model_state_dict': OrderedDict(...), 'model_config': {'arch_type': 'cnn_v3', ...}}`

## Core data types
- `GameState.board` is `bytearray` (225 bytes), NOT numpy — `state_to_tensor()` converts to `np.ndarray(3,15,15) float32`
- MCTS internal state copy: `GameState(board=bytearray(root_state.board), ...)` with empty history (performance, ~20-30% faster)
- `GomokuRules.apply_move_fast(state, action)` mutates board in-place

## Multiprocessing
- `mp.set_start_method('spawn')` at module level in `az_train.py`
- GPU inference: `InferenceServer` (single model) / `DualInferenceServer` (two models) batch worker requests via queues
- `DualInferenceServer`: model_id 0 = best, model_id 1 = new

## MCTS tree reuse (self-play only)
1. `AZAgent.get_move()` manually advances root to own previous action's child (`az_agent.py:165-171`)
2. `MCTS.search(last_action=...)` advances root to opponent's move (`mcts.py:258-287`)
3. `raw_value` cached on `MCTSNode` avoids re-evaluating reused subtrees
4. Both sides share the same `MCTS` tree — continuous reuse across the entire game

## Self-play vs arena exploration
| | Self-play | Arena |
|---|---|---|
| Decision mode | MCTS (400 sims) | Nucleus sampling (no MCTS) |
| Dirichlet noise (epsilon) | 0.25 (ON) | N/A |
| Temperature threshold | 4 moves | 4 moves (nucleus early temp) |
| Below threshold | T=1.0 | T=1.5 |
| Above threshold | T=1e-3 (near-deterministic) | Nucleus top-p=0.6, T=1.0 |

## Nucleus sampling (arena & external agents)
- `search/sampling.py` — shared nucleus (top-p) sampling function: sorts policy probs descending, keeps top moves until cumulative prob >= `nucleus_p`, samples from filtered set
- `AZAgent` supports `mode="mcts"` (default) or `mode="nucleus"`; `get_move()` dispatches accordingly
- `AZAgent.get_hint_move(state)` returns recommended move without modifying internal tree state (used by `human_vs_ai.py` hint button)
- External args: `--mode mcts|nucleus`, `--nucleus-p 0.6` (human_vs_ai.py); `--agent1-mode`/`--agent2-mode` (run_arena.py)

## Training loop
- 3-phase per iteration: self-play → training → arena → (optional) baseline eval vs rule engine
- Cosine LR schedule with warmup; HuberLoss for value, log_softmax CE for policy
- 8-way D4 symmetry augmentation per batch; advantage clipping at training time
- Arena collapse threshold: arena win rate < 0.35 → roll back model, reset optimizer
- Arena data is NOT fed to replay buffer (nucleus sampling has no MCTS policy targets)
- Two-stage safe checkpoint: saves after training phase (phase=2) so crash during arena doesn't lose progress
