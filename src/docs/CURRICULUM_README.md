# RLCatan Curriculum Learning Readme



Curriculum learning trains the agent in phases, starting with a simplified game with fewer actions, lower target VP, and progressively increases difficulty. 
Each phase can also apply reward shaping to guide early learning.

---

## Run Training

```powershell
cd C:\Users\Matthew\Documents\RLCatan\src
.\venv\Scripts\activate
$env:RLCATAN_CURRICULUM_JSON="..\configs\curriculum.json"
python -m rlcatan.training.train_ppo_v1
```

---

## Curriculum Configuration (`configs/curriculum.json`)

Each phase can:
- Disable actions (simplify the action space)
- Set a target VP (future: can be enforced by the env)
- Define when to advance to the next phase
- Configure reward shaping

### Example configuration

```json
{
  "phases": [
    {
      "name": "phase-1-basic",
      "target_vp": 5,
      "disabled_action_types": ["MOVE_ROBBER", "BUY_DEVELOPMENT_CARD"],
      "min_episodes": 100,
      "advance_threshold": {"metric": "win_rate", "value": 0.6},
      "reward_shaping": {
        "enabled": true,
        "gain_scale": 0.15,
        "spend_scale": 0.05,
        "decay_factor": 0.999,
        "build_scale": 0.02,
        "debug": false
      }
    },
    {
      "name": "phase-2-intermediate",
      "target_vp": 8,
      "disabled_action_types": ["MOVE_ROBBER"],
      "min_episodes": 200,
      "advance_threshold": {"metric": "win_rate", "value": 0.5},
      "reward_shaping": {
        "enabled": true,
        "gain_scale": 0.1,
        "spend_scale": 0.05,
        "decay_factor": 0.999,
        "build_scale": 0.02,
        "debug": false
      }
    },
    {
      "name": "phase-3-full",
      "target_vp": 10,
      "disabled_action_types": [],
      "min_episodes": 300,
      "advance_threshold": {"metric": "win_rate", "value": 0.4},
      "reward_shaping": {
        "enabled": true,
        "gain_scale": 0.05,
        "spend_scale": 0.03,
        "decay_factor": 0.999,
        "build_scale": 0.01,
        "debug": false
      }
    }
  ]
}
```

### Phase fields explanation:

- `name`: Phase name for logs.
- `target_vp`: Intended VP goal for the phase.
- `disabled_action_types`: Actions to remove from the agent’s action space.
- `min_episodes`: Minimum episodes before phase can advance.
- `advance_threshold`:
  - `metric`: One of `win_rate`, `avg_reward`, `avg_vp`.
  - `value`: Threshold required to advance.
- `reward_shaping`: Optional config passed to `RewardWrapper`.
  - `enabled`: Turn shaping on/off for the phase.
  - `gain_scale`: Reward per resource gained.
  - `spend_scale`: Reward per resource spent.
  - `decay_factor`: Per-step reward decay multiplier.
  - `build_scale`: Flat bonus for build actions.
  - `player_idx`: Optional fixed player index to track.
  - `resource_attr`: Key for resource counts in `info`.
  - `debug`: Verbose shaping diagnostics.

---

## What You’ll See in the Console

- Curriculum start info:
  - Phase name, target VP, disabled actions
  - Reward shaping config if enabled
- PPO rollout tables every few seconds:
  - `ep_rew_mean` mean episode reward
  - `ep_len_mean` mean episode length

Example:

```
[Curriculum] Starting with phase: phase-1-basic
[Curriculum] Target VP: 5, Disabled actions: ['MOVE_ROBBER', 'BUY_DEVELOPMENT_CARD']
[Curriculum] Reward shaping enabled with config: {...}

---------------------------------
| rollout/           |          |
|    ep_len_mean     | 500      |
|    ep_rew_mean     | -0.2     |
|    total_timesteps | 2048     |
---------------------------------
```

Phase advancement looks like this:

```
CURRICULUM PHASE ADVANCED!
Previous phase: phase-1-basic
New phase: phase-2-intermediate
Target VP: 8
Disabled actions: ['MOVE_ROBBER']
```

---

### Example training session

```powershell
cd C:\Users\Matthew\Documents\RLCatan\src
.\venv\Scripts\activate
$env:RLCATAN_CURRICULUM_JSON="..\configs\curriculum.json"
python -c "from rlcatan.training.train_ppo_v1 import train_ppo; train_ppo(total_timesteps=50000, curriculum_json='..\\configs\\curriculum.json', save_path=None)"
```

### Change reward shaping strength

Edit `configs/curriculum.json`, adjust `gain_scale`, `spend_scale`, or `build_scale` in early phases, then rerun training.

### Save models

Models are saved to `src/models/ppo_v1.zip` by default when `save_path` is provided.

---

## Common Issues

### Training appears stuck

This is normal at startup. PPO only prints after collecting a full rollout (`n_steps`, default 2048). Wait 7–10 seconds for the first table.


### Curriculum doesn’t advance

- `min_episodes` not reached yet.
- `advance_threshold` is too high.
- The metric name is wrong.

Try lowering thresholds first (e.g., `win_rate` 0.3–0.5).

---

## Files to Know

- Training entrypoint: `src/rlcatan/training/train_ppo_v1.py`
- Curriculum manager: `src/rlcatan/training/curriculum.py`
- Curriculum callback: `src/rlcatan/training/curriculum_callback.py`
- Reward wrapper: `src/catanatron/catanatron/gym/reward_wrapper.py`
- Curriculum config: `configs/curriculum.json`

---

## Next Steps

- Train model with increased timesteps using McMaster servers.
- Tune reward shaping and phase thresholds based on the learning curves.
