# A2C on LunarLander - Practical 4

This practical implements the Advantage Actor-Critic (A2C) algorithm trained on the LunarLander-v3 environment.

## Files

- `networks.py` - ActorNetwork and CriticNetwork definitions
- `a2c_agent.py` - A2C loss computation
- `train.py` - Main training loop (600 episodes)
- `experiments.py` - Hyperparameter experiments
- `report/` - Directory for report files

## Installation

```bash
pip install torch gymnasium[box2d] matplotlib numpy
```

## Quick Start

### Basic Training (600 episodes)

```bash
python train.py
```

This will:

- Train A2C for 600 episodes on LunarLander-v3
- Print progress every 50 episodes
- Save the trained actor network to `actor_lunarlander.pth`
- Generate and display a learning curve plot

Expected output pattern:

```
Ep   50 | Reward: -187.3 | Avg(50): -201.4
Ep  100 | Reward: -143.1 | Avg(50): -168.2
...
Ep  600 | Reward: 178.6 | Avg(50): 156.4
```

### Run Hyperparameter Experiments (Task 5)

```bash
python experiments.py
```

This tests:

- **Entropy coefficient**: c_entropy ∈ {0.0, 0.01, 0.05}
- **Critic coefficient**: c_value ∈ {0.1, 0.5, 1.0}
- **Hidden dimension**: hidden_dim ∈ {64, 128, 256}

Results are plotted and saved to `a2c_experiments.png`.

## Architecture

### ActorNetwork

- Input: 8D state (LunarLander observation)
- Hidden: 128 neurons (ReLU)
- Output: 4 action probabilities (Softmax)

### CriticNetwork

- Input: 8D state
- Hidden: 128 neurons (ReLU)
- Output: 1 scalar value estimate V(s)

## Key Hyperparameters

| Parameter                   | Value  |
| --------------------------- | ------ |
| Learning rate (actor)       | 0.0003 |
| Learning rate (critic)      | 0.001  |
| Discount factor (γ)         | 0.99   |
| Critic loss coeff (c_value) | 0.5    |
| Entropy coeff (c_entropy)   | 0.01   |

## Algorithm Overview

A2C extends REINFORCE by:

1. Adding a Critic network that learns V(s)
2. Computing advantages: A(s,a) = G_t - V(s)
3. Reducing variance through advantage-based updates
4. Adding entropy regularization for exploration

### Loss Function

```
L_total = L_actor + c_value * L_critic + c_entropy * H
```

Where:

- L_actor = -Σ A(s,a) \* log π(a|s)
- L_critic = Σ (G_t - V(s))²
- H = -Σ π(a|s) \* log π(a|s)

## Expected Performance

- Episodes 1-100: Highly negative rewards (-200 to -100) - normal due to new agent
- Episodes 100-300: Steady improvement visible
- Episodes 300-600: Average reward increases toward solving threshold (200+)

Full convergence typically requires 500-1000 episodes.

## Notes

- LunarLander-v3 is used (v2 is deprecated)
- Rewards start very negative due to fuel costs
- Both actor and critic are updated every episode
- Advantage computation uses `.detach()` to prevent critic updates via actor loss
