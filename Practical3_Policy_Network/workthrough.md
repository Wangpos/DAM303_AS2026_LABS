# Practical 3: Policy Network Implementation Workthrough

## Overview

This document provides a detailed walkthrough of how the `practical3_policy_network.ipynb` notebook implements the REINFORCE algorithm for CartPole-v1. It explains the implementation, calculations, and step-by-step workflow.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Task 1: Policy Network Implementation](#task-1-policy-network-implementation)
3. [Task 2: Return Calculation](#task-2-return-calculation)
4. [Task 3: REINFORCE Update](#task-3-reinforce-update)
5. [Task 4: Training Loop](#task-4-training-loop)
6. [Task 5: Experiments](#task-5-experiments)
7. [Complete Workflow Example](#complete-workflow-example)

---

## Architecture Overview

### From Q-Learning to Policy Networks

- **Practical 2 (Q-Learning)**: Used a table to store Q-values for discrete states
- **Practical 3 (Policy Network)**: Uses a neural network to approximate the policy π(a|s)

### Environment: CartPole-v1

| Property              | Details                                                                              |
| --------------------- | ------------------------------------------------------------------------------------ |
| **Observation Space** | 4 continuous values: cart position, cart velocity, pole angle, pole angular velocity |
| **Action Space**      | 2 discrete actions: 0 (push left) or 1 (push right)                                  |
| **Reward**            | +1 for each step the pole stays upright                                              |
| **Episode Ends**      | Pole angle > 12°, cart out of bounds, or 500 steps reached                           |
| **Solved**            | Average reward ≥ 475 over 100 consecutive episodes                                   |

---

## Task 1: Policy Network Implementation

### Purpose

Create a neural network that takes a CartPole state vector as input and outputs action probabilities.

### Architecture

```
Input Layer (4 neurons)
    ↓
Hidden Layer 1 (64 neurons) + ReLU
    ↓
Hidden Layer 2 (64 neurons) + ReLU
    ↓
Output Layer (2 neurons) + Softmax
    ↓
Action Probabilities [P(Left), P(Right)]
```

### Implementation Details

#### Class Definition: `PolicyNetwork`

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)      # 4 → 64
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)     # 64 → 64
        self.fc3 = nn.Linear(hidden_dim, action_dim)     # 64 → 2
```

#### Forward Pass

```python
def forward(self, x):
    """
    Input: x of shape (4,) — CartPole state vector

    Step 1: Apply first linear layer and ReLU
        x = fc1(x)  # Shape: (64,)
        x = ReLU(x) # Activation function, shape: (64,)

    Step 2: Apply second linear layer and ReLU
        x = fc2(x)  # Shape: (64,)
        x = ReLU(x) # Activation function, shape: (64,)

    Step 3: Apply output linear layer
        x = fc3(x)  # Shape: (2,)
        x = Softmax(x, dim=-1) # Convert to probabilities summing to 1

    Output: Action probabilities of shape (2,)
             Example: [0.35, 0.65] means 35% left, 65% right
    """
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.softmax(self.fc3(x), dim=-1)
    return x
```

#### Action Selection: `select_action()`

```python
def select_action(self, state):
    """
    Purpose: Given a state, sample an action according to the learned policy

    Step 1: Convert numpy array to PyTorch tensor
        state_tensor = torch.FloatTensor(state)  # Convert [0.1, 0.2, -0.1, 0.05]

    Step 2: Get action probabilities from the network
        probs = self.forward(state_tensor)       # Example: [0.45, 0.55]

    Step 3: Create probability distribution
        dist = Categorical(probs)                # Discrete probability distribution

    Step 4: Sample an action from the distribution
        action = dist.sample()                   # Returns 0 or 1 probabilistically
        # If action=0 (left):  selected with probability 0.45
        # If action=1 (right): selected with probability 0.55

    Step 5: Compute log probability of the selected action
        log_prob = dist.log_prob(action)
        # Example: if action=1, log_prob = log(0.55) ≈ -0.598
        # This is stored for the gradient computation later

    Return: (action, log_prob)
            action: integer 0 or 1
            log_prob: tensor with gradient tracking enabled
    """
    state_tensor = torch.FloatTensor(state)
    probs = self.forward(state_tensor)
    dist = Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)
```

#### Why This Design?

- **ReLU activations**: Introduce non-linearity, allow network to learn complex state-action mappings
- **Softmax output**: Ensures probabilities sum to 1, valid probability distribution
- **Log probabilities**: Needed for gradient computation via log-likelihood

---

## Task 2: Return Calculation

### Purpose

Compute the discounted cumulative reward from each time step to the end of the episode.

### Mathematical Background

**Discounted Return at time t:**
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... + \gamma^{T-t} r_T$$

where:

- $r_t$ = reward at time t
- $\gamma$ = discount factor (typically 0.99)
- $T$ = episode length

### Implementation: `compute_returns()`

```python
def compute_returns(rewards, gamma=0.99):
    """
    Purpose: Calculate discounted returns working BACKWARDS from episode end

    Example: rewards = [1, 1, 1] (episode of length 3, each step gives reward 1)

    Iteration order: BACKWARDS (from end to start)
    ─────────────────────────────────────────

    Initialize: G = 0, returns = []

    [Iteration 1] Last timestep (t=2):
        G = reward[2] + gamma * G
        G = 1 + 0.99 * 0 = 1.0
        Insert G at position 0 → returns = [1.0]

    [Iteration 2] Middle timestep (t=1):
        G = reward[1] + gamma * G
        G = 1 + 0.99 * 1.0 = 1.99
        Insert G at position 0 → returns = [1.99, 1.0]

    [Iteration 3] First timestep (t=0):
        G = reward[0] + gamma * G
        G = 1 + 0.99 * 1.99 = 2.9701
        Insert G at position 0 → returns = [2.9701, 1.99, 1.0]

    Return: torch.FloatTensor([2.9701, 1.99, 1.0])

    Verification:
    ────────────
    G_0 = 1 + 0.99*1 + 0.99²*1 = 1 + 0.99 + 0.9801 = 2.9701 ✓
    G_1 = 1 + 0.99*1 = 1 + 0.99 = 1.99 ✓
    G_2 = 1 = 1.0 ✓
    """
    returns = []
    G = 0
    for reward in reversed(rewards):  # Iterate backward
        G = reward + gamma * G
        returns.insert(0, G)  # Insert at beginning to maintain forward order
    return torch.FloatTensor(returns)
```

### Key Points

- **Why backwards?** We need future rewards to compute current returns
- **Why insert at position 0?** To maintain timeline order: G[0] is first, G[T-1] is last
- **Why discount?** Future rewards are worth less than immediate rewards

---

## Task 3: REINFORCE Update

### Purpose

Use policy gradient theorem to update network weights to increase probability of good actions.

### Mathematical Background

**REINFORCE Loss:**
$$\text{loss} = -\sum_{t=0}^{T-1} G_t \cdot \log \pi(a_t | s_t)$$

Why the negative sign?

- PyTorch minimizes the loss
- We want to maximize $\sum G_t \cdot \log \pi(a_t | s_t)$
- Maximizing X = Minimizing -X

### Implementation: `reinforce_update()`

```python
def reinforce_update(log_probs, returns, optimizer):
    """
    Purpose: Perform one gradient update to improve the policy

    Inputs:
    ──────
    log_probs: List of log probability tensors from the episode
               Example: [-0.598, -0.654, -0.673] (log probabilities for each step)

    returns: Tensor of discounted returns
             Example: [2.9701, 1.99, 1.0] (normalized or not)

    optimizer: PyTorch optimizer (Adam in our case)

    ═════════════════════════════════════════════════════════════════
    STEP 1: Stack log probabilities into a single tensor
    ═════════════════════════════════════════════════════════════════

    log_probs_tensor = torch.stack(log_probs)

    Before:  [-0.598]    [-0.654]    [-0.673]  (three separate 0-D tensors)
    After:   [-0.598, -0.654, -0.673]  (1-D tensor of shape (3,))

    ═════════════════════════════════════════════════════════════════
    STEP 2: Compute REINFORCE loss
    ═════════════════════════════════════════════════════════════════

    loss = -torch.sum(returns * log_probs_tensor)

    Element-wise multiplication:
    returns × log_probs = [2.9701, 1.99, 1.0] * [-0.598, -0.654, -0.673]
                        = [-1.777, -1.302, -0.673]

    Sum:
    sum = -1.777 + -1.302 + -0.673 = -3.752

    Negation (for minimization):
    loss = -(-3.752) = 3.752

    Interpretation:
    ───────────────
    A positive loss value means on average the policy took good actions
    (high returns weighted with high log probabilities).
    By minimizing this loss, we're actually maximizing expected return.

    ═════════════════════════════════════════════════════════════════
    STEP 3: Backpropagation and weight update
    ═════════════════════════════════════════════════════════════════

    [3a] Clear old gradients
    optimizer.zero_grad()
    # PyTorch accumulates gradients by default, so we must clear them first

    [3b] Compute gradients via backpropagation
    loss.backward()
    # This computes ∂loss/∂w for every weight w in the network
    # Uses chain rule automatically through PyTorch's autograd

    [3c] Update weights
    optimizer.step()
    # Adam optimizer updates each weight:
    # w_new = w_old - learning_rate * gradient

    Return: loss.item()
            # Convert tensor to Python scalar for logging
    """

    # Step 1: Stack log probabilities
    log_probs_tensor = torch.stack(log_probs)

    # Step 2: Compute loss
    loss = -torch.sum(returns * log_probs_tensor)

    # Step 3: Gradient update (CRUCIAL ORDER)
    optimizer.zero_grad()      # Clear old gradients FIRST
    loss.backward()            # Compute new gradients
    optimizer.step()           # Apply update

    return loss.item()
```

### Critical Implementation Detail: Gradient Order

```
✓ CORRECT ORDER:
  1. optimizer.zero_grad()   ← Clear old gradients
  2. loss.backward()         ← Compute new gradients
  3. optimizer.step()        ← Update weights

✗ WRONG ORDER (common mistake):
  1. loss.backward()         ← Compute gradients
  2. optimizer.step()        ← Update weights
  3. optimizer.zero_grad()   ← Clear gradients (TOO LATE!)
                              This causes gradients to accumulate!
```

---

## Task 4: Training Loop

### Overall Flow

```
Initialize:
├─ Create environment (CartPole-v1)
├─ Create policy network
├─ Create optimizer (Adam with learning rate 0.001)
└─ Initialize reward tracking list

Loop 500 times (for each episode):
├─ Reset environment to initial state
├─ Initialize empty lists: log_probs = [], rewards = []
│
├─ WHILE episode not done:
│  ├─ Get current state from environment
│  ├─ Policy selects action using current probabilities
│  ├─ Take action in environment
│  ├─ Receive: new_state, reward, done flags
│  ├─ Append log_prob to log_probs
│  ├─ Append reward to rewards
│  └─ If done, exit inner loop
│
├─ Compute discounted returns from rewards list
├─ IF using baseline: Normalize returns to mean=0, std=1
├─ Call REINFORCE update with log_probs, returns, optimizer
│
├─ Log episode reward and progress every 50 episodes
└─ Continue to next episode

Close environment and plot results
```

### Detailed Implementation

```python
# Configuration
N_EPISODES = 500      # Train for 500 episodes
GAMMA = 0.99          # Discount factor (patient agent)
LEARNING_RATE = 0.001 # Adam learning rate
USE_BASELINE = True   # Enable return normalization

# Initialize
env = gym.make('CartPole-v1')
policy = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=64)
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
all_rewards = []

# Main training loop
for ep in range(N_EPISODES):

    # 1. Reset environment
    state, _ = env.reset()  # state is array [cart_pos, cart_vel, pole_angle, angle_vel]
    log_probs = []
    rewards = []

    # 2. Run one complete episode
    done = False
    while not done:

        # a) Select action
        action, log_prob = policy.select_action(state)
        # action ∈ {0, 1}, log_prob is scalar tensor

        # b) Take action, get feedback
        state, reward, terminated, truncated, _ = env.step(action)
        # state: new state (4,)
        # reward: 1 (pole still upright)
        # terminated: True if pole fell or cart out of bounds
        # truncated: True if 500 steps reached (time limit)

        # c) Record for gradient computation
        done = terminated or truncated
        log_probs.append(log_prob)
        rewards.append(reward)

    # 3. Compute returns
    # returns[t] = sum of discounted future rewards from step t onward
    returns = compute_returns(rewards, gamma=GAMMA)

    # 4. Apply baseline (optional but recommended)
    if USE_BASELINE:
        # Normalize so returns has mean 0 and std 1
        # This reduces variance in gradient estimates
        returns = normalise_returns(returns)

    # 5. Update policy
    # This is where learning happens
    loss = reinforce_update(log_probs, returns, optimizer)

    # 6. Track progress
    ep_reward = sum(rewards)
    all_rewards.append(ep_reward)

    if (ep + 1) % 50 == 0:
        avg_last_50 = np.mean(all_rewards[-50:])
        print(f'Episode {ep+1:4d} | Reward: {ep_reward:6.1f} | Avg(50): {avg_last_50:6.1f} | Loss: {loss:.4f}')

env.close()
```

### Key Points

1. **`state, _ = env.reset()`**: Returns state and info dict
2. **`env.step(action)`**: Returns (state, reward, terminated, truncated, info)
3. **Done flag**: Must check BOTH terminated (failure) AND truncated (timeout)
4. **Log probability tracking**: Needed for gradient computation, must be kept in original order
5. **Return computation**: Done AFTER full episode completes

---

## Task 5: Experiments

### Experiment A: Effect of Baseline (Return Normalization)

**Hypothesis**: Normalizing returns reduces gradient variance and stabilizes learning.

**Implementation**:

```python
def train_cartpole(use_baseline):
    # ... training loop ...

    if use_baseline:
        returns = normalise_returns(returns)  # Apply normalization

    # Returns without baseline: use raw G_t values
    # Returns with baseline: use (G_t - mean) / std
```

**What happens**:

- WITHOUT baseline: Raw returns can have large magnitude (e.g., [100, 99, 98])
  - Gradient has high variance
  - Unstable updates
- WITH baseline: Normalized returns centered at 0 (e.g., [1.2, 0.05, -1.25])
  - Smaller magnitude→ smaller gradients
  - More stable updates
  - Faster convergence

**Results**: WITH baseline should show smoother learning curve.

---

### Experiment B: Effect of Discount Factor (gamma)

**Hypothesis**: Different gamma values create different agent "personalities".

```
gamma = 0.99 (Patient agent)
├─ Considers 100+ future steps
├─ Slow learning (needs more experience)
├─ Better long-term planning
└─ Final performance likely best

gamma = 0.90 (Balanced agent)
├─ Considers ~20 future steps
├─ Moderate learning speed
└─ Reasonable balance

gamma = 0.50 (Myopic agent)
├─ Considers ~2-3 future steps
├─ Rapid initial learning
├─ Poor long-term planning
└─ Final performance limited
```

**Implementation**:

```python
returns = compute_returns(rewards, gamma=0.99)  # vs 0.90 vs 0.50
```

**Results**:

- gamma=0.99: Higher final reward but slower
- gamma=0.50: Lower final reward but faster initial progress

---

### Experiment C: Effect of Learning Rate (lr)

**Hypothesis**: Learning rate controls step size in parameter space.

```
lr = 0.0001 (Very small)
├─ Tiny gradient steps
├─ Very slow convergence
└─ May not learn in 250 episodes

lr = 0.001 (Default, good)
├─ Balanced step size
├─ Stable convergence
├─ Good final performance
└─ Recommended

lr = 0.01 (Large)
├─ Large gradient steps
├─ Fast initial progress
├─ Risk of instability/divergence
└─ May oscillate or fail to converge
```

**Implementation**:

```python
optimizer = optim.Adam(policy.parameters(), lr=0.001)  # vs 0.0001 vs 0.01
```

---

## Complete Workflow Example

### One Full Episode Walkthrough

Suppose this episode unfolds:

```
Step 0:
├─ State: [0.01, 0.02, -0.01, 0.03]
├─ Policy outputs: [0.45, 0.55]  (45% left, 55% right)
├─ Sampled action: 1 (right)
├─ Log probability: log(0.55) = -0.598
├─ Environment reward: +1
└─ New state: [0.02, 0.14, -0.02, -0.10]

Step 1:
├─ State: [0.02, 0.14, -0.02, -0.10]
├─ Policy outputs: [0.52, 0.48]  (52% left, 48% right)
├─ Sampled action: 0 (left)
├─ Log probability: log(0.52) = -0.654
├─ Environment reward: +1
└─ New state: [0.05, -0.05, -0.04, 0.15]

Step 2:
├─ State: [0.05, -0.05, -0.04, 0.15]
├─ Policy outputs: [0.49, 0.51]  (49% left, 51% right)
├─ Sampled action: 1 (right)
├─ Log probability: log(0.51) = -0.673
├─ Environment reward: +1
└─ Pole falls → episode ends (terminated=True)

Episode summary:
├─ log_probs = [-0.598, -0.654, -0.673]
└─ rewards = [1, 1, 1]
```

### Computing Returns (gamma=0.99)

```
Backward iteration:
─────────────────

G = 0, returns = []

Step 2 (last):
  G = 1 + 0.99 * 0 = 1.0
  returns = [1.0]

Step 1:
  G = 1 + 0.99 * 1.0 = 1.99
  returns = [1.99, 1.0]

Step 0 (first):
  G = 1 + 0.99 * 1.99 = 2.9701
  returns = [2.9701, 1.99, 1.0]

Final returns tensor: [2.9701, 1.99, 1.0]
```

### Normalizing Returns (if USE_BASELINE=True)

```
mean(returns) = (2.9701 + 1.99 + 1.0) / 3 = 1.987
std(returns) = sqrt(((2.9701-1.987)² + (1.99-1.987)² + (1.0-1.987)²) / 3)
             ≈ 0.807

Normalization formula: (G_t - mean) / (std + 1e-8)

G_0_norm = (2.9701 - 1.987) / 0.807 = 1.218
G_1_norm = (1.99 - 1.987) / 0.807 = 0.004
G_2_norm = (1.0 - 1.987) / 0.807 = -1.223

Normalized returns: [1.218, 0.004, -1.223]
Mean ≈ 0.0, Std ≈ 1.0 ✓
```

### Computing Loss and Gradients

```
loss = -sum(returns * log_probs)
     = -(G_0 * log_p_0 + G_1 * log_p_1 + G_2 * log_p_2)
     = -(1.218 * (-0.598) + 0.004 * (-0.654) + (-1.223) * (-0.673))
     = -(-0.728 + -0.003 + 0.823)
     = -(0.092)
     = -0.092

loss.backward() computes gradients:
├─ ∂loss/∂w for weights in fc3 (output layer)
├─ ∂loss/∂w for weights in fc2 (hidden layer 2)
└─ ∂loss/∂w for weights in fc1 (hidden layer 1)

optimizer.step() updates all weights:
w_new = w_old - learning_rate * gradient
```

### Interpretation

- **Positive actions (high G_t, high log_prob)**: Loss decreases → weights adjust to increase these action probabilities
- **Negative actions (low G_t, low log_prob)**: Loss increases → weights adjust to decrease these action probabilities
- **Result**: Over time, policy learns to take actions that yield higher returns

---

## Key Insights

### Why REINFORCE Works

1. **Gradient follows returns**: Actions leading to high returns get higher probability
2. **Stochastic exploration**: Softmax ensures some exploration (non-zero probability for all actions)
3. **Convergence**: Guaranteed convergence to local optimum (not global)

### Why Normalization Helps

- Reduces the magnitude of returns
- Centers gradient signal around zero
- Decreases variance in gradient estimates
- Makes learning more stable

### Why Order Matters

- **Backward return computation**: Need future rewards to compute present returns
- **Gradient order**: Must zero_grad before backward
- **Episode order**: Log probs and rewards must be in same time order

### Computational Complexity

- **per episode**: O(episode_length × network_size)
- **per 500 episodes**: CartPole episodes average 50-400 steps
- **Overall**: ~50,000 to 200,000 forward/backward passes

---

## Expected Training Progression

| Episode Range | Typical Reward | Policy State           | Loss          |
| ------------- | -------------- | ---------------------- | ------------- |
| 1-50          | 20-50          | Random                 | High variance |
| 51-150        | 50-100         | Learning basic balance | Decreasing    |
| 151-250       | 100-200        | Good balance           | Moderate      |
| 251-350       | 200-300        | Strong balance         | Increasing\*  |
| 351-450       | 250-350        | Near-optimal balance   | High\*        |
| 451-500       | 300-400        | Excellent balance      | Very high\*   |

\*Loss increases with episode length because more steps = more log_prob terms to sum

---

## Conclusion

The REINFORCE algorithm provides a simple yet effective way to train policies on continuous state spaces. By collecting an episode, computing returns, and using log-probability weighted by returns as the training signal, the network learns to increase the probability of good actions over time. The combination of return normalization, appropriate discount factor, and learning rate tuning enables stable convergence on CartPole.
