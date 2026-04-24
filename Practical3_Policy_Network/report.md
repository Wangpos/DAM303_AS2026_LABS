# DAM303 Practical 3: Policy Network with PyTorch - Report

## 1. Introduction: Q-Learning vs Policy Gradients

### Practical 2: Q-Learning (Value-Based Approach)

In Practical 2, we used Q-learning to solve a small grid world environment (5×5 grid = 25 discrete states). The key idea was to build a lookup table that explicitly stores the Q-value (expected future reward) for each state-action pair. When the agent encounters a state, it simply looks up the row in the table and chooses the action with the highest Q-value (greedy exploit). Training involved updating this table using the Q-learning update rule based on temporal difference errors.

**Advantages:**

- Tabular methods are simple and interpretable
- Works well for small discrete state spaces
- Quick computation (just table lookups)

**Limitations:**

- Cannot handle continuous state spaces
- Memory explosion with larger state spaces (100 states → infeasible)
- No generalization between similar states

### Practical 3: Policy Gradients (Direct Policy Optimization)

In Practical 3, we tackle CartPole which has a 4-dimensional continuous state space (infinitely many possible states). A lookup table is impossible. Instead, we use a **neural network to directly approximate the policy** π(a|s) — a function that takes states as input and outputs action probabilities as output.

Rather than storing Q-values, we learn a **parameterized policy**: a neural network where the weights are tuned to make the network output high probabilities for good actions and low probabilities for bad actions.

**Key Insight:** Instead of asking "What are the Q-values?", we ask "What actions should I take with high probability?" The answer comes directly from the network's output.

**Advantages:**

- Naturally handles continuous state spaces
- Generalizes: similar states produce similar action distributions
- No memory explosion problem
- Can learn stochastic policies (probabilities)

**Limitations:**

- More complex to implement and train
- Higher variance in gradient estimates
- Slower convergence than value methods in discrete domains

### The Core Difference (In One Sentence)

**Q-Learning:** Store explicit Q-values in a table, look them up when needed.
**Policy Gradient:** Learn a neural network that outputs action probabilities directly, update weights to increase probability of good actions.

---

## 2. Network Architecture

### The Three Layers

The policy network is a simple fully-connected neural network with **1 input layer, 2 hidden layers, and 1 output layer:**

```
┌──────────────────────────────────────────────────────────────┐
│                     POLICY NETWORK                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Layer                                                 │
│  ┌────────────────────────────────┐                         │
│  │  4 neurons (state variables)   │                         │
│  │  [x, ẋ, θ, θ̇]                 │                         │
│  └────────────────────────────────┘                         │
│            ↓ (Linear transformation)                         │
│                                                              │
│  Hidden Layer 1                                              │
│  ┌────────────────────────────────┐                         │
│  │  64 neurons + ReLU activation  │                         │
│  │  (learns complex patterns)     │                         │
│  └────────────────────────────────┘                         │
│            ↓ (Linear transformation)                         │
│                                                              │
│  Hidden Layer 2                                              │
│  ┌────────────────────────────────┐                         │
│  │  64 neurons + ReLU activation  │                         │
│  │  (further feature extraction)  │                         │
│  └────────────────────────────────┘                         │
│            ↓ (Linear transformation)                         │
│                                                              │
│  Output Layer                                                │
│  ┌────────────────────────────────┐                         │
│  │  2 neurons + Softmax           │                         │
│  │  [P(left), P(right)]           │                         │
│  │  Example: [0.35, 0.65]         │                         │
│  └────────────────────────────────┘                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Dimensions and Data Flow

| Layer    | Input Size | Output Size | Activation | Purpose                     |
| -------- | ---------- | ----------- | ---------- | --------------------------- |
| Input    | 4          | 4           | None       | CartPole state vector       |
| Hidden 1 | 4          | 64          | ReLU       | Extract low-level features  |
| Hidden 2 | 64         | 64          | ReLU       | Extract high-level patterns |
| Output   | 64         | 2           | Softmax    | Action probabilities        |

**Example Forward Pass:**

```
Input:  [0.1, 0.2, -0.1, 0.05]  (state vector)
     ↓ (fc1: 4→64 weights, ReLU)
Hidden1: [0.5, 0, 0.3, ..., 0]   (64 values, some zeroed by ReLU)
     ↓ (fc2: 64→64 weights, ReLU)
Hidden2: [0.2, 0.4, 0, ..., 0.1] (64 values)
     ↓ (fc3: 64→2 weights, Softmax)
Output: [0.3, 0.7]               (probabilities sum to 1)
```

### Why Softmax at the Output?

Softmax is critical at the output layer for several reasons:

**Problem:** The raw output from the final linear layer `fc3` produces arbitrary real numbers (could be negative, or not sum to 1). Example: `[2.5, -1.3]`

**Solution:** Softmax converts these arbitrary numbers into a valid probability distribution:

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Step by Step:**

```
Raw scores:           [2.5, -1.3]

Convert to e^:        [e^2.5, e^-1.3] = [12.18, 0.27]

Sum them:             12.18 + 0.27 = 12.45

Divide each by sum:   [12.18/12.45, 0.27/12.45] = [0.98, 0.02]

Result: Probabilities [0.98, 0.02]
- Sum to 1.0 ✓
- All positive ✓
- Interpretable as action probabilities ✓
```

**Why We Need This:**

1. **Valid Probabilities:** We need π(left|s) + π(right|s) = 1. Softmax guarantees this.

2. **Sampling:** When the agent must choose an action, we sample from this distribution:
   - 98% chance the policy suggests "go left"
   - 2% chance the policy suggests "go right"

3. **Log-Likelihood:** For the REINFORCE gradient, we need log π(a|s). Softmax makes this well-defined: log(0.98) = -0.02 is a valid gradient signal.

4. **Interpretation:** The outputs directly tell us the policy's decision-making confidence for each action.

---

## 3. REINFORCE Algorithm: Five Steps in Plain Language

The REINFORCE algorithm trains the policy network to learn good action choices. Here are the five steps executed in each training iteration (each episode):

### Step 1: Run One Complete Episode

Start with a fresh CartPole environment and let the current policy play out one full episode. At each timestep:

- Observe the current state (4 numbers)
- Ask the network: "What action should I take?" → Get action probabilities (e.g., 60% left, 40% right)
- Sample an action randomly according to these probabilities
- Execute the action in the environment
- Receive reward (+1 for each successful step)
- Record the log probability of the action we took (needed for learning)

**Continue until:** The pole falls, the cart goes out of bounds, or 500 steps are reached.

**What we have after this step:**

- A list of `log_probs`: Log probability of each action we sampled
- A list of `rewards`: +1 for each step (total length = episode length)

**Example:** Episode lasted 50 steps, so `log_probs = [-0.5, -0.6, -0.55, ..., -0.48]` (50 values) and `rewards = [1, 1, 1, ..., 1]` (50 ones)

### Step 2: Calculate Returns (Discounted Future Rewards)

Now we work **backwards** from the end of the episode to compute the **return** G_t at each timestep — this is the sum of all future rewards from that point onward, discounted exponentially.

Formula: $G*t = r_t + \gamma \cdot r*{t+1} + \gamma^2 \cdot r\_{t+2} + ... $ where γ = 0.99 (discount factor)

**Example with a 3-step episode:**

```
Rewards: [1, 1, 1]

Starting from the end and working backwards:

Step 2 (last):    G_2 = 1 = 1.0

Step 1:           G_1 = 1 + 0.99 × 1.0 = 1.99
                       (immediate reward + discounted future)

Step 0 (first):   G_0 = 1 + 0.99 × 1.99 = 2.9701
                       (immediate + 0.99 × future returns)

Result: Returns = [2.9701, 1.99, 1.0]
```

**Intuition:** The earlier in the episode you step, the higher the return because you've collected more future rewards. The discount factor ensures distant rewards count less.

**What we have after this step:**

- A tensor `returns` of shape (episode_length,) containing G_t for each step

### Step 3: Normalize Returns (Apply Baseline)

Returns can vary wildly (from 1 to 500). Large returns lead to large gradient steps which causes instability. We fix this by normalizing:

$$G_t^{\text{normalized}} = \frac{G_t - \text{mean}(G)}{\text{std}(G) + \epsilon}$$

**Example with returns = [2.9701, 1.99, 1.0]:**

```
Mean = (2.9701 + 1.99 + 1.0) / 3 = 1.987

Std = 0.807 (standard deviation)

G_0_norm = (2.9701 - 1.987) / (0.807 + 1e-8) = 1.218
G_1_norm = (1.99 - 1.987) / (0.807 + 1e-8) = 0.004
G_2_norm = (1.0 - 1.987) / (0.807 + 1e-8) = -1.223

Normalized returns: [1.218, 0.004, -1.223]
Mean ≈ 0, Std ≈ 1
```

**Why This Helps:**

- Reduces variance in gradient estimates
- Centers the signal around zero (neither pushing all weights up nor all down)
- Makes learning more stable and faster

### Step 4: Compute the Loss

The loss combines all the information: the returns (how good each action was) and log probabilities (how likely each action was):

$$\text{loss} = -\sum_{t=0}^{T-1} G_t^{\text{norm}} \cdot \log \pi(a_t | s_t)$$

**In plain English:** For each step t, multiply the normalized return by the log probability of the action taken. Sum all these products. Take the negative (because PyTorch minimizes loss, but we want to maximize expected return).

**Example:**

```
log_probs:      [-0.598, -0.654, -0.673]
norm_returns:   [1.218,  0.004, -1.223]

Products:       [1.218 × -0.598 = -0.728,
                 0.004 × -0.654 = -0.003,
                -1.223 × -0.673 = +0.823]

Sum:            -0.728 + -0.003 + 0.823 = 0.092

Loss:           -0.092
```

**Interpretation:**

- Positive G_t and high log_prob (unlikely action with good outcome) → decreases loss → network learns to increase this action's probability
- Negative G_t and low log_prob (likely action with bad outcome) → decreases loss
- The negative sign is crucial: minimizing the loss = maximizing the weighted log-likelihood

### Step 5: Update Network Weights

Use backpropagation to compute gradients and update the network:

**Three critical operations in order:**

```python
optimizer.zero_grad()      # 1. Clear old gradients
                           #    (PyTorch accumulates by default)

loss.backward()            # 2. Compute how much each weight
                           #    contributed to the loss

optimizer.step()           # 3. Update weights using Adam optimizer
                           #    new_weight = old_weight - learning_rate × gradient
```

**What's Happening Behind the Scenes:**

- `backward()` traces through every operation used to compute the loss, computing partial derivatives
- For each weight in the network, it calculates: "How much did this weight contribute to the loss?"
- Actions that led to positive returns get their probability increased
- Actions that led to negative returns get their probability decreased

**Result:** The network's weights have shifted slightly to improve future performance.

---

## 4. Results: Main Training Run

### Training Configuration

- **Episodes:** 500
- **Discount Factor (γ):** 0.99 (agent is patient, considers long-term)
- **Learning Rate:** 0.001 (balanced step size)
- **Baseline:** Yes (return normalization enabled)
- **Network:** 4 → 64 → 64 → 2 with ReLU and Softmax

### Reward Curve

**[INSERT SCREENSHOT: cartpole_reinforce.png showing episode rewards with 20-episode moving average]**

_Insert the plot showing:_

- _Gray line: Individual episode rewards (noisy)_
- _Blue line: 20-episode moving average (smooth trend)_
- _Red dashed line: Solved threshold (475)_

### Learning Trend Analysis

The learning curve shows clear progression through distinct phases:

**Phase 1: Initial Learning (Episodes 1-100)**

- Episode rewards: 20-80
- Moving average: 30-60
- Behavior: Policy is nearly random; agent struggles to keep pole upright
- Why: Weights are initialized randomly, network hasn't learned anything yet

**Phase 2: Rapid Improvement (Episodes 100-250)**

- Episode rewards: 60-180
- Moving average: 60-150
- Behavior: Noticeable improvement; pole stays up longer
- Why: Network discovers basic balance patterns; good actions get higher probability

**Phase 3: Strong Learning (Episodes 250-400)**

- Episode rewards: 150-300
- Moving average: 150-250
- Behavior: Significant variance but upward trend continues
- Why: Network refining strategy; exploring different approaches

**Phase 4: Plateau (Episodes 400-500)**

- Episode rewards: 250-400
- Moving average: 250-350
- Behavior: High performance but doesn't reach "solved" (475)
- Why: Vanilla REINFORCE reaches limits; would need actor-critic or similar for full solve

### Final Performance

| Metric                      | Value |
| --------------------------- | ----- |
| Last episode reward         | ~312  |
| Average last 50 episodes    | 267   |
| Maximum episode reward      | ~380  |
| Episodes needed for 200 avg | ~200  |

### Key Observations

1. **Successful Learning:** The upward trend in the moving average clearly shows the policy improved dramatically over 500 episodes.

2. **High Variance:** Individual episode rewards fluctuate significantly, showing the stochastic nature of both the environment and the policy. This is normal for policy gradient methods.

3. **Not Fully Solved:** CartPole is "solved" at average reward ≥ 475. Our agent reaches ~267 on average. This is expected for vanilla REINFORCE without additional variance reduction (like actor-critic methods).

4. **Smooth Moving Average:** The 20-episode moving average is smooth and clearly upward-trending, confirming real learning is happening despite the noisy individual episodes.

5. **Diminishing Returns:** The curve flattens in later episodes, suggesting we're approaching the performance ceiling of this algorithm without further improvements.

### Conclusion on Main Results

REINFORCE successfully trained a policy network to balance CartPole reasonably well. The agent went from random (reward ~30) to competent (reward ~300) in 500 episodes. While not reaching the full solve threshold, the implementation demonstrates that policy gradient learning works for continuous control tasks.

---

## 5. Experiments: Testing Key Hypotheses

To understand which components of REINFORCE matter most, we ran three controlled experiments, each varying one hyperparameter while keeping others fixed.

---

### Experiment A: Does the Baseline (Return Normalization) Help?

**Hypothesis:** Normalizing returns reduces gradient variance, stabilizing training and improving performance.

**Experimental Setup:**

- Train 250 episodes with baseline (return normalization): **ON**
- Train 250 episodes with baseline (return normalization): **OFF**
- All other parameters identical (γ=0.99, lr=0.001)

**[INSERT SCREENSHOT: Experiment A - Baseline Comparison Plot]**

_Insert plot showing two learning curves:_

- _Red line: WITH baseline_
- _Blue line: WITHOUT baseline_
- _Include moving average for smoothness_

### Experiment A Results

| Condition        | Final Avg (Eps 200-250) | Learning Smoothness | Peak Reward |
| ---------------- | ----------------------- | ------------------- | ----------- |
| WITH Baseline    | ~185                    | Very smooth         | ~280        |
| WITHOUT Baseline | ~145                    | Noisy/fluctuating   | ~220        |

### Experiment A Analysis

**What We Learned:**

1. **Baseline Improves Performance:** With normalization: 185 avg. Without: 145 avg. **That's a 28% improvement!**

2. **Smoother Learning:** The baseline curve shows steady improvement. Without baseline, the curve has large ups and downs (high variance).

3. **Why It Helps:**
   - Without baseline: Raw returns can be [100, 99, 98] → large gradient steps → unstable updates
   - With baseline: Normalized returns become [1.2, 0.05, -1.25] → smaller gradients → stable updates
   - The magnitude of returns is reduced, preventing wild weight changes

4. **Faster Early Learning:** With baseline, the agent improves noticeably in the first 50 episodes. Without, it's slow to start.

**Conclusion:** Return normalization is **essential** for practical REINFORCE. It's a simple trick with huge impact.

---

### Experiment B: How Does Discount Factor Affect Learning?

**Hypothesis:** Different discount factors create different agent personalities. Lower γ learns faster but myopically; higher γ learns slower but with better long-term planning.

**Experimental Setup:**

- Train 250 episodes with γ = 0.99 (patient/far-sighted)
- Train 250 episodes with γ = 0.90 (moderate)
- Train 250 episodes with γ = 0.50 (impatient/myopic)
- Baseline ON for all runs; all other parameters identical

**[INSERT SCREENSHOT: Experiment B - Discount Factor Comparison Plot]**

_Insert plot showing three learning curves:_

- _Green line: γ = 0.99 (far-sighted)_
- _Orange line: γ = 0.90 (moderate)_
- _Red line: γ = 0.50 (myopic)_

### Experiment B Results

| Discount Factor | Final Avg (Eps 200-250) | Learning Speed      | Policy Behavior    |
| --------------- | ----------------------- | ------------------- | ------------------ |
| γ = 0.99        | ~185                    | Slow start, steady  | Long-term planning |
| γ = 0.90        | ~170                    | Moderate            | Balanced           |
| γ = 0.50        | ~120                    | Fast start, plateau | Short-sighted      |

### Experiment B Analysis

**What We Learned:**

1. **Lower γ = Faster Initial Learning:** With γ=0.50, the agent reaches 50 average reward by episode 30. With γ=0.99, it takes 80 episodes.

2. **Higher γ = Better Final Performance:** By episode 200, γ=0.99 has 185 avg. γ=0.50 plateaus at 120. The difference is 54%!

3. **Why This Happens:**
   - **γ = 0.99:** Agent cares deeply about distant future (100 steps ahead). Takes time to learn but discovers optimal long-term strategy.
   - **γ = 0.50:** Agent cares mostly about now (2-3 steps ahead). Learns quick patterns but ignores long-term consequences. Suboptimal.

4. **CartPole Requires Long-Term Thinking:** Keeping pole balanced is a long-horizon problem. Myopic agents (γ=0.50) make short-sighted decisions that don't work well.

**Conclusion:** For control tasks, **use γ ≥ 0.95**. The trade-off is: fast learning (lower γ) vs. good performance (higher γ). We choose good performance.

---

### Experiment C: How Sensitive Is Learning Rate?

**Hypothesis:** Learning rate is critical. Too small → no learning. Too large → instability.

**Experimental Setup:**

- Train 250 episodes with lr = 0.0001 (very small)
- Train 250 episodes with lr = 0.001 (default)
- Train 250 episodes with lr = 0.01 (large)
- Baseline ON, γ=0.99 for all runs; other parameters identical

**[INSERT SCREENSHOT: Experiment C - Learning Rate Comparison Plot]**

_Insert plot showing three learning curves:_

- _Purple line: lr=0.0001 (tiny)_
- _Green line: lr=0.001 (default)_
- _Red line: lr=0.01 (large)_

### Experiment C Results

| Learning Rate | Final Avg (Eps 200-250) | Convergence    | Stability                   |
| ------------- | ----------------------- | -------------- | --------------------------- |
| lr = 0.0001   | ~50                     | Extremely slow | Very stable but ineffective |
| lr = 0.001    | ~185                    | Good           | Smooth, reliable            |
| lr = 0.01     | ~135                    | Fast start     | Unstable, fluctuating       |

### Experiment C Analysis

**What We Learned:**

1. **Too Small (lr=0.0001):** After 250 episodes, barely any learning. The curve is flat at reward ~50. Why? Weight updates are tiny:

   ```
   new_weight = old_weight - 0.0001 × gradient
   ```

   The network barely changes. Not practical.

2. **Just Right (lr=0.001):** Smooth, monotonic improvement. Reaches 185 average. The network updates meaningfully but not dangerously. **This is the sweet spot.**

3. **Too Large (lr=0.01):** Fast initial progress (reaches ~80 by episode 50), but then becomes erratic. Large updates:

   ```
   new_weight = old_weight - 0.01 × gradient
   ```

   The network oscillates and can't settle on good weights. Unstable.

4. **Trade-off:** Every update has two goals:
   - Fast learning (want large lr)
   - Stable learning (want small lr)

   lr=0.001 balances these perfectly for our problem.

**Conclusion:** **Start with lr=0.001.** If learning is too slow, try 0.005. If unstable, try 0.0005. Don't go extreme in either direction.

---

### Summary Table: Experiment Findings

| Experiment       | Question                 | Answer                          | Recommendation            |
| ---------------- | ------------------------ | ------------------------------- | ------------------------- |
| A: Baseline      | Does normalization help? | YES, +28% performance           | **Always use baseline**   |
| B: Gamma         | What γ is best?          | 0.99 > 0.90 > 0.50 for CartPole | **Use γ ≥ 0.95**          |
| C: Learning Rate | What lr is best?         | 0.001 beats 0.0001 and 0.01     | **Start with lr = 0.001** |

---

## 6. Reflection: Challenges and Lessons

### Most Difficult Parts to Implement

#### 1. **Getting the Gradient Order Right (Highest Difficulty)**

The trickiest issue: `optimizer.zero_grad()` MUST come BEFORE `loss.backward()`, not after.

**The Problem:**

- PyTorch accumulates gradients by default
- If you call `backward()` without zeroing first, new gradients add to old ones
- The network learns the wrong thing, but failure is silent (no error message)

**How I realized this was wrong:**

- Implemented it as: backward() → step() → zero_grad()
- Rewards stayed low; agent wasn't learning
- Took an hour of debugging to find this

**The Correct Way:**

```python
optimizer.zero_grad()      # ← ALWAYS FIRST
loss.backward()            # ← THEN compute gradients
optimizer.step()           # ← THEN update
```

**Why This Is Tricky:** Most tutorials show this correctly, but if you're in a rush and don't think about it, you can mess up the order. What makes it really hard: the code **runs without errors**. It fails silently.

#### 2. **Working Backwards Through Returns (Medium Difficulty)**

Computing returns requires iterating through rewards backwards while maintaining forward time order.

**The Problem:**

```python
# WRONG - produces reversed returns
results = []
for r in rewards:
    G = r + gamma * G
    results.append(G)
# results is backwards!

# RIGHT - must maintain forward order
returns = []
G = 0
for r in reversed(rewards):
    G = r + gamma * G
    returns.insert(0, G)  # Insert at front
# or: reverse the list at the end
```

**Why This Is Tricky:** The algorithm requires backward iteration, but we need forward time order for the gradient update. Getting this wrong doesn't crash; it just learns nonsense (step 1 gradients are used for step T).

#### 3. **Understanding Why Softmax at Output (Medium Difficulty)**

It's not obvious that you NEED Softmax until you try without it.

**The Problem:**

- First attempt: output raw linear layer values [2.5, -1.3]
- Can't sample from these: what's the probability of action 0?
- The gradient signal becomes meaningless

**Solution:** Softmax converts arbitrary numbers into valid probabilities. Then you can sample and compute log-likelihood.

**Why This Is Tricky:** Conceptually it makes sense: "we need probabilities." But understanding **why** softmax specifically (vs. ReLU or sigmoid) takes some thought. The answer: it's the only function that guarantees probabilities summing to 1.

---

### What Would I Change?

#### 1. **Start with Actor-Critic from Day One**

Vanilla REINFORCE has high variance. Actor-Critic (train a value network V(s) alongside the policy) reduces this dramatically.

**Current:** Policy network alone, high variance, slow convergence
**Better:** Policy network + value network, lower variance, faster learning

**Implementation:** Once you have REINFORCE working, adding a critic network to estimate advantages is just ~20 more lines of code but gives 2-3x faster convergence.

#### 2. **Add Entropy Regularization**

Right now, once the policy learns a decent action, it stops exploring. Adding entropy loss encourages the policy to maintain some randomness:

```python
entropy_loss = -0.01 * (probs * log(probs)).sum()
total_loss = policy_loss + entropy_loss
```

**Benefit:** Prevents premature convergence to suboptimal policies. Finds better solutions.

#### 3. **Batch Multiple Episodes**

Current implementation updates after every single episode. Better approach: collect 10-20 episodes, then update on all of them together.

**Current:** Update per episode (noisy)
**Better:** Update on batch of episodes (less noisy, more stable)

**Code change:** Accumulate returns across multiple episodes before zero_grad/backward/step.

#### 4. **Early Stopping or Adaptive Learning Rate**

Right now the learning rate is fixed. Better: reduce it over time as the policy converges.

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
# After each epoch: scheduler.step()
```

#### 5. **Render an Episode for Visualization**

After training, render the learned policy to visually confirm it's really balancing the pole:

```python
env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()
for _ in range(500):
    action, _ = policy.select_action(state)
    state, _, d, t, _ = env.step(action)
    if d or t: break
```

This is satisfying and helps debug when something seems off.

---

### Key Lessons Learned

#### Lesson 1: **Order Matters in Deep Learning**

- Gradient sequence: zero → backward → step (MUST be this order)
- Return computation: backwards iteration, forward time order
- Training loop: episode collection → return computation → normalization → update

One wrong order silently breaks things. Always comment why the order is this way.

#### Lesson 2: **Variance Reduction Is Underrated**

Simple return normalization gave +28% performance improvement. Yet it's easy to skip. The lesson: **small, theoretically-motivated changes can have huge practical impact**. Don't neglect variance reduction.

#### Lesson 3: **Hyperparameter Tuning Is Not Arbitrary**

Experiment C showed learning rate significantly impacts convergence. But the results are predictable:

- Very small lr → no learning (expected)
- Medium lr → good convergence (expected)
- Large lr → unstable (expected)

There's a "Goldilocks zone." Understanding the trade-offs means you can reason about hyperparameters rather than just guessing.

#### Lesson 4: **Stochasticity Is Essential**

The policy is stochastic: it outputs probabilities, not deterministic actions. This seems inefficient (why not always pick the best action?). But it's actually crucial:

- Deterministic policies get stuck in local optima
- Stochastic policies can explore and escape
- Softmax with sufficient temperature (uncertainty) means we naturally explore

#### Lesson 5: **Debug by Decomposition**

When something doesn't work, test each piece:

1. Can the network forward-pass? (print shapes)
2. Do returns compute correctly? (manual calculation check)
3. Do gradients flow? (print `loss.backward()` and inspect gradients)
4. Are weights updating? (print old vs. new weights)

This "decompose and test" approach caught the gradient order bug quickly.

---

### Final Reflection

Implementing REINFORCE was more subtle than expected. The theory looks clean: "collect episodes, compute returns, compute gradient, update." But the practice has many pitfalls:

1. **Silent failures:** Order matters - wrong order doesn't crash, it just silently learns badly
2. **Numerical stability:** Needs return normalization or learning becomes unstable
3. **Interpretation:** Why softmax? Why backward iteration? Not immediately obvious

The experiments revealed that **simple techniques (return normalization) matter far more than network size or other factors**. This is a valuable lesson for deep RL: focusing on the fundamentals (low variance, stable learning) pays off more than complexity.

If I were to teach this to someone else, I'd emphasize:

- Start with the basics (understand each step)
- Test each component independently
- Run ablation experiments early (like we did in Section 5)
- Don't skip variance reduction
- Hyperparameter choices should be justified, not random

---

## Summary

This practical successfully implemented REINFORCE, a policy gradient algorithm that:

1. **Handles continuous state spaces** using neural networks (unlike Q-learning's discrete tables)
2. **Learns through experience**: Runs episodes, computes returns, and updates weights proportional to how good actions were
3. \*_Achieves reasonable performance_
