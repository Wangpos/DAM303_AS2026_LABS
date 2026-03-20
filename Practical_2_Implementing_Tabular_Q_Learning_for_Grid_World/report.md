# DAM303: Deep Reinforcement Learning

Software Engineering Department

## Practical Report 2

**Submitted by**
Name: Tshering Wangpo Dorji  
Student No: 02230311

Royal University of Bhutan, College of Science and Technology

## 1. Introduction

This practical focuses on implementing Tabular Q-Learning in a 4x4 Grid World environment. The core objective is to understand how an agent can learn a good decision policy through repeated interaction with an environment, without being explicitly told the best path.

In this task, the agent starts at state 0 and aims to reach the goal state 15. A penalty state (state 9) is introduced to make the problem more realistic by forcing the agent to avoid risky moves. Through training, the agent updates a Q-table that stores the expected long-term reward of taking each action in each state.

This practical is important because it builds a strong foundation for Deep Reinforcement Learning by first understanding the tabular approach clearly and intuitively.

## 2. Methodology

The implementation was done using Python and NumPy. The environment was modeled as a 4x4 grid with 16 states and 4 actions:

- `0` = Up
- `1` = Down
- `2` = Left
- `3` = Right

The Q-table was initialized with zeros of shape `(16, 4)`. During training, the following components were used:

1. Reward Function:

- `+10` for reaching the goal state (15)
- `-5` for entering the penalty state (9)
- `-0.1` for all other states

2. Action Selection Policy (Epsilon-Greedy):

- With probability $\epsilon$, a random action was selected (exploration).
- Otherwise, the action with maximum Q-value was selected (exploitation).

3. Q-Value Update (Bellman Equation):

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \big[r + \gamma \max_{a'}Q(s', a') - Q(s, a)\big]
$$

4. Hyperparameters:

- Learning rate ($\alpha$): `0.1`
- Discount factor ($\gamma$): `0.99`
- Initial epsilon: `1.0`
- Epsilon decay: `0.995`
- Minimum epsilon: `0.01`
- Episodes: `500`

After each episode, epsilon was decayed and bounded below by `epsilon_min` to maintain limited exploration.

## 3. Results

After training, the following output was generated:

```text
Training complete.
Learned Q-Table:
[[ 8.67  9.02  8.81  8.03]
 [ 5.04  5.41  5.29  8.88]
 [ 6.81  9.34  5.32  3.98]
 [ 1.16  6.51  3.42  3.02]
 [ 8.58  9.21  8.83  8.71]
 [ 5.26  0.92  5.29  9.34]
 [ 7.21  9.6   7.11  6.37]
 [ 1.69  9.03  5.93  4.37]
 [ 8.63  9.41  8.98  4.31]
 [ 7.04  9.6   7.22  7.88]
 [ 8.37  9.8   3.33  9.1 ]
 [ 4.43  9.93  6.66  7.07]
 [ 9.07  9.34  9.3   9.6 ]
 [ 4.33  9.53  9.28  9.8 ]
 [ 9.5   9.79  9.55 10.  ]
 [ 0.    0.    0.    0.  ]]

Greedy path: [0, 4, 8, 12, 13, 14, 15]
```

The greedy path successfully reaches the goal state. The learned values in the Q-table are generally higher along actions that move the agent toward the goal while avoiding unnecessary penalties.

## 4. Observations

Several meaningful observations were made from this practical:

1. Learning Progress:
   At the beginning, actions were mostly random due to high epsilon. Over time, as epsilon decayed, the agent increasingly preferred better actions.

2. Policy Formation:
   The Q-table clearly captured useful directional preferences from most states. This indicates that the agent learned a coherent navigation strategy.

3. Penalty Awareness:
   Because the penalty state had a strong negative reward, the agent learned to avoid risky transitions that could reduce total return.

4. Efficient Route:
   The final greedy path reached the goal in a short sequence of moves: `0 -> 4 -> 8 -> 12 -> 13 -> 14 -> 15`, showing practical convergence.

## 8. Reflection Questions

### 1. What happens to the agent's behaviour as epsilon decreases over time? Explain the trade-off between exploration and exploitation.

As epsilon decreases, the agent gradually shifts from exploration to exploitation. In the early training phase, a high epsilon value causes frequent random action selection, which helps the agent discover different states, rewards, and penalties. This broad search is important because the agent initially has no reliable knowledge.

Later, when epsilon becomes smaller, the agent relies more on the highest Q-values already learned. At this stage, behavior becomes more stable and goal-directed. The trade-off is that exploration helps discover better strategies, while exploitation helps maximize reward using known strategies. Too much exploration slows convergence; too much exploitation too early can trap the agent in suboptimal behavior.

### 2. Why do we initialise the Q-table with zeros? Would it matter if we used a different initial value such as a large positive number?

Initializing the Q-table with zeros is a neutral and simple starting point. It assumes no prior preference for any action in any state and lets learning be driven by actual experience.

Yes, the initial values do matter. If we initialize with very large positive numbers, the agent may become overly optimistic and may take longer to correct unrealistic estimates. In some cases, optimistic initialization can encourage extra exploration, but it can also slow stabilization of Q-values. For this practical, zero initialization is appropriate because it is stable, interpretable, and commonly used for baseline tabular Q-learning.

### 3. After training, what does the Q-table represent? How could you extract the agent's learned policy from it?

After training, the Q-table represents the estimated long-term cumulative reward for each state-action pair, assuming continued learning under the update rule. In other words, each entry $Q(s,a)$ indicates how good it is to take action $a$ from state $s$.

The learned policy can be extracted by selecting the best action in each state using:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

Practically, this is implemented with `np.argmax(Q[state])`. Repeating this from the start state produces the greedy path, which in this experiment reached the goal as `[0, 4, 8, 12, 13, 14, 15]`.


## 5. Conclusion

This practical successfully demonstrated the implementation of Tabular Q-Learning for Grid World navigation. The agent was able to learn a goal-directed policy through iterative reward-based updates and epsilon-greedy exploration.

The final Q-table and greedy-path evaluation confirm that the model learned meaningful action values and can reliably reach the goal state. Overall, this practical provided a clear and practical understanding of reinforcement learning fundamentals that can be extended to larger and more complex environments in future work.