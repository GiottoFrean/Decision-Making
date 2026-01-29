# Sequential Decision Making

This chapter addresses sequential decisions.

---

## 1 - Markov Decision Processes

A Markov Decision Process (MDP) models sequential decision making with states, actions, a transition function, and a reward function. The next state depends only on the current state and action, not the full history.

The utility of a policy can be defined for finite horizons (sum of rewards) or infinite horizons with a discount factor (making near-term rewards more valuable than distant ones).

The Bellman equation relates utility across time steps and is the foundation for computing optimal policies.

The notebook demonstrates this with a maze navigation problem.

---

## 2 - Getting a Policy from a Value Function

A policy specifies which action to take in each state. Given a value function, the optimal policy chooses the action that maximizes expected future reward.

**Policy iteration** alternates between:

1. Evaluation: Compute value function for current policy
2. Improvement: Update policy to be greedy with respect to values

Policy iteration is guaranteed to converge to the optimal policy in finite iterations because:
- There are finitely many policies
- Each iteration strictly improves the policy (unless already optimal)
- We can't cycle since we only move to strictly better policies

**Value iteration** is an alternative that combines evaluation and improvement in one step.

Start with arbitrary values (e.g., all zeros), and iterate until convergence. The Bellman update is a contraction mapping that's guaranteed to converge to the optimal values, then we extract the optimal policy by being greedy.

Comparison to policy iteration:
- Value iteration: Simpler, one operation per iteration
- Policy iteration: May converge in fewer iterations
- Both find the optimal policy