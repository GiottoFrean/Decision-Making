# Sequential Decision Making

This chapter addresses **sequential decisions** - choosing actions over time when each action affects future states.

---

## 1 - Markov Decision Processes

A **Markov Decision Process (MDP)** models sequential decision making with states, actions, a transition function, and a reward function.

**Markov property:** The next state depends only on the current state and action, not the full history.

The utility of a policy can be defined for finite horizons (sum of rewards) or infinite horizons with a discount factor (making near-term rewards more valuable than distant ones).

The **Bellman equation** relates utility across time steps and is the foundation for computing optimal policies.

The notebook demonstrates this with a **maze navigation** problem where an agent navigates a grid world with stochastic transitions, positive rewards at goal locations, and negative rewards at dangerous states.

---

## 2 - Getting a Policy from a Value Function

A **policy** π(s) specifies which action to take in each state. Given a value function U(s), the optimal policy is:

$$\pi^*(s) = \arg\max_a \left[ R(s,a) + \lambda \sum_{s'} T(s'|s,a) U^*(s') \right]$$

**Policy iteration** alternates between:

1. **Policy evaluation**: Compute value function for current policy
   - Solve system of linear equations or iterate Bellman equation
   
2. **Policy improvement**: Update policy to be greedy with respect to values
   - For each state, choose action that maximizes expected value

**Convergence:** Policy iteration is guaranteed to converge to the optimal policy in finite iterations because:
- There are finitely many policies
- Each iteration strictly improves the policy (unless already optimal)
- We can't cycle since we only move to strictly better policies

The notebook shows this works for both finite and infinite horizon problems. Starting from a random policy, iteration quickly converges to optimal behavior.

---
specifies which action to take in each state. Given a value function, the optimal policy chooses the action that maximizes expected future reward.

**Policy iteration** alternates between:
 using the Bellman optimality equation.

Start with arbitrary values (e.g., all zeros), and iterate until convergence. The Bellman update is a **contraction mapping** that's guaranteed to converge to the optimal values, then we extract the optimal policy by being greedy.

**Comparison to policy iteration:**
- Value iteration: Simpler, one operation per iteration
- Policy iteration: May converge in fewer iterations
- Both find the optimal policy

**Linear programming alternative:**

The Bellman optimality equations can also be formulated as a linear program. This allows using standard LP solvers, thoughth tractable solutions:

**Linear dynamics:**
$$s_{t+1} = T_s \cdot s_t + T_a \cdot a_t + w_t$$
where w is Gaussian noise

**Quadratic reward:**
$$R(s,a) = -s^T R_s s - a^T R_a a$$
(Negative to represent cost; want states and actions near zero)

Despite infinite state/action spaces, LQR has a **closed-form solution**:
- Optimal policy is linear: $a^* = K \cdot s$ for some matrix K
- Value function is quadratic
- Can solve via dynamic programming or solving Riccati equations

<p align="center">
  <img src="../images/Chapter 7/Linear Quadratic Regulator (MDP in continuous space)_cell4_img1.png" alt="LQR system" width="500"/>
</p>

The notebook demonstrates a simple 1D system where:
- State accumulates over time
- Control actions can adjust the state
- Optimal policy balances state deviation costs against control effort
 where the dynamics are linear and the cost is quadratic.

Despite infinite state/action spaces, LQR has a **closed-form solution** where the optimal policy is linear.

<p align="center">
  <img src="../images/Chapter 7/Linear Quadratic Regulator (MDP in continuous space)_cell4_img1.png" alt="LQR system" width="500"/>
</p>

The notebook demonstrates a simple 1D system where state accumulates over time and control actions can adjust it, with the optimal policy balancing state deviation costs against control effort.

<p align="center">
  <img src="../images/Chapter 7/Linear Quadratic Regulator (MDP in continuous space)_cell6_img1.png" alt="LQR trajectory" width="500"/>
</p>

**Applications:**
- Robotics (motor control)
- Economics (optimal control policies)
- Aerospace (aircraft stabilization)

LQR is the foundation for more sophisticated continuous control methods like iterative LQR and model predictive control