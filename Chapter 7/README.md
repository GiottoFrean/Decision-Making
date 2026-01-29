# Sequential Decision Making

The final chapter addresses **sequential decisions** - choosing actions over time when each action affects future states.

---

## 1 - Markov Decision Processes

A **Markov Decision Process (MDP)** models sequential decision making with:
- **States** S: Possible situations
- **Actions** A: Choices available
- **Transition function** T(s'|s,a): Probability of reaching s' after taking action a in state s
- **Reward function** R(s,a): Immediate reward for taking action a in state s

**Markov property:** The next state depends only on the current state and action, not the full history.

**Utility functions:**

For finite horizon (n steps):
$$U^\pi = \sum_{i=1}^n R(s_i, a_i)$$

For infinite horizon with discount λ < 1:
$$U^\pi = \sum_{i=1}^\infty \lambda^{i-1} R(s_i, a_i)$$

The discount ensures rewards are bounded and makes near-term rewards more valuable than distant ones.

**Bellman equation** relates utility across time steps:
$$U_m^\pi(s) = R(s,\pi(s)) + \lambda \sum_{s'} T(s'|s,\pi(s)) U_{m-1}^\pi(s')$$

This recursion is the foundation for computing optimal policies.

The notebook demonstrates this with a **maze navigation** problem:
- Agent navigates a grid world
- Stochastic transitions (sometimes move in unintended direction)  
- Positive rewards at goal locations
- Negative rewards (penalties) at dangerous states
- Must find path that maximizes expected cumulative reward

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

## 3 - Value Iteration (and Linear Programming)

Rather than alternating between policy evaluation and improvement, **value iteration** directly updates values:

$$U_{k+1}(s) = \max_a \left[ R(s,a) + \lambda \sum_{s'} T(s'|s,a) U_k(s') \right]$$

Start with arbitrary values (e.g., all zeros), and iterate until convergence.

**Why it works:** The Bellman update is a **contraction mapping** when λ < 1:
- It brings value estimates closer together each iteration
- Guaranteed to converge to unique fixed point (the optimal values)
- Then extract optimal policy by being greedy

**Comparison to policy iteration:**
- Value iteration: Simpler, one operation per iteration
- Policy iteration: May converge in fewer iterations
- Both find the optimal policy

**Linear programming alternative:**

The Bellman optimality equations can be formulated as a linear program:

Minimize: $\sum_s U(s)$

Subject to: $U(s) \geq R(s,a) + \lambda \sum_{s'} T(s'|s,a) U(s')$ for all s, a

This reformulation allows using standard LP solvers. However, it's typically slower than value/policy iteration for MDPs.

The notebook demonstrates all three methods on a larger maze problem, showing they all converge to the same optimal policy.

---

## 4 - Linear Quadratic Regulator (MDP in Continuous Space)

Real-world systems often have **continuous states and actions**, not discrete grids.

The **Linear Quadratic Regulator (LQR)** is a special case with tractable solutions:

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

<p align="center">
  <img src="../images/Chapter 7/Linear Quadratic Regulator (MDP in continuous space)_cell6_img1.png" alt="LQR trajectory" width="500"/>
</p>

**Applications:**
- Robotics (motor control)
- Economics (optimal control policies)
- Aerospace (aircraft stabilization)

LQR is the foundation for more sophisticated continuous control methods. Though limited to linear dynamics and quadratic costs, it provides intuition and can be extended (iterative LQR, model predictive control) to handle nonlinear systems.
