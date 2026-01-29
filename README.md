# Decision Making Under Uncertainty

This repository contains interactive Jupyter notebooks covering the mathematical foundations of probabilistic reasoning and decision-making. It progresses from basic probability theory through inference and learning to sequential decision problems.

**What you'll learn:**
- How to represent and reason about uncertainty using probabilistic graphical models
- Exact and approximate inference algorithms for computing probability distributions
- Methods for learning model parameters and structure from data
- Decision theory and utility functions for optimal action selection
- Sequential decision making with Markov Decision Processes

The notebooks build progressively, starting with foundational concepts and advancing to sophisticated algorithms used in modern AI systems. Each chapter includes both mathematical derivations and practical Python implementations.

**Prerequisites:**
- Basic probability and statistics
- Linear algebra (vectors, matrices)
- Calculus (derivatives, integrals)
- Python programming

**Structure:**
- **Chapter 2**: Probability foundations, factors, and graphical models
- **Chapter 3**: Inference algorithms (exact and sampling-based)
- **Chapter 4**: Learning parameters from data
- **Chapter 5**: Learning graphical model structure
- **Chapter 6**: Decision theory and utility
- **Chapter 7**: Sequential decision making with MDPs

---


## Chapter 2

This chapter covers the basics of probability theory.

---

### 1 - Rules of Probability, Factors and Graphs

This notebook introduces the Kolmogorov axioms and fundamental probability laws, then shows how to represent probability distributions as **factors**.

The notebook covers:
- **Factor operations**: conditioning, marginalization, multiplication
- **Directed graphical models** (Bayesian networks) that encode conditional independence
- **Markov blanket**: each variable is independent of all others given its parents, children, and children's parents

---

### 2 - Common Distributions

This notebook catalogs some important distributions for machine learning and decision theory.

**Discrete distributions:**
- **Bernoulli/Binomial**: Binary outcomes (coin flips)

<p align="center">
  <img src="images/Chapter 2/2, common distributions_cell5_img1.png" alt="Binomial distribution" width="400"/>
</p>

- **Categorical/Multinomial**: Multiple discrete outcomes (dice rolls)

**Continuous distributions:**
- **Uniform**: Constant probability over an interval
- **Gaussian**: The ubiquitous bell curve

<p align="center">
  <img src="images/Chapter 2/2, common distributions_cell12_img1.png" alt="Gaussian distribution" width="400"/>
</p>

**Advanced models:**
- **Mixture of Gaussians**: Combines multiple Gaussian components, useful for multimodal data
- **Linear Gaussian models**: Where means depend linearly on parent variables, enabling tractable inference


## Chapter 3

Inference is about computing probabilities given what we know.

---

### 1 - Exact Inference

Inference means answering queries like "What's the probability of Y given what we know about X?"

$$P(Y|X_{\text{known}}) = \sum_{X_{\text{unknown}}}P(Y,X_{\text{unknown}}|X_{\text{known}})$$

We condition on known variables and marginalize out the unknowns. With factors, this is straightforward:
1. Condition on known variables (fix their values)
2. Marginalize out irrelevant variables (sum over their values)
3. Normalize the result

The **sum-product variable elimination** algorithm avoids building the full joint table by carefully ordering operations on factors. This exploits the conditional independence structure.

---

### 2 - Inference, Integration with Samples

In many cases we want to calculate an integral of a continuous function. When exact inference is intractable, we can use sampling methods.

The basic **Monte Carlo** formula approximates with:
$$\int f(x)p(x) dx \approx \frac{1}{N}\sum_{n=1}^N f(x_n)$$

**Importance sampling** handles cases where we can't sample directly from the distribution. We sample from a proposal distribution q(x) and reweight. This is particularly useful when we only know an unnormalized version of p(x).

---

### 3 - Sampling Factors

This notebook covers algorithms for drawing samples from Bayesian networks, such as **rejection sampling**, **likelihood weighting**, and **Gibbs sampling**.

---

### 4 - Inference with Gaussians

Gaussians have special properties that make inference tractable in closed form.

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <img src="images/Chapter 3/4, Inference with Gaussians_cell5_img1.png" alt="Original Gaussian" width="400"/>
  <img src="images/Chapter 3/4, Inference with Gaussians_cell7_img1.png" alt="Conditioned Gaussian" width="400"/>
</div>

The conditional mean and covariance have closed-form expressions.


## Chapter 4

This chapter covers how to learn model parameters from data.

---

### 1 - Maximum Likelihood Estimate

The Maximum Likelihood Estimate (MLE) chooses parameters that maximize the probability of the observed data.

For a **Bernoulli distribution** (coin flips), the MLE is simply the sample mean. For a **Gaussian**, it's the sample mean and variance. For **Bayesian networks**, we can estimate each conditional probability table independently by counting frequencies in the data.

MLE is simple and works well with lots of data, but can overfit with small datasets and doesn't quantify uncertainty in the parameters.

---

### 2 - Bayesian Parameter Learning

Rather than picking a single "best" parameter, Bayesian learning maintains a full distribution over parameters. We start with a **prior** encoding our beliefs, then update to a **posterior** after seeing data.

**Conjugate priors** make this tractable. For a Bernoulli likelihood, the **Beta distribution** is conjugate:

<p align="center">
  <img src="images/Chapter 4/2, Bayesian Parameter Learning_cell9_img1.png" alt="Beta prior" width="400"/>
</p>

After observing data, the posterior is also Beta with updated parameters:

<p align="center">
  <img src="images/Chapter 4/2, Bayesian Parameter Learning_cell10_img1.png" alt="Beta posterior" width="400"/>
</p>

For categorical data, the **Dirichlet distribution** is the multi-dimensional analog of the Beta.

<p align="center">
  <img src="images/Chapter 4/2, Bayesian Parameter Learning_cell17_img1.png" alt="Dirichlet distribution" width="500"/>
</p>

**Maximum A Posteriori (MAP)** estimates choose the most probable parameter:
$$\theta_{\text{MAP}} = \arg\max_\theta P(\theta|D)$$

This is a compromise between full Bayesian inference and MLE, often used for computational efficiency.
. ## 3 - Non-Parametric Models

Parametric models assume data comes from a specific family (Gaussian, etc.). **Non-parametric models** are more flexible.

**Kernel Density Estimation (KDE)** places a kernel function (e.g., Gaussian) at each data point:

$$p(x) = \frac{1}{N}\sum_{i=1}^N K_\sigma(x - x_i)$$
. The bandwidthimages/Chapter 4/3, Non-Parametric Models_cell5_img1.png" alt="KDE small bandwidth" width="300"/>
  <img src="images/Chapter 4/3, Non-Parametric Models_cell5_img2.png" alt="KDE medium bandwidth" width="300"/>
  <img src="images/Chapter 4/3, Non-Parametric Models_cell5_img3.png" alt="KDE large bandwidth" width="300"/>
</p>
<p align="center"><em>Kernel density estimates with small, medium, and large bandwidths.</em></p>

Small σ captures fine details but can overfit. Large σ is smoother but may miss structure. The choice of bandwidth is crucial.

---

### 4 - Learning With Missing Data

Real datasets often have missing values. Several strategies exist:

**Simple imputation:**
- Fill with mean, median, or mode
- Fast but ignores uncertainty

**Model-based imputation:**
- Fit a model (e.g., Gaussian)
- Sample missing values from the conditional distribution

<p align="center">
  <img src="images/Chapter 4/4, Learning With Missing Data_cell5_img1.png" alt="Original data" width="350"/>
  <img src="images/Chapter 4/4, Learning With Missing Data_cell11_img1.png" alt="Model-based imputation" width="350"/>
</p>

**K-nearest neighbors:**
- Fill missing values using similar complete data points
- Works well when similar examples exist

<p align="center">
  <img src="images/Chapter 4/4, Learning With Missing Data_cell5_img1.png" alt="Original data" width="350"/>
  <img src="images/Chapter 4/4, Learning With Missing Data_cell14_img1.png" alt="Model-based imputation" width="350"/>
</p>


**Full Bayesian approach:**
- Treat missing values as latent variables and integrate them out
- Principled but computationally expensive

---

### 5 - The EM Algorithm

The **Expectation-Maximization (EM)** algorithm handles learning with latent (hidden) variables or missing data.

EM alternates between two steps:
1. **E-step**: Infer distribution over hidden variables given current parameters
2. **M-step**: Update parameters to maximize expected likelihood

<p align="center">
  <img src="images/Chapter 4/5, The EM algorithm_cell5_img1.png" alt="EM iteration 1" width="350"/>
  <img src="images/Chapter 4/5, The EM algorithm_cell7_img1.png" alt="EM iteration 5" width="350"/>
</p>
<p align="center"><em>EM algorithm progressively improving a mixture of Gaussians fit.</em></p>

The algorithm is guaranteed to improve the likelihood at each iteration, though it may converge to a local optimum.

EM is particularly powerful for:
- **Mixture models**: Learning cluster assignments and parameters simultaneously
- **Bayesian networks with missing data**: Filling in unobserved variables while learning parameters

The key insight is that by treating hidden variables as if we knew their distribution (E-step), we can optimize parameters as if we had complete data (M-step).


## Chapter 5

Beyond learning parameters, we can learn the **structure** of the graphical model itself - which variables depend on which others.

---

### 1 - Searching PGMs

This notebook addresses the question: given data, what is the best graph structure?

The challenge is that the space of possible graphs grows super-exponentially with the number of variables. The approach is to define a **scoring function** (e.g., likelihood, BIC, or AIC) that measures how well a structure fits the data, then search through possible structures.

**Key insight:** Multiple graphs can represent the same set of conditional independencies. These form **equivalence classes**. We can't distinguish between them from data alone - only their independence structure matters.

The notebook demonstrates this by testing different 3-variable structures and comparing their likelihoods on sample data. With enough data, the true structure (or its equivalence class) should score highest.

**Challenges:**
- Computational cost of searching exponentially many structures
- Need sufficient data to distinguish structures reliably
- Regularization (via BIC/AIC penalties) to avoid overfitting with complex graphs


## Chapter 6

This chapter introduces **decision theory** - how to choose actions optimally when outcomes are uncertain.

---

### 1 - Utility

How do we make rational decisions? **Utility theory** provides the foundation.

**Von Neumann-Morgenstern Axioms:**
If preferences satisfy completeness, transitivity, continuity, and independence, then there exists a **utility function** such that we should choose actions to maximize **expected utility**.

**Risk attitudes:**
- **Risk-neutral**: Care only about expected value
- **Risk-averse**: Prefer certain outcomes over gambles with same expected value
- **Risk-seeking**: Prefer gambles over certain outcomes

For example, most people are risk-averse about money: they'd rather have $50 for sure than a 50/50 chance at $0 or $100, even though both have the same expected value.

The shape of the utility function encodes risk attitudes: concave for risk-averse, linear for risk-neutral, convex for risk-seeking. This explains why people buy insurance (pay to reduce risk) and lottery tickets (pay for small chances of big rewards).

---

### 2 - Decision Networks, Value of Information, Irrationality

**Decision networks** (influence diagrams) extend Bayesian networks with decision nodes (squares) for actions we can choose and utility nodes (diamonds) for rewards/costs.

To find the optimal decision: for each possible action, infer the resulting probability distribution over outcomes, compute expected utility, and choose the action with maximum expected utility.

**Value of Information (VOI):**

Sometimes we can gather information before deciding. The value of information is the improvement in expected utility from knowing something before you act. For example: Should you check the weather forecast before deciding whether to bring an umbrella?

The notebook demonstrates this with a bus decision problem: should you wait for the bus or walk? Checking a rain forecast has positive value if it helps you make a better decision.

**Key insights:**
- Information is never harmful (in expectation)
- Information has no value if it doesn't change your decision
- VOI can guide which sensors/experiments to use

**Irrationality:**

The notebook also touches on how real humans violate the axioms of rational decision-making, such as framing effects, the sunk cost fallacy, and overestimating rare events.


## Chapter 7

This chapter addresses **sequential decisions** - choosing actions over time when each action affects future states.

---

### 1 - Markov Decision Processes

A **Markov Decision Process (MDP)** models sequential decision making with states, actions, a transition function, and a reward function.

**Markov property:** The next state depends only on the current state and action, not the full history.

The utility of a policy can be defined for finite horizons (sum of rewards) or infinite horizons with a discount factor (making near-term rewards more valuable than distant ones).

The **Bellman equation** relates utility across time steps and is the foundation for computing optimal policies.

The notebook demonstrates this with a **maze navigation** problem where an agent navigates a grid world with stochastic transitions, positive rewards at goal locations, and negative rewards at dangerous states.

---

### 2 - Getting a Policy from a Value Function

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
  <img src="images/Chapter 7/Linear Quadratic Regulator (MDP in continuous space)_cell4_img1.png" alt="LQR system" width="500"/>
</p>

The notebook demonstrates a simple 1D system where:
- State accumulates over time
- Control actions can adjust the state
- Optimal policy balances state deviation costs against control effort
 where the dynamics are linear and the cost is quadratic.

Despite infinite state/action spaces, LQR has a **closed-form solution** where the optimal policy is linear.

<p align="center">
  <img src="images/Chapter 7/Linear Quadratic Regulator (MDP in continuous space)_cell4_img1.png" alt="LQR system" width="500"/>
</p>

The notebook demonstrates a simple 1D system where state accumulates over time and control actions can adjust it, with the optimal policy balancing state deviation costs against control effort.

<p align="center">
  <img src="images/Chapter 7/Linear Quadratic Regulator (MDP in continuous space)_cell6_img1.png" alt="LQR trajectory" width="500"/>
</p>

**Applications:**
- Robotics (motor control)
- Economics (optimal control policies)
- Aerospace (aircraft stabilization)

LQR is the foundation for more sophisticated continuous control methods like iterative LQR and model predictive control
