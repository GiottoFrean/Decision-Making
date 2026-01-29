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

Understanding probability is fundamental to reasoning under uncertainty. This chapter covers the mathematical foundations needed for probabilistic graphical models and decision-making.

---

### 1 - Rules of Probability, Factors and Graphs

This notebook introduces the Kolmogorov axioms and fundamental probability laws, then shows how to represent probability distributions as **factors** - table-based data structures that make computation tractable.

The key insight is that we don't need to store the full joint probability table. Instead, we can represent it as a product of smaller factors, one for each variable conditioned on its parents:

$$P(A, B, C, D) = P(A)P(B|A)P(C|A)P(D|B,C)$$

The notebook covers:
- **Factor operations**: conditioning (fixing variables), marginalization (summing out), and multiplication
- **Directed graphical models** (Bayesian networks) that encode conditional independence
- **Markov blanket**: each variable is independent of all others given its parents, children, and children's parents

These graph structures let us exploit independence assumptions to make inference computationally feasible.

---

### 2 - Common Distributions

This notebook catalogs the most important probability distributions in machine learning and decision theory.

**Discrete distributions:**
- **Bernoulli/Binomial**: Binary outcomes (coin flips)
- **Categorical/Multinomial**: Multiple discrete outcomes (dice rolls)

<p align="center">
  <img src="images/Chapter 2/2, common distributions_cell5_img1.png" alt="Binomial distribution" width="400"/>
</p>

**Continuous distributions:**
- **Uniform**: Constant probability over an interval
- **Gaussian**: The ubiquitous bell curve

<p align="center">
  <img src="images/Chapter 2/2, common distributions_cell12_img1.png" alt="Gaussian distribution" width="400"/>
</p>

**Advanced models:**
- **Mixture of Gaussians**: Combines multiple Gaussian components, useful for multimodal data

<p align="center">
  <img src="images/Chapter 2/2, common distributions_cell20_img1.png" alt="Mixture of Gaussians" width="400"/>
</p>

- **Linear Gaussian models**: Where means depend linearly on parent variables, enabling tractable inference

Understanding these distributions is essential as they form the building blocks for more complex probabilistic models.


## Chapter 3

Once we have a probabilistic model, we need **inference** - computing probability distributions over variables of interest. This chapter explores both exact and approximate inference methods.

---

### 1 - Exact Inference

Inference means answering queries like "What's the probability of Y given what we know about X?"

$$P(Y|X_{\text{known}}) = \sum_{X_{\text{unknown}}}P(Y,X_{\text{unknown}}|X_{\text{known}})$$

We condition on known variables and marginalize out the unknowns. With factors, this is straightforward:
1. Condition on known variables (fix their values)
2. Marginalize out irrelevant variables (sum over their values)
3. Normalize the result

The **sum-product variable elimination** algorithm avoids building the full joint table by carefully ordering operations on factors. This exploits the graph structure, making inference tractable even when the full joint would be impossibly large.

The notebook demonstrates this with a simple example: deciding whether to buy tea on the way home based on who's at home.

---

### 2 - Inference, Integration with Samples

When exact inference is intractable, we can use **Monte Carlo methods** - using samples to approximate integrals and expectations.

The basic Monte Carlo formula approximates an expectation:
$$E[f(x)] \approx \frac{1}{N}\sum_{i=1}^N f(x_i) \text{ where } x_i \sim p(x)$$

<p align="center">
  <img src="images/Chapter 3/2, Inference, Integration with samples_cell9_img1.png" alt="Monte Carlo convergence" width="500"/>
</p>
<p align="center"><em>Monte Carlo estimates converge to the true value as sample size increases.</em></p>

**Importance sampling** handles cases where we can't sample directly from the target distribution. We sample from a proposal distribution q(x) and reweight:

$$E_p[f(x)] \approx \sum_{i=1}^N w_i f(x_i) \text{ where } x_i \sim q(x), \; w_i = \frac{p(x_i)}{q(x_i)}$$

This is particularly useful when we only know an unnormalized version of p(x).

<p align="center">
  <img src="images/Chapter 3/2, Inference, Integration with samples_cell19_img1.png" alt="Importance sampling" width="500"/>
</p>

---

### 3 - Sampling Factors

This notebook covers algorithms for drawing samples from Bayesian networks:

**Direct sampling** (top-down sampling):
- Sample each variable in topological order conditioned on its parents
- Simple but can't incorporate evidence

**Rejection sampling**:
- Generate samples from the joint, discard those inconsistent with evidence
- Correct but wasteful when evidence is unlikely

**Likelihood weighting**:
- Sample non-evidence variables, fix evidence variables
- Weight each sample by the probability of the evidence
- More efficient than rejection sampling

**Gibbs sampling** (MCMC):
- Initialize all variables
- Repeatedly sample each variable conditioned on all others
- After burn-in, samples approximate the joint distribution
- Works even with evidence

Each method trades off simplicity, efficiency, and applicability.

---

### 4 - Inference with Gaussians

Gaussians have special properties that make inference tractable in closed form.

**Marginalization** is trivial - just extract the relevant dimensions:

<p align="center">
  <img src="images/Chapter 3/4, Inference with Gaussians_cell5_img1.png" alt="Marginalizing Gaussians" width="400"/>
</p>

**Conditioning** uses a simple formula. If we partition variables into observed (a) and unobserved (b):

$$p(x_b | x_a) = \mathcal{N}(\mu_{b|a}, \Sigma_{b|a})$$

where the conditional mean and covariance have closed-form expressions.

<p align="center">
  <img src="images/Chapter 3/4, Inference with Gaussians_cell10_img1.png" alt="Conditioning Gaussians" width="400"/>
</p>

This makes **linear Gaussian models** extremely powerful - we can do exact inference efficiently even in high dimensions. This is the foundation for algorithms like Kalman filtering.


## Chapter 4

So far we've assumed we know the parameters of our probability distributions. This chapter covers how to **learn** parameters from data.

---

### 1 - Maximum Likelihood Estimate

The Maximum Likelihood Estimate (MLE) chooses parameters that maximize the probability of the observed data:

$$\theta_{\text{MLE}} = \arg\max_\theta P(D|\theta)$$

Assuming independent samples, we maximize the product (or equivalently, the sum of log probabilities):

$$\theta_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log P(D_i|\theta)$$

**Examples:**

For a **Bernoulli distribution** (coin flips), the MLE is simply the sample mean:
$$\theta_{\text{MLE}} = \frac{k}{n}$$

For a **Gaussian**, the MLE estimates are:
$$\mu_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N x_i, \quad \sigma^2_{\text{MLE}} = \frac{1}{N}\sum_{i=1}^N (x_i - \mu)^2$$

For **Bayesian networks**, we can estimate each conditional probability table independently by counting frequencies in the data.

MLE is simple and works well with lots of data, but can overfit with small datasets and doesn't quantify uncertainty in the parameters.

---

### 2 - Bayesian Parameter Learning

Rather than picking a single "best" parameter, Bayesian learning maintains a full distribution over parameters:

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

We start with a **prior** P(θ) encoding our beliefs, then update to a **posterior** P(θ|D) after seeing data.

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

---

### 3 - Non-Parametric Models

Parametric models assume data comes from a specific family (Gaussian, etc.). **Non-parametric models** are more flexible.

**Kernel Density Estimation (KDE)** places a kernel function (e.g., Gaussian) at each data point:

$$p(x) = \frac{1}{N}\sum_{i=1}^N K_\sigma(x - x_i)$$

The bandwidth σ controls smoothness:

<p align="center">
  <img src="images/Chapter 4/3, Non-Parametric Models_cell5_img1.png" alt="KDE small bandwidth" width="300"/>
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
  <img src="images/Chapter 4/4, Learning With Missing Data_cell7_img1.png" alt="Model-based imputation" width="350"/>
</p>

**K-nearest neighbors:**
- Fill missing values using similar complete data points
- Works well when similar examples exist

<p align="center">
  <img src="images/Chapter 4/4, Learning With Missing Data_cell14_img1.png" alt="KNN imputation" width="400"/>
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

The challenge is that the space of possible graphs grows super-exponentially with the number of variables. For just 3 variables, there are many possible structures:
- A → B → C
- A → B ← C  
- A ← B → C
- A ← B ← C
- And many more with different edge directions

**Approach:**
1. Define a **scoring function** (e.g., likelihood, BIC, or AIC) that measures how well a structure fits the data
2. **Search** through possible structures to find the best score
3. Use heuristics like greedy search, hill climbing, or more sophisticated algorithms

**Key insight:** Multiple graphs can represent the same set of conditional independencies. These form **equivalence classes**. For example:
- A → B → C
- A ← B ← C  
- A ← B → C

All three encode: "A and C are conditionally independent given B." We can't distinguish between them from data alone - only their independence structure matters.

The notebook demonstrates this by testing different 3-variable structures and comparing their likelihoods on sample data. With enough data, the true structure (or its equivalence class) should score highest.

**Challenges:**
- Computational cost of searching exponentially many structures
- Need sufficient data to distinguish structures reliably
- Regularization (via BIC/AIC penalties) to avoid overfitting with complex graphs

Structure learning is powerful but difficult. It's most practical when:
- You have domain knowledge to constrain the search
- The number of variables is modest
- You have large amounts of data


## Chapter 6

Moving from inference to action, this chapter introduces **decision theory** - how to choose actions optimally when outcomes are uncertain.

---

### 1 - Utility

How do we make rational decisions? **Utility theory** provides the foundation.

**Von Neumann-Morgenstern Axioms:**
If preferences satisfy:
- **Completeness**: Can compare any two options
- **Transitivity**: If A ≻ B and B ≻ C, then A ≻ C
- **Continuity**: No option is infinitely better or worse
- **Independence**: Preferences between lotteries don't depend on irrelevant alternatives

Then there exists a **utility function** U such that we should choose actions to maximize **expected utility**:

$$EU(a|o) = \sum_{s'} P(s'|a, o) \cdot U(s')$$

**Risk attitudes:**
- **Risk-neutral**: Care only about expected value
- **Risk-averse**: Prefer certain outcomes over gambles with same expected value
- **Risk-seeking**: Prefer gambles over certain outcomes

For example, most people are risk-averse about money: they'd rather have \\$50 for sure than a 50/50 chance at \\$0 or \\$100, even though both have the same expected value.

The shape of the utility function encodes risk attitudes:
- Concave: risk-averse
- Linear: risk-neutral
- Convex: risk-seeking

This explains why people buy insurance (pay to reduce risk) and lottery tickets (pay for small chances of big rewards).

---

### 2 - Decision Networks, Value of Information, Irrationality

**Decision networks** (influence diagrams) extend Bayesian networks with:
- **Decision nodes** (squares): Actions we can choose
- **Utility nodes** (diamonds): Rewards/costs

To find the optimal decision:
1. For each possible action
2. Infer the resulting probability distribution over outcomes
3. Compute expected utility
4. Choose the action with maximum expected utility

**Value of Information (VOI):**

Sometimes we can gather information before deciding. Is it worth it?

$$VOI = EU(\text{with information}) - EU(\text{without information})$$

For example: Should you check the weather forecast before deciding whether to bring an umbrella? The VOI is the improvement in expected utility from knowing the forecast.

The notebook demonstrates this with a bus decision problem: should you wait for the bus or walk? Checking a rain forecast has positive VOI if it helps you make a better decision.

**Key insights:**
- Information is never harmful (in expectation) - VOI ≥ 0
- Information has no value if it doesn't change your decision
- VOI can guide which sensors/experiments to use

**Irrationality:**

The notebook also touches on how real humans violate these axioms:
- Framing effects (same choice described differently leads to different decisions)
- Sunk cost fallacy
- Probability weighting (overestimating rare events)

Understanding normative decision theory helps us build better AI systems and understand human biases.


## Chapter 7

The final chapter addresses **sequential decisions** - choosing actions over time when each action affects future states.

---

### 1 - Markov Decision Processes

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

### 3 - Value Iteration (and Linear Programming)

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

### 4 - Linear Quadratic Regulator (MDP in Continuous Space)

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
  <img src="images/Chapter 7/Linear Quadratic Regulator (MDP in continuous space)_cell4_img1.png" alt="LQR system" width="500"/>
</p>

The notebook demonstrates a simple 1D system where:
- State accumulates over time
- Control actions can adjust the state
- Optimal policy balances state deviation costs against control effort

<p align="center">
  <img src="images/Chapter 7/Linear Quadratic Regulator (MDP in continuous space)_cell6_img1.png" alt="LQR trajectory" width="500"/>
</p>

**Applications:**
- Robotics (motor control)
- Economics (optimal control policies)
- Aerospace (aircraft stabilization)

LQR is the foundation for more sophisticated continuous control methods. Though limited to linear dynamics and quadratic costs, it provides intuition and can be extended (iterative LQR, model predictive control) to handle nonlinear systems.
