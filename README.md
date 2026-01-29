# Decision Making Under Uncertainty

This repository contains my interactive Jupyter notebooks covering content from the book "Algorithms for Decision Making" by Mykel J. Kochenderfer, Tim A. Wheeler, and Kyle H. Wray.

---


## Chapter 2

This chapter covers the basics of probability theory.

---

### 1 - Rules of Probability, Factors and Graphs

This notebook introduces the Kolmogorov axioms and fundamental probability laws, then shows how to represent probability distributions as factors.

The notebook covers:
- **Factor operations**: conditioning, marginalization, multiplication
- **Directed graphical models** (Bayesian networks) that encode conditional independence
- **Markov blanket**: each variable is independent of all others given its parents, children, and children's parents

---

### 2 - Common Distributions

This notebook catalogs some important distributions for machine learning and decision theory.

Discrete distributions:
- **Bernoulli/Binomial**: Binary outcomes (coin flips)

<p align="center">
  <img src="images/Chapter 2/2, common distributions_cell5_img1.png" alt="Binomial distribution" width="400"/>
</p>

- **Categorical/Multinomial**: Multiple discrete outcomes (dice rolls)

Continuous distributions:
- **Uniform**: Constant probability over an interval
- **Gaussian**: The ubiquitous bell curve

<p align="center">
  <img src="images/Chapter 2/2, common distributions_cell12_img1.png" alt="Gaussian distribution" width="400"/>
</p>

Advanced models:
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

For a Bernoulli distribution (coin flips), the MLE is simply the sample mean. For a Gaussian, it's the sample mean and variance. For Bayesian networks, we can estimate each conditional probability table independently by counting frequencies in the data.

MLE is simple and works well with lots of data, but can overfit with small datasets and doesn't quantify uncertainty in the parameters.

---

### 2 - Bayesian Parameter Learning

Rather than picking a single "best" parameter, Bayesian learning maintains a full distribution over parameters. We start with a prior encoding our beliefs, then update after seeing data.

**Conjugate priors** make this tractable. For a Bernoulli likelihood, the Beta distribution is conjugate:

<p align="center">
  <img src="images/Chapter 4/2, Bayesian Parameter Learning_cell9_img1.png" alt="Beta prior" width="400"/>
</p>

After observing data, the posterior is also Beta with updated parameters:

<p align="center">
  <img src="images/Chapter 4/2, Bayesian Parameter Learning_cell10_img1.png" alt="Beta posterior" width="400"/>
</p>

For categorical data, the Dirichlet distribution is the multi-dimensional analog of the Beta.

<p align="center">
  <img src="images/Chapter 4/2, Bayesian Parameter Learning_cell17_img1.png" alt="Dirichlet distribution" width="500"/>
</p>

**Maximum A Posteriori (MAP)** estimates choose the most probable parameter:
$$\theta_{\text{MAP}} = \arg\max_\theta P(\theta|D)$$

This is a compromise between full Bayesian inference and MLE, often used for computational efficiency.

---

### 3 - Non-Parametric Models

Parametric models assume data comes from a specific family (Gaussian, etc.). Non-parametric models are more flexible.

Kernel Density Estimation (KDE) places a kernel function (e.g., Gaussian) at each data point.

<p align="center">
  <img src="images/Chapter 4/3, Non-Parametric Models_cell5_img1.png" alt="KDE small bandwidth" width="300"/>
  <img src="images/Chapter 4/3, Non-Parametric Models_cell5_img2.png" alt="KDE medium bandwidth" width="300"/>
  <img src="images/Chapter 4/3, Non-Parametric Models_cell5_img3.png" alt="KDE large bandwidth" width="300"/>
</p>
<p align="center"><em>Kernel density estimates with small, medium, and large bandwidths.</em></p>

Small σ captures fine details but can overfit. Large σ is smoother but may miss structure.

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

The Expectation-Maximization (EM) algorithm handles learning with latent (hidden) variables or missing data.

EM alternates between two steps:
1. **E-step**: Infer distribution over hidden variables given current parameters
2. **M-step**: Update parameters to maximize expected likelihood

<p align="center">
  <img src="images/Chapter 4/5, The EM algorithm_cell5_img1.png" alt="original data" width="350"/>
</p>
<p align="center">
  <img src="images/Chapter 4/5, The EM algorithm_cell7_img1.png" alt="EM iteration 10" width="350"/>
</p>

The algorithm is guaranteed to improve the likelihood at each iteration, though it may converge to a local optimum.


## Chapter 5

Beyond learning parameters, we can learn the **structure** of the graphical model itself, i.e., which variables depend on which others.

---

### 1 - Searching PGMs

This notebook tests different 3-variable graph structures by comparing their likelihoods. The notebook also shows that many different graphs can encode the same conditional independence information.


## Chapter 6

This chapter introduces decision theory.

---

### 1 - Utility

How do we make rational decisions? Introduces Von Neumann-Morgenstern Axioms and risk attitudes.

---

### 2 - Decision Networks, Value of Information, Irrationality

Decision networks (influence diagrams) extend Bayesian networks with decision nodes (squares) for actions we can choose and utility nodes (diamonds) for rewards/costs.

To find the optimal decision: for each possible action, infer the resulting probability distribution over outcomes, compute expected utility, and choose the action with maximum expected utility.

**Value of Information (VOI):**

Sometimes we can gather information before deciding. The value of information is the improvement in expected utility from knowing something before you act. For example: Should you check the weather forecast before deciding whether to bring an umbrella?


## Chapter 7

This chapter addresses sequential decisions.

---

### 1 - Markov Decision Processes

A Markov Decision Process (MDP) models sequential decision making with states, actions, a transition function, and a reward function. The next state depends only on the current state and action, not the full history.

The utility of a policy can be defined for finite horizons (sum of rewards) or infinite horizons with a discount factor (making near-term rewards more valuable than distant ones).

The Bellman equation relates utility across time steps and is the foundation for computing optimal policies.

The notebook demonstrates this with a maze navigation problem.

---

### 2 - Getting a Policy from a Value Function

A policy specifies which action to take in each state. Given a value function, the optimal policy chooses the action that maximizes expected future reward.

**Policy iteration** alternates between:

1. Evaluation: Compute value function for current policy
2. Improvement: Update policy to be greedy with respect to values

Policy iteration is guaranteed to converge to the optimal policy in finite iterations because:
- There are finitely many policies
- Each iteration strictly improves the policy (unless already optimal)
- We can't cycle since we only move to strictly better policies

**Value iteration** is an alternative that combines evaluation and improvement in one step.

Start with arbitrary values (e.g., all zeros), and iterate until convergence. The Bellman update is a **contraction mapping** that's guaranteed to converge to the optimal values, then we extract the optimal policy by being greedy.

Comparison to policy iteration:
- Value iteration: Simpler, one operation per iteration
- Policy iteration: May converge in fewer iterations
- Both find the optimal policy
