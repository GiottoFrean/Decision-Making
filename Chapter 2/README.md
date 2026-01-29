# Probability Foundations

Understanding probability is fundamental to reasoning under uncertainty. This chapter covers the mathematical foundations needed for probabilistic graphical models and decision-making.

---

## 1 - Rules of Probability, Factors and Graphs

This notebook introduces the Kolmogorov axioms and fundamental probability laws, then shows how to represent probability distributions as **factors** - table-based data structures that make computation tractable.

The key insight is that we don't need to store the full joint probability table. Instead, we can represent it as a product of smaller factors, one for each variable conditioned on its parents:

$$P(A, B, C, D) = P(A)P(B|A)P(C|A)P(D|B,C)$$

The notebook covers:
- **Factor operations**: conditioning (fixing variables), marginalization (summing out), and multiplication
- **Directed graphical models** (Bayesian networks) that encode conditional independence
- **Markov blanket**: each variable is independent of all others given its parents, children, and children's parents

These graph structures let us exploit independence assumptions to make inference computationally feasible.

---

## 2 - Common Distributions

This notebook catalogs the most important probability distributions in machine learning and decision theory.

**Discrete distributions:**
- **Bernoulli/Binomial**: Binary outcomes (coin flips)
- **Categorical/Multinomial**: Multiple discrete outcomes (dice rolls)

<p align="center">
  <img src="../images/Chapter 2/2, common distributions_cell5_img1.png" alt="Binomial distribution" width="400"/>
</p>

**Continuous distributions:**
- **Uniform**: Constant probability over an interval
- **Gaussian**: The ubiquitous bell curve

<p align="center">
  <img src="../images/Chapter 2/2, common distributions_cell12_img1.png" alt="Gaussian distribution" width="400"/>
</p>

**Advanced models:**
- **Mixture of Gaussians**: Combines multiple Gaussian components, useful for multimodal data

<p align="center">
  <img src="../images/Chapter 2/2, common distributions_cell20_img1.png" alt="Mixture of Gaussians" width="400"/>
</p>

- **Linear Gaussian models**: Where means depend linearly on parent variables, enabling tractable inference

Understanding these distributions is essential as they form the building blocks for more complex probabilistic models.
