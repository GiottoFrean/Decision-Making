# Probability Foundations

This chapter covers the basics of probability theory.

---

## 1 - Rules of Probability, Factors and Graphs

This notebook introduces the Kolmogorov axioms and fundamental probability laws, then shows how to represent probability distributions as factors.

The notebook covers:
- **Factor operations**: conditioning, marginalization, multiplication
- **Directed graphical models** (Bayesian networks) that encode conditional independence
- **Markov blanket**: each variable is independent of all others given its parents, children, and children's parents

---

## 2 - Common Distributions

This notebook catalogs some important distributions for machine learning and decision theory.

Discrete distributions:
- **Bernoulli/Binomial**: Binary outcomes (coin flips)

<p align="center">
  <img src="../images/Chapter 2/2, common distributions_cell5_img1.png" alt="Binomial distribution" width="400"/>
</p>

- **Categorical/Multinomial**: Multiple discrete outcomes (dice rolls)

Continuous distributions:
- **Uniform**: Constant probability over an interval
- **Gaussian**: The ubiquitous bell curve

<p align="center">
  <img src="../images/Chapter 2/2, common distributions_cell12_img1.png" alt="Gaussian distribution" width="400"/>
</p>

Advanced models:
- **Mixture of Gaussians**: Combines multiple Gaussian components, useful for multimodal data
- **Linear Gaussian models**: Where means depend linearly on parent variables, enabling tractable inference