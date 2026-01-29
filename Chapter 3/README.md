# Inference

Once we have a probabilistic model, we need **inference** - computing probability distributions over variables of interest. This chapter explores both exact and approximate inference methods.

---

## 1 - Exact Inference

Inference means answering queries like "What's the probability of Y given what we know about X?"

$$P(Y|X_{\text{known}}) = \sum_{X_{\text{unknown}}}P(Y,X_{\text{unknown}}|X_{\text{known}})$$

We condition on known variables and marginalize out the unknowns. With factors, this is straightforward:
1. Condition on known variables (fix their values)
2. Marginalize out irrelevant variables (sum over their values)
3. Normalize the result

The **sum-product variable elimination** algorithm avoids building the full joint table by carefully ordering operations on factors. This exploits the graph structure, making inference tractable even when the full joint would be impossibly large.

The notebook demonstrates this with a simple example: deciding whether to buy tea on the way home based on who's at home.

---

## 2 - Inference, Integration with Samples

When exact inference is intractable, we can use **Monte Carlo methods** - using samples to approximate integrals and expectations.

The basic Monte Carlo formula approximates an expectation:
$$E[f(x)] \approx \frac{1}{N}\sum_{i=1}^N f(x_i) \text{ where } x_i \sim p(x)$$

<p align="center">
  <img src="../images/Chapter 3/2, Inference, Integration with samples_cell9_img1.png" alt="Monte Carlo convergence" width="500"/>
</p>
<p align="center"><em>Monte Carlo estimates converge to the true value as sample size increases.</em></p>

**Importance sampling** handles cases where we can't sample directly from the target distribution. We sample from a proposal distribution q(x) and reweight:

$$E_p[f(x)] \approx \sum_{i=1}^N w_i f(x_i) \text{ where } x_i \sim q(x), \; w_i = \frac{p(x_i)}{q(x_i)}$$

This is particularly useful when we only know an unnormalized version of p(x).

<p align="center">
  <img src="../images/Chapter 3/2, Inference, Integration with samples_cell19_img1.png" alt="Importance sampling" width="500"/>
</p>

---

## 3 - Sampling Factors

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

## 4 - Inference with Gaussians

Gaussians have special properties that make inference tractable in closed form.

**Marginalization** is trivial - just extract the relevant dimensions:

<p align="center">
  <img src="../images/Chapter 3/4, Inference with Gaussians_cell5_img1.png" alt="Marginalizing Gaussians" width="400"/>
</p>

**Conditioning** uses a simple formula. If we partition variables into observed (a) and unobserved (b):

$$p(x_b | x_a) = \mathcal{N}(\mu_{b|a}, \Sigma_{b|a})$$

where the conditional mean and covariance have closed-form expressions.

<p align="center">
  <img src="../images/Chapter 3/4, Inference with Gaussians_cell10_img1.png" alt="Conditioning Gaussians" width="400"/>
</p>

This makes **linear Gaussian models** extremely powerful - we can do exact inference efficiently even in high dimensions. This is the foundation for algorithms like Kalman filtering.
