# Inference

Inference is about computing probabilities given what we know.

---

## 1 - Exact Inference

Inference means answering queries like "What's the probability of Y given what we know about X?"

$$P(Y|X_{\text{known}}) = \sum_{X_{\text{unknown}}}P(Y,X_{\text{unknown}}|X_{\text{known}})$$

We condition on known variables and marginalize out the unknowns. With factors, this is straightforward:
1. Condition on known variables (fix their values)
2. Marginalize out irrelevant variables (sum over their values)
3. Normalize the result

The **sum-product variable elimination** algorithm avoids building the full joint table by carefully ordering operations on factors. This exploits the conditional independence structure.

---

## 2 - Inference, Integration with Samples

In many cases we want to calculate an integral of a continuous function. When exact inference is intractable, we can use sampling methods.

The basic **Monte Carlo** formula approximates with:
$$\int f(x)p(x) dx \approx \frac{1}{N}\sum_{n=1}^N f(x_n)$$

**Importance sampling** handles cases where we can't sample directly from the distribution. We sample from a proposal distribution q(x) and reweight:

$$E_p[f(x)] \approx \sum_{i=1}^N w_i f(x_i) \text{ where } x_i \sim q(x), \; w_i = \frac{p(x_i)}{q(x_i)}$$

This is particularly useful when we only know an unnormalized version of p(x).

---

## 3 - Sampling Factors

This notebook covers algorithms for drawing samples from Bayesian networks, such as **rejection sampling**, **likelihood weighting**, and **Gibbs sampling**.

---

## 4 - Inference with Gaussians

Gaussians have special properties that make inference tractable in closed form.

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <img src="../images/Chapter 3/4, Inference with Gaussians_cell5_img1.png" alt="Original Gaussian" width="400"/>
  <img src="../images/Chapter 3/4, Inference with Gaussians_cell7_img1.png" alt="Conditioned Gaussian" width="400"/>
</div>

where the conditional mean and covariance have closed-form expressions.
