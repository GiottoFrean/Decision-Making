# Learning

This chapter covers how to learn model parameters from data.

---

## 1 - Maximum Likelihood Estimate

The Maximum Likelihood Estimate (MLE) chooses parameters that maximize the probability of the observed data.

For a **Bernoulli distribution** (coin flips), the MLE is simply the sample mean. For a **Gaussian**, it's the sample mean and variance. For **Bayesian networks**, we can estimate each conditional probability table independently by counting frequencies in the data.

MLE is simple and works well with lots of data, but can overfit with small datasets and doesn't quantify uncertainty in the parameters.

---

## 2 - Bayesian Parameter Learning

Rather than picking a single "best" parameter, Bayesian learning maintains a full distribution over parameters. We start with a **prior** encoding our beliefs, then update to a **posterior** after seeing data.

**Conjugate priors** make this tractable. For a Bernoulli likelihood, the **Beta distribution** is conjugate:

<p align="center">
  <img src="../images/Chapter 4/2, Bayesian Parameter Learning_cell9_img1.png" alt="Beta prior" width="400"/>
</p>

After observing data, the posterior is also Beta with updated parameters:

<p align="center">
  <img src="../images/Chapter 4/2, Bayesian Parameter Learning_cell10_img1.png" alt="Beta posterior" width="400"/>
</p>

For categorical data, the **Dirichlet distribution** is the multi-dimensional analog of the Beta.

<p align="center">
  <img src="../images/Chapter 4/2, Bayesian Parameter Learning_cell17_img1.png" alt="Dirichlet distribution" width="500"/>
</p>

**Maximum A Posteriori (MAP)** estimates choose the most probable parameter:
$$\theta_{\text{MAP}} = \arg\max_\theta P(\theta|D)$$

This is a compromise between full Bayesian inference and MLE, often used for computational efficiency.
. ## 3 - Non-Parametric Models

Parametric models assume data comes from a specific family (Gaussian, etc.). **Non-parametric models** are more flexible.

**Kernel Density Estimation (KDE)** places a kernel function (e.g., Gaussian) at each data point:

$$p(x) = \frac{1}{N}\sum_{i=1}^N K_\sigma(x - x_i)$$
. The bandwidthimages/Chapter 4/3, Non-Parametric Models_cell5_img1.png" alt="KDE small bandwidth" width="300"/>
  <img src="../images/Chapter 4/3, Non-Parametric Models_cell5_img2.png" alt="KDE medium bandwidth" width="300"/>
  <img src="../images/Chapter 4/3, Non-Parametric Models_cell5_img3.png" alt="KDE large bandwidth" width="300"/>
</p>
<p align="center"><em>Kernel density estimates with small, medium, and large bandwidths.</em></p>

Small σ captures fine details but can overfit. Large σ is smoother but may miss structure. The choice of bandwidth is crucial.

---

## 4 - Learning With Missing Data

Real datasets often have missing values. Several strategies exist:

**Simple imputation:**
- Fill with mean, median, or mode
- Fast but ignores uncertainty

**Model-based imputation:**
- Fit a model (e.g., Gaussian)
- Sample missing values from the conditional distribution

<p align="center">
  <img src="../images/Chapter 4/4, Learning With Missing Data_cell5_img1.png" alt="Original data" width="350"/>
  <img src="../images/Chapter 4/4, Learning With Missing Data_cell11_img1.png" alt="Model-based imputation" width="350"/>
</p>

**K-nearest neighbors:**
- Fill missing values using similar complete data points
- Works well when similar examples exist

<p align="center">
  <img src="../images/Chapter 4/4, Learning With Missing Data_cell5_img1.png" alt="Original data" width="350"/>
  <img src="../images/Chapter 4/4, Learning With Missing Data_cell14_img1.png" alt="Model-based imputation" width="350"/>
</p>


**Full Bayesian approach:**
- Treat missing values as latent variables and integrate them out
- Principled but computationally expensive

---

## 5 - The EM Algorithm

The **Expectation-Maximization (EM)** algorithm handles learning with latent (hidden) variables or missing data.

EM alternates between two steps:
1. **E-step**: Infer distribution over hidden variables given current parameters
2. **M-step**: Update parameters to maximize expected likelihood

<p align="center">
  <img src="../images/Chapter 4/5, The EM algorithm_cell5_img1.png" alt="EM iteration 1" width="350"/>
  <img src="../images/Chapter 4/5, The EM algorithm_cell7_img1.png" alt="EM iteration 5" width="350"/>
</p>
<p align="center"><em>EM algorithm progressively improving a mixture of Gaussians fit.</em></p>

The algorithm is guaranteed to improve the likelihood at each iteration, though it may converge to a local optimum.

EM is particularly powerful for:
- **Mixture models**: Learning cluster assignments and parameters simultaneously
- **Bayesian networks with missing data**: Filling in unobserved variables while learning parameters

The key insight is that by treating hidden variables as if we knew their distribution (E-step), we can optimize parameters as if we had complete data (M-step).
