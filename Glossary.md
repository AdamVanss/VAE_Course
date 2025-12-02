# Glossary of Terms and Symbols

## Core Symbols

### Data and Latent Variables

| Symbol | Meaning | Typical Shape |
|--------|---------|---------------|
| \(x\) | Observed data (e.g., image) | \((N, D)\) or \((N, C, H, W)\) |
| \(x^{(i)}\) | The i-th data point | \((D,)\) |
| \(z\) | Latent variable / latent code | \((N, d_z)\) |
| \(D\) or \(d_x\) | Dimension of data | Scalar (e.g., 784 for MNIST) |
| \(d_z\) | Dimension of latent space | Scalar (e.g., 20) |
| \(N\) | Number of data points | Scalar |
| \(\mathcal{D}\) | Dataset \(\{x^{(1)}, \ldots, x^{(N)}\}\) | — |

### Parameters

| Symbol | Meaning |
|--------|---------|
| \(\theta\) | Decoder (generative model) parameters |
| \(\phi\) | Encoder (inference model) parameters |
| \(\mu\) | Mean of a Gaussian distribution |
| \(\sigma^2\) | Variance of a Gaussian distribution |
| \(\sigma\) | Standard deviation |
| \(\log \sigma^2\) | Log-variance (often parameterized directly) |
| \(\mu_\phi(x)\) | Encoder mean output |
| \(\sigma_\phi^2(x)\) | Encoder variance output |
| \(\mu_\theta(z)\) | Decoder mean output |

### Probability Distributions

| Symbol | Meaning | Description |
|--------|---------|-------------|
| \(p_{\text{data}}(x)\) | True data distribution | Unknown, we only have samples |
| \(p_\theta(x)\) | Model marginal likelihood | \(\int p(z)p_\theta(x|z)dz\) |
| \(p(z)\) | Prior distribution | Typically \(\mathcal{N}(0, I)\) |
| \(p_\theta(x\|z)\) | Likelihood / Decoder | Distribution over x given z |
| \(p_\theta(z\|x)\) | True posterior | Intractable |
| \(q_\phi(z\|x)\) | Approximate posterior / Encoder | Tractable approximation |
| \(p_\theta(x, z)\) | Joint distribution | \(p(z)p_\theta(x|z)\) |

### Loss and Objective

| Symbol | Meaning |
|--------|---------|
| \(\mathcal{L}\) | ELBO (Evidence Lower BOund) |
| \(\mathcal{L}(\theta, \phi; x)\) | ELBO for data point x |
| \(D_{\text{KL}}(q \| p)\) | KL divergence from p to q |
| \(\text{KL}\) | Shorthand for KL divergence |
| \(\mathbb{E}\) | Expectation |
| \(\mathbb{E}_{q}\) | Expectation under distribution q |

---

## Distribution Notation

### Gaussian (Normal) Distribution

| Notation | Meaning |
|----------|---------|
| \(\mathcal{N}(\mu, \sigma^2)\) | Univariate Gaussian with mean μ, variance σ² |
| \(\mathcal{N}(x; \mu, \sigma^2)\) | Density of x under this Gaussian |
| \(\mathcal{N}(0, I)\) | Standard multivariate Gaussian |
| \(\mathcal{N}(\mu, \Sigma)\) | Multivariate Gaussian with mean μ, covariance Σ |
| \(\mathcal{N}(\mu, \text{diag}(\sigma^2))\) | Diagonal covariance (independent dimensions) |

### Other Distributions

| Notation | Meaning |
|----------|---------|
| \(\text{Bernoulli}(p)\) | Bernoulli distribution with probability p |
| \(\text{Categorical}(p_1, \ldots, p_K)\) | Categorical over K classes |
| \(\text{Uniform}(a, b)\) | Uniform distribution on [a, b] |

---

## Mathematical Notation

### Operators

| Symbol | Meaning |
|--------|---------|
| \(\nabla_\theta\) | Gradient with respect to θ |
| \(\frac{\partial}{\partial \theta}\) | Partial derivative with respect to θ |
| \(\log\) | Natural logarithm (ln) |
| \(\exp\) | Exponential function |
| \(\sum\) | Summation |
| \(\int\) | Integration |
| \(\prod\) | Product |
| \(\odot\) | Element-wise (Hadamard) product |
| \(\|\cdot\|\) | Norm (usually L2) |
| \(\text{tr}(\cdot)\) | Matrix trace |
| \(\|\cdot\|\) | Determinant (when applied to matrices) |
| \(\text{sg}[\cdot]\) | Stop-gradient operator |

### Set Notation

| Symbol | Meaning |
|--------|---------|
| \(\mathbb{R}\) | Real numbers |
| \(\mathbb{R}^d\) | d-dimensional real space |
| \(\in\) | Element of |
| \(\{0, 1\}\) | Binary set |
| \([0, 1]\) | Closed interval from 0 to 1 |

---

## Key Equations Reference

### ELBO

\[
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))
\]

### KL Divergence (General)

\[
D_{\text{KL}}(q \| p) = \mathbb{E}_q\left[\log \frac{q(x)}{p(x)}\right] = \mathbb{E}_q[\log q(x)] - \mathbb{E}_q[\log p(x)]
\]

### KL Divergence (Gaussian to Standard Normal)

\[
D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2}\sum_j \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)
\]

### Reparameterization

\[
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

### Gaussian Log-Likelihood

\[
\log \mathcal{N}(x; \mu, \sigma^2) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}
\]

### Binary Cross-Entropy

\[
\text{BCE}(x, p) = -[x \log p + (1-x) \log(1-p)]
\]

---

## Terms and Definitions

### A

**Aggregate Posterior:** The average of approximate posteriors over the data: \(\bar{q}(z) = \frac{1}{N}\sum_i q(z|x^{(i)})\). Ideally matches the prior.

**Amortized Inference:** Using a neural network (encoder) to directly output posterior parameters, rather than optimizing per data point.

**Annealing (KL):** Gradually increasing the KL weight during training to prevent posterior collapse.

### B

**β-VAE:** VAE variant that scales the KL term by β > 1 to encourage disentanglement.

**ELBO (Evidence Lower BOund):** The quantity VAEs optimize; a lower bound on log p(x).

### C

**Codebook:** In VQ-VAE, a dictionary of discrete embedding vectors used for quantization.

**Conditional VAE (CVAE):** VAE that conditions encoder and decoder on auxiliary information (labels, etc.).

**Commitment Loss:** In VQ-VAE, encourages encoder outputs to stay close to codebook entries.

### D

**Decoder:** Neural network that maps latent z to distribution parameters over x. Denoted \(p_\theta(x|z)\).

**Dequantization:** Adding uniform noise to discrete data to make it continuous.

**Disentanglement:** Property where different latent dimensions capture different, independent factors of variation.

### E

**Encoder:** Neural network that maps data x to distribution parameters over z. Denoted \(q_\phi(z|x)\).

**Evidence:** The marginal likelihood \(p(x)\), also called model evidence.

### F

**Free Bits:** Technique to prevent posterior collapse by enforcing minimum KL per dimension.

### G

**Generative Model:** A model that can generate new samples from a learned distribution.

### H

**Hierarchical VAE:** VAE with multiple layers of latent variables.

### I

**IWAE (Importance Weighted Autoencoder):** VAE variant using multiple samples for a tighter bound.

**Importance Sampling:** Technique for estimating expectations by sampling from a proposal distribution.

### J

**Jensen's Inequality:** For convex f: \(f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]\). For concave f (like log): \(f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]\).

### K

**KL Divergence:** Measure of how one probability distribution differs from another. Always ≥ 0.

### L

**Latent Variable:** Hidden/unobserved variable that captures underlying structure.

**Likelihood:** The probability of data given latent variable: \(p(x|z)\).

**Log-Variance Parameterization:** Encoding variance as \(\log \sigma^2\) for numerical stability.

### M

**Manifold Hypothesis:** The idea that high-dimensional data lies on a lower-dimensional manifold.

**Marginal Likelihood:** The probability of data under the model: \(p(x) = \int p(z)p(x|z)dz\).

**MLE (Maximum Likelihood Estimation):** Finding parameters that maximize data likelihood.

### N

**Normalizing Flow:** Sequence of invertible transformations for flexible density modeling.

### P

**Posterior:** Distribution over latent given data: \(p(z|x)\).

**Posterior Collapse:** Failure mode where encoder outputs match the prior, ignoring input.

**Prior:** Distribution over latent before seeing data: \(p(z)\).

### Q

**q(z|x):** Standard notation for the approximate posterior (encoder).

### R

**Reconstruction Loss:** The \(-\mathbb{E}[\log p(x|z)]\) term; measures how well we can reconstruct input.

**Reparameterization Trick:** Expressing stochastic sampling as deterministic function of noise to enable backpropagation.

### S

**Score Function Estimator:** Alternative to reparameterization using \(\nabla \log p\); high variance.

**Straight-Through Estimator:** Gradient approximation for non-differentiable operations.

### V

**Variational Inference:** Approximating intractable posteriors with tractable distributions by optimization.

**VampPrior:** Prior defined as mixture of variational posteriors at learned pseudo-inputs.

**VQ-VAE:** VAE with discrete latent codes via vector quantization.

---

## Abbreviations

| Abbreviation | Full Form |
|--------------|-----------|
| VAE | Variational Autoencoder |
| ELBO | Evidence Lower BOund |
| KL | Kullback-Leibler (divergence) |
| BCE | Binary Cross-Entropy |
| MSE | Mean Squared Error |
| MLE | Maximum Likelihood Estimation |
| CVAE | Conditional VAE |
| VQ-VAE | Vector Quantized VAE |
| IWAE | Importance Weighted Autoencoder |
| FID | Fréchet Inception Distance |
| t-SNE | t-distributed Stochastic Neighbor Embedding |
| UMAP | Uniform Manifold Approximation and Projection |
| MIG | Mutual Information Gap |
| DCI | Disentanglement, Completeness, Informativeness |
| NVAE | Nouveau VAE (Vahdat & Kautz) |
| BPD | Bits Per Dimension |

---

## Parameter Dimensions Quick Reference

For a VAE on MNIST (28×28 images, latent dim 20):

| Variable | Shape |
|----------|-------|
| Input x | (batch, 784) |
| Hidden h | (batch, 400) |
| Mean μ | (batch, 20) |
| Log-var | (batch, 20) |
| Sample z | (batch, 20) |
| Reconstruction | (batch, 784) |


