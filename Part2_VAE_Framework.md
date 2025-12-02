# Part 2: The VAE Framework

## Table of Contents
- [2.1 The Evidence Lower Bound (ELBO)](#21-the-evidence-lower-bound-elbo)
- [2.2 Complete ELBO Derivation](#22-complete-elbo-derivation)
- [2.3 Interpreting the ELBO Terms](#23-interpreting-the-elbo-terms)
- [2.4 The Reparameterization Trick](#24-the-reparameterization-trick)
- [2.5 KL Divergence for Gaussians](#25-kl-divergence-for-gaussians)
- [2.6 How Neural Networks Become Distributions](#26-how-neural-networks-become-distributions)
- [2.7 The Full VAE Architecture](#27-the-full-vae-architecture)
- [2.8 Recap](#28-recap)

---

## 2.1 The Evidence Lower Bound (ELBO)

### The Central Object of VAEs

The ELBO (Evidence Lower BOund) is the quantity we actually optimize in VAE training. Understanding it deeply is essential.

**The Setup:**
- We have data \(x\)
- We have a generative model: \(p(z)\) (prior), \(p_\theta(x|z)\) (decoder)
- We want to maximize \(\log p_\theta(x)\) but it's intractable
- We introduce \(q_\phi(z|x)\) (encoder) as an approximate posterior

**The ELBO:**
\[
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{\text{KL}}\left(q_\phi(z|x) \| p(z)\right)
\]

**The Fundamental Inequality:**
\[
\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x)
\]

The ELBO is a **lower bound** on the log-evidence (log marginal likelihood). Maximizing the ELBO:
1. Pushes up the lower bound, hence pushes up \(\log p_\theta(x)\) (improves the model)
2. Tightens the bound by making \(q_\phi(z|x)\) closer to the true posterior \(p_\theta(z|x)\)

---

## 2.2 Complete ELBO Derivation

We'll derive the ELBO three different ways to build deep understanding.

### Derivation 1: From Jensen's Inequality

Start with the log marginal likelihood:
\[
\log p_\theta(x) = \log \int p_\theta(x, z) \, dz
\]

Introduce \(q_\phi(z|x)\) by multiplying and dividing:
\[
\log p_\theta(x) = \log \int q_\phi(z|x) \frac{p_\theta(x, z)}{q_\phi(z|x)} \, dz
\]

This is an expectation:
\[
\log p_\theta(x) = \log \mathbb{E}_{q_\phi(z|x)}\left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right]
\]

Apply **Jensen's inequality** (for concave function log):
\[
\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]
\]

Therefore:
\[
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]
\]

The right-hand side IS the ELBO:
\[
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]
\]

Let's expand this:
\[
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x, z) - \log q_\phi(z|x)\right]
\]

Since \(p_\theta(x, z) = p(z) p_\theta(x|z)\):
\[
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}\left[\log p(z) + \log p_\theta(x|z) - \log q_\phi(z|x)\right]
\]

Rearranging:
\[
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] + \mathbb{E}_{q_\phi(z|x)}\left[\log p(z) - \log q_\phi(z|x)\right]
\]

The second term is \(-D_{\text{KL}}(q_\phi(z|x) \| p(z))\):
\[
\boxed{\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{\text{KL}}\left(q_\phi(z|x) \| p(z)\right)}
\]

### Derivation 2: From the KL to the True Posterior

Consider the KL divergence between our approximate posterior and the true posterior:
\[
D_{\text{KL}}\left(q_\phi(z|x) \| p_\theta(z|x)\right) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{q_\phi(z|x)}{p_\theta(z|x)}\right]
\]

Using Bayes' rule: \(p_\theta(z|x) = \frac{p_\theta(x|z)p(z)}{p_\theta(x)}\)
\[
D_{\text{KL}}(q \| p_\theta(z|x)) = \mathbb{E}_{q}\left[\log q_\phi(z|x) - \log p_\theta(x|z) - \log p(z) + \log p_\theta(x)\right]
\]

Since \(\log p_\theta(x)\) doesn't depend on \(z\):
\[
D_{\text{KL}}(q \| p_\theta(z|x)) = -\mathbb{E}_{q}\left[\log p_\theta(x|z)\right] + D_{\text{KL}}(q_\phi(z|x) \| p(z)) + \log p_\theta(x)
\]

Rearranging:
\[
\log p_\theta(x) = \mathbb{E}_{q}\left[\log p_\theta(x|z)\right] - D_{\text{KL}}(q_\phi(z|x) \| p(z)) + D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))
\]

Therefore:
\[
\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + D_{\text{KL}}\left(q_\phi(z|x) \| p_\theta(z|x)\right)
\]

Since KL divergence is always ≥ 0:
\[
\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x)
\]

**Key insight:** The gap between \(\log p_\theta(x)\) and the ELBO is exactly \(D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))\)—how far our encoder is from the true posterior.

### Derivation 3: Importance Sampling View

The marginal likelihood can be written:
\[
p_\theta(x) = \int q_\phi(z|x) \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)} dz = \mathbb{E}_{q_\phi(z|x)}\left[\frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}\right]
\]

Define the **importance weight**:
\[
w(z) = \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}
\]

Then \(p_\theta(x) = \mathbb{E}_{q}[w(z)]\).

Taking the log and applying Jensen's:
\[
\log p_\theta(x) = \log \mathbb{E}_q[w] \geq \mathbb{E}_q[\log w] = \mathcal{L}
\]

This view is useful for understanding IWAE (Importance Weighted Autoencoders).

---

## 2.3 Interpreting the ELBO Terms

The ELBO has two terms with distinct roles:

\[
\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction term}} - \underbrace{D_{\text{KL}}(q_\phi(z|x) \| p(z))}_{\text{Regularization term}}
\]

### Reconstruction Term: \(\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]\)

**What it measures:** How well can we reconstruct \(x\) from a latent code sampled from our encoder?

**Interpretation:**
1. Sample \(z \sim q_\phi(z|x)\) (encode \(x\) to get a distribution, sample from it)
2. Compute \(\log p_\theta(x|z)\) (how likely is original \(x\) under decoded distribution?)
3. Average over many samples (or use 1 sample as Monte Carlo estimate)

**Maximizing this term:**
- Pushes the decoder to assign high probability to \(x\) given \(z\)
- Pushes the encoder to produce \(z\) codes that are useful for reconstruction
- This is like minimizing reconstruction error

**Connection to MSE:** For a Gaussian decoder with fixed variance \(\sigma^2\):
\[
\log p_\theta(x|z) = -\frac{1}{2\sigma^2}\|x - \mu_\theta(z)\|^2 + \text{const}
\]
Maximizing log-likelihood = minimizing MSE (up to constants).

### Regularization Term: \(D_{\text{KL}}(q_\phi(z|x) \| p(z))\)

**What it measures:** How different is the encoder's output distribution from the prior?

**Interpretation:**
- The prior \(p(z) = \mathcal{N}(0, I)\) defines what the latent space "should" look like
- The KL term penalizes deviation from this target
- It acts as a **regularizer** preventing the encoder from producing arbitrary distributions

**Why we need it:**
- Without KL regularization, the encoder could map each \(x\) to a unique, deterministic \(z\)
- The latent space would have "holes"—regions where no training data maps
- Sampling from the prior would produce garbage
- The KL term forces the latent space to be smooth and covered

**Minimizing this term:**
- Pushes \(q_\phi(z|x)\) toward \(\mathcal{N}(0, I)\)
- Encourages the encoder to spread data across the prior
- Enables meaningful sampling and interpolation

### The Trade-off

These two terms are in tension:
- **Reconstruction** wants distinctive \(z\) codes (sharp posteriors, far from prior)
- **Regularization** wants \(z\) codes near the prior (blurry posteriors, clustered)

Good training finds a balance where:
- Similar inputs map to nearby latent codes
- The overall latent distribution matches the prior
- Both reconstruction and generation work well

This trade-off is the source of the "blurriness" often seen in VAE reconstructions.

---

## 2.4 The Reparameterization Trick

### The Problem

We need to compute gradients:
\[
\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)]
\]

where \(q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x))\).

**Naive approach:** 
\[
\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \nabla_\phi \int q_\phi(z|x) f(z) dz
\]

The integral depends on \(\phi\) through the distribution \(q_\phi\). We can't simply move the gradient inside because the sampling distribution changes with \(\phi\).

**Score function estimator (REINFORCE):**
\[
\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[f(z) \nabla_\phi \log q_\phi(z|x)]
\]

This works but has **extremely high variance**, making training unstable.

### The Solution: Reparameterization

Express the random variable \(z\) as a deterministic function of \(\phi\) and a noise variable \(\epsilon\):
\[
z = g_\phi(\epsilon, x) = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

where \(\odot\) denotes element-wise multiplication.

Now the expectation is over \(\epsilon\), which doesn't depend on \(\phi\):
\[
\mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[f(\mu_\phi(x) + \sigma_\phi(x) \odot \epsilon)]
\]

The gradient can be moved inside:
\[
\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{\epsilon}[\nabla_\phi f(\mu_\phi(x) + \sigma_\phi(x) \odot \epsilon)]
\]

We can now estimate this with Monte Carlo:
\[
\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] \approx \frac{1}{L}\sum_{\ell=1}^{L} \nabla_\phi f(\mu_\phi(x) + \sigma_\phi(x) \odot \epsilon^{(\ell)})
\]

### Formal Justification

**Theorem (Leibniz integral rule / Interchanging gradient and expectation):**

If \(z = g_\phi(\epsilon, x)\) where \(\epsilon \sim p(\epsilon)\) (independent of \(\phi\)), and \(f\) and \(g\) are differentiable, then:
\[
\nabla_\phi \mathbb{E}_{p(\epsilon)}[f(g_\phi(\epsilon, x))] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon, x))]
\]

**Proof sketch:**
\begin{align}
\nabla_\phi \mathbb{E}_{p(\epsilon)}[f(g_\phi(\epsilon, x))] &= \nabla_\phi \int p(\epsilon) f(g_\phi(\epsilon, x)) d\epsilon \\
&= \int p(\epsilon) \nabla_\phi f(g_\phi(\epsilon, x)) d\epsilon \quad \text{(Leibniz rule)} \\
&= \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon, x))]
\end{align}

The key is that \(p(\epsilon)\) doesn't depend on \(\phi\), so the gradient passes through the integral.

By the chain rule:
\[
\nabla_\phi f(g_\phi(\epsilon, x)) = \nabla_z f(z)|_{z=g_\phi(\epsilon,x)} \cdot \nabla_\phi g_\phi(\epsilon, x)
\]

This is just standard backpropagation through the deterministic function \(g_\phi\).

### Visualizing the Trick

**Without reparameterization:**
```
           ┌─────────────┐
   x ─────▶│  Encoder    │─────▶ μ, σ² ─────▶ SAMPLE z ~ N(μ, σ²)
           │  (φ)        │         │                  ↓
           └─────────────┘         │           [ BLOCKED! ]
                                   │           Can't backprop
                                   │           through sampling
```

**With reparameterization:**
```
           ┌─────────────┐
   x ─────▶│  Encoder    │─────▶ μ, σ²
           │  (φ)        │         │
           └─────────────┘         │
                                   ↓
                         z = μ + σ ⊙ ε   ◀───── ε ~ N(0,I)
                                   │             (constant noise)
                                   ↓
                          [ BACKPROP OK! ]
                          Gradients flow through
                          μ and σ (deterministic)
```

### Code Implementation

```python
def reparameterize(mu, log_var):
    """
    Reparameterization trick: z = mu + std * epsilon
    
    Args:
        mu: (batch_size, latent_dim) - mean from encoder
        log_var: (batch_size, latent_dim) - log variance from encoder
    
    Returns:
        z: (batch_size, latent_dim) - sampled latent code
    """
    std = torch.exp(0.5 * log_var)  # (batch_size, latent_dim)
    eps = torch.randn_like(std)      # (batch_size, latent_dim), from N(0,I)
    z = mu + std * eps               # (batch_size, latent_dim)
    return z
```

**Why log variance?** We parameterize with \(\log \sigma^2\) instead of \(\sigma\) because:
1. \(\sigma\) must be positive; \(\log \sigma^2\) is unconstrained
2. More numerically stable for very small/large variances
3. The network can output any real number

To get \(\sigma\): `std = exp(0.5 * log_var)` since \(\exp(\frac{1}{2}\log\sigma^2) = \exp(\log\sigma) = \sigma\)

---

## 2.5 KL Divergence for Gaussians

### The Closed-Form Formula

For the encoder \(q_\phi(z|x) = \mathcal{N}(z; \mu, \text{diag}(\sigma^2))\) and prior \(p(z) = \mathcal{N}(z; 0, I)\):

\[
D_{\text{KL}}(q_\phi(z|x) \| p(z)) = \frac{1}{2}\sum_{j=1}^{d_z}\left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)
\]

where \(d_z\) is the latent dimension.

### Complete Derivation

The KL divergence between two Gaussians:
\[
D_{\text{KL}}(\mathcal{N}(\mu_1, \Sigma_1) \| \mathcal{N}(\mu_2, \Sigma_2))
\]

For multivariate Gaussians (general formula):
\[
D_{\text{KL}} = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1)\right]
\]

**For our case:** \(\mu_1 = \mu\), \(\Sigma_1 = \text{diag}(\sigma^2)\), \(\mu_2 = 0\), \(\Sigma_2 = I\)

Let's compute each term:

**Term 1:** \(\log\frac{|\Sigma_2|}{|\Sigma_1|} = \log\frac{|I|}{|\text{diag}(\sigma^2)|} = \log\frac{1}{\prod_j \sigma_j^2} = -\sum_j \log \sigma_j^2\)

**Term 2:** \(-d = -d_z\)

**Term 3:** \(\text{tr}(\Sigma_2^{-1}\Sigma_1) = \text{tr}(I \cdot \text{diag}(\sigma^2)) = \text{tr}(\text{diag}(\sigma^2)) = \sum_j \sigma_j^2\)

**Term 4:** \((\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1) = (0-\mu)^T I (0-\mu) = \mu^T\mu = \sum_j \mu_j^2\)

**Putting it together:**
\begin{align}
D_{\text{KL}} &= \frac{1}{2}\left[-\sum_j \log \sigma_j^2 - d_z + \sum_j \sigma_j^2 + \sum_j \mu_j^2\right] \\
&= \frac{1}{2}\sum_{j=1}^{d_z}\left(-\log \sigma_j^2 - 1 + \sigma_j^2 + \mu_j^2\right) \\
&= \frac{1}{2}\sum_{j=1}^{d_z}\left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)
\end{align}

### Code Implementation

```python
def kl_divergence(mu, log_var):
    """
    KL divergence between N(mu, sigma^2) and N(0, I)
    
    Args:
        mu: (batch_size, latent_dim) - encoder mean
        log_var: (batch_size, latent_dim) - encoder log variance
    
    Returns:
        kl: (batch_size,) - KL divergence for each sample
    """
    # Formula: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    # With log_var = log(sigma^2):
    # = 0.5 * sum(mu^2 + exp(log_var) - log_var - 1)
    
    kl = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1, dim=1)
    return kl  # (batch_size,)
```

### Numerical Example

Let \(d_z = 2\), \(\mu = [0.5, -0.3]\), \(\sigma^2 = [1.2, 0.8]\):

For dimension 1:
- \(\mu_1^2 = 0.25\)
- \(\sigma_1^2 = 1.2\)
- \(\log \sigma_1^2 = \log(1.2) \approx 0.182\)
- Term: \(0.25 + 1.2 - 0.182 - 1 = 0.268\)

For dimension 2:
- \(\mu_2^2 = 0.09\)
- \(\sigma_2^2 = 0.8\)
- \(\log \sigma_2^2 = \log(0.8) \approx -0.223\)
- Term: \(0.09 + 0.8 - (-0.223) - 1 = 0.113\)

Total KL: \(\frac{1}{2}(0.268 + 0.113) = 0.191\)

---

## 2.6 How Neural Networks Become Distributions

### The Fundamental Question

**"The decoder outputs a vector—how is it a distribution?"**

This causes more confusion than almost anything else in VAEs. Let's be absolutely clear.

### A Neural Network Outputs Distribution Parameters

The decoder network is a function \(f_\theta: \mathbb{R}^{d_z} \to \mathbb{R}^k\) that maps latent codes to **parameters**.

These parameters specify a probability distribution over \(x\).

**Example: Gaussian Decoder**

```python
class GaussianDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(256, output_dim)     # Mean
        self.log_var_layer = nn.Linear(256, output_dim)  # Log variance
    
    def forward(self, z):
        h = self.network(z)
        mu = self.mu_layer(h)           # Parameters of p(x|z)
        log_var = self.log_var_layer(h)
        return mu, log_var
```

The network outputs \(\mu_\theta(z)\) and \(\log \sigma^2_\theta(z)\).

The **distribution** is:
\[
p_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \text{diag}(\sigma^2_\theta(z)))
\]

The network itself is NOT the distribution. It's a **parameter generator**.

### What Does \(p_\theta(x|z)\) Mean for Images?

Consider a 28×28 MNIST image. \(x \in \mathbb{R}^{784}\).

#### Factorized Gaussian Model

We assume pixels are conditionally independent given \(z\):
\[
p_\theta(x|z) = \prod_{i=1}^{784} p_\theta(x_i|z) = \prod_{i=1}^{784} \mathcal{N}(x_i; \mu_i(z), \sigma_i^2(z))
\]

The decoder outputs:
- \(\mu(z) \in \mathbb{R}^{784}\): predicted mean for each pixel
- \(\sigma^2(z) \in \mathbb{R}^{784}\): predicted variance for each pixel

**Probability of a specific image:**
\[
p_\theta(x|z) = \prod_{i=1}^{784} \frac{1}{\sqrt{2\pi\sigma_i^2}} \exp\left(-\frac{(x_i - \mu_i)^2}{2\sigma_i^2}\right)
\]

**Log probability:**
\[
\log p_\theta(x|z) = \sum_{i=1}^{784} \left[-\frac{1}{2}\log(2\pi\sigma_i^2) - \frac{(x_i-\mu_i)^2}{2\sigma_i^2}\right]
\]

#### Bernoulli Model (for binary images)

For binarized images (pixels ∈ {0, 1}):
\[
p_\theta(x|z) = \prod_{i=1}^{784} \text{Bernoulli}(x_i; p_i(z)) = \prod_{i=1}^{784} p_i^{x_i}(1-p_i)^{1-x_i}
\]

The decoder outputs logits \(\ell(z) \in \mathbb{R}^{784}\), and \(p_i = \sigma(\ell_i)\).

**Log probability:**
\[
\log p_\theta(x|z) = \sum_{i=1}^{784} \left[x_i \log p_i + (1-x_i)\log(1-p_i)\right]
\]

This is the negative binary cross-entropy.

### Continuous Pixels vs Discrete Pixels

**Problem:** Real images have discrete pixel values (0-255). The Gaussian density can be > 1 for continuous densities, which is confusing.

**Solutions:**

1. **Treat as continuous:** Scale to [0,1], use Gaussian. Log-likelihood can be positive or negative depending on variance. This is a density, not a probability mass.

2. **Dequantization:** Add uniform noise to pixels before training. Converts discrete to continuous.

3. **Discretized logistic:** Model each pixel as a discretized logistic distribution (as in PixelCNN++). Sum probability mass over the pixel's bin.

4. **Bernoulli for binarized data:** Threshold images at 0.5. This is common for MNIST experiments.

### Concrete Numerical Example

**Setup:** 4-pixel image (2×2), \(x = [0.2, 0.8, 0.5, 0.3]\)

**Decoder outputs (Gaussian):**
- \(\mu = [0.25, 0.75, 0.45, 0.35]\)
- \(\sigma^2 = [0.01, 0.01, 0.01, 0.01]\) (fixed small variance)

**Computing log p(x|z):**

For each pixel \(i\):
\[
\log p(x_i|z) = -\frac{1}{2}\log(2\pi \cdot 0.01) - \frac{(x_i - \mu_i)^2}{2 \cdot 0.01}
\]

\(\log(2\pi \cdot 0.01) = \log(0.0628) \approx -2.77\), so \(-\frac{1}{2} \cdot (-2.77) = 1.385\)

| Pixel | \(x_i\) | \(\mu_i\) | \((x_i-\mu_i)^2\) | \(-\frac{(x_i-\mu_i)^2}{0.02}\) | Total |
|-------|---------|-----------|-------------------|--------------------------------|-------|
| 1 | 0.2 | 0.25 | 0.0025 | -0.125 | 1.26 |
| 2 | 0.8 | 0.75 | 0.0025 | -0.125 | 1.26 |
| 3 | 0.5 | 0.45 | 0.0025 | -0.125 | 1.26 |
| 4 | 0.3 | 0.35 | 0.0025 | -0.125 | 1.26 |

**Total:** \(\log p(x|z) = 4 \times 1.26 = 5.04\)

This positive value is expected for continuous densities with small variance. It says this reconstruction is quite likely under the model.

---

## 2.7 The Full VAE Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         VAE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   INPUT x ──┬──────────────────────────────────┬─────────────── │
│             │                                  │                 │
│             ▼                                  │                 │
│   ┌─────────────────┐                         │                 │
│   │    ENCODER      │                         │                 │
│   │   q_φ(z|x)      │                         │                 │
│   │                 │                         │                 │
│   │  Neural Net     │                         │                 │
│   │  x → (μ, σ²)    │                         │                 │
│   └────────┬────────┘                         │                 │
│            │ μ, log(σ²)                       │                 │
│            ▼                                  │                 │
│   ┌─────────────────┐                         │                 │
│   │ REPARAMETERIZE  │                         │                 │
│   │ z = μ + σ⊙ε    │◀──── ε ~ N(0,I)         │                 │
│   └────────┬────────┘                         │                 │
│            │ z                                │                 │
│            ▼                                  ▼                 │
│   ┌─────────────────┐               ┌─────────────────┐        │
│   │    DECODER      │               │   KL LOSS       │        │
│   │   p_θ(x|z)      │               │                 │        │
│   │                 │               │ KL(q_φ(z|x)||   │        │
│   │  Neural Net     │               │    p(z))        │        │
│   │  z → (μ_x, σ²_x)│               └────────┬────────┘        │
│   └────────┬────────┘                        │                 │
│            │ reconstruction                  │                 │
│            │ distribution                    │                 │
│            ▼                                 │                 │
│   ┌─────────────────┐                        │                 │
│   │  RECONSTRUCTION │                        │                 │
│   │      LOSS       │                        │                 │
│   │                 │                        │                 │
│   │ -log p_θ(x|z)   │                        │                 │
│   │ (compare to x)  │                        │                 │
│   └────────┬────────┘                        │                 │
│            │                                 │                 │
│            └────────────────┬────────────────┘                 │
│                             │                                   │
│                             ▼                                   │
│                    ┌─────────────────┐                         │
│                    │   TOTAL LOSS    │                         │
│                    │                 │                         │
│                    │ -ELBO =         │                         │
│                    │ Recon + KL      │                         │
│                    └─────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Training vs Generation

**Training (inference mode):**
1. Take real data \(x\)
2. Encode: \(\mu, \log\sigma^2 = \text{Encoder}(x)\)
3. Sample: \(z = \mu + \sigma \odot \epsilon\)
4. Decode: \(\mu_x = \text{Decoder}(z)\)
5. Compute loss: \(-\mathcal{L} = \text{ReconLoss}(x, \mu_x) + \text{KL}(\mu, \log\sigma^2)\)
6. Backpropagate and update \(\theta, \phi\)

**Generation (sampling mode):**
1. Sample \(z \sim \mathcal{N}(0, I)\) from the prior
2. Decode: \(\mu_x = \text{Decoder}(z)\)
3. Optionally sample \(x \sim p_\theta(x|z)\), or just use \(\mu_x\)

### Complete Forward Pass Code

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder: x -> (mu, log_var)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: z -> reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # Output in [0, 1] for pixel values
        )
    
    def encode(self, x):
        """x: (batch, input_dim) -> mu, log_var: (batch, latent_dim)"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Sample z from q(z|x) using reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        """z: (batch, latent_dim) -> reconstruction: (batch, input_dim)"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass for training"""
        mu, log_var = self.encode(x)       # (batch, latent_dim)
        z = self.reparameterize(mu, log_var)  # (batch, latent_dim)
        x_recon = self.decode(z)           # (batch, input_dim)
        return x_recon, mu, log_var
    
    def sample(self, num_samples, device):
        """Generate new samples from the model"""
        z = torch.randn(num_samples, self.fc_mu.out_features).to(device)
        return self.decode(z)
```

---

## 2.8 Recap

### Mathematical Summary

| Object | Symbol | Meaning |
|--------|--------|---------|
| Data | \(x\) | Observed (e.g., image) |
| Latent | \(z\) | Hidden representation |
| Prior | \(p(z)\) | \(\mathcal{N}(0, I)\) |
| Decoder | \(p_\theta(x\|z)\) | Generative model |
| Encoder | \(q_\phi(z\|x)\) | Approximate posterior |
| ELBO | \(\mathcal{L}\) | \(\mathbb{E}_q[\log p_\theta(x\|z)] - D_{KL}(q\|p)\) |

### Key Equations

1. **ELBO:** \(\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))\)

2. **Bound:** \(\log p_\theta(x) \geq \mathcal{L}\)

3. **Gap:** \(\log p_\theta(x) - \mathcal{L} = D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))\)

4. **KL (Gaussian):** \(D_{\text{KL}} = \frac{1}{2}\sum_j (\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)\)

5. **Reparameterization:** \(z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)\)

### Intuitive Summary

- **Encoder learns** to map data to a distribution over latent codes
- **Decoder learns** to reconstruct data from latent codes
- **KL term** regularizes the latent space to match the prior
- **Reconstruction term** ensures we can recover the original data
- **Reparameterization** enables gradient-based training

### What to Draw

1. **ELBO components diagram:** Two arrows pushing the same \(\log p(x)\) up: reconstruction and KL regularization
2. **Training curve:** Plot reconstruction loss and KL loss separately over epochs
3. **Latent space:** 2D scatter plot of encoded training data (colored by class if labels available)

---

**Next:** Part 3 covers implementation details with complete, runnable code.


