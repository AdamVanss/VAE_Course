# Appendix A: Complete Mathematical Derivations

## A.1 ELBO Derivation (Three Methods)

### Method 1: Jensen's Inequality

**Starting point:** The log marginal likelihood.
\[
\log p_\theta(x) = \log \int p_\theta(x, z) \, dz
\]

**Step 1:** Multiply and divide by \(q_\phi(z|x)\):
\[
\log p_\theta(x) = \log \int q_\phi(z|x) \frac{p_\theta(x, z)}{q_\phi(z|x)} \, dz
\]

**Step 2:** Recognize this as an expectation:
\[
\log p_\theta(x) = \log \mathbb{E}_{q_\phi(z|x)}\left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right]
\]

**Step 3:** Apply Jensen's inequality. For concave function \(f\) (like log):
\[
f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]
\]

Therefore:
\[
\log p_\theta(x) = \log \mathbb{E}_q\left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right] \geq \mathbb{E}_q\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]
\]

**Step 4:** The right-hand side is the ELBO:
\[
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]
\]

**Step 5:** Expand using \(p_\theta(x, z) = p(z) p_\theta(x|z)\):
\begin{align}
\mathcal{L} &= \mathbb{E}_q[\log p_\theta(x, z) - \log q_\phi(z|x)] \\
&= \mathbb{E}_q[\log p(z) + \log p_\theta(x|z) - \log q_\phi(z|x)] \\
&= \mathbb{E}_q[\log p_\theta(x|z)] + \mathbb{E}_q[\log p(z) - \log q_\phi(z|x)] \\
&= \mathbb{E}_q[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))
\end{align}

---

### Method 2: KL to True Posterior

**Starting point:** KL divergence between approximate and true posterior.
\[
D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x)) = \mathbb{E}_{q}\left[\log \frac{q_\phi(z|x)}{p_\theta(z|x)}\right]
\]

**Step 1:** Apply Bayes' rule to \(p_\theta(z|x)\):
\[
p_\theta(z|x) = \frac{p_\theta(x|z) p(z)}{p_\theta(x)}
\]

**Step 2:** Substitute:
\begin{align}
D_{\text{KL}}(q \| p_\theta(z|x)) &= \mathbb{E}_q\left[\log q_\phi(z|x) - \log \frac{p_\theta(x|z) p(z)}{p_\theta(x)}\right] \\
&= \mathbb{E}_q[\log q_\phi(z|x) - \log p_\theta(x|z) - \log p(z) + \log p_\theta(x)]
\end{align}

**Step 3:** Note that \(\log p_\theta(x)\) doesn't depend on \(z\):
\[
D_{\text{KL}}(q \| p_\theta(z|x)) = \mathbb{E}_q[\log q_\phi(z|x) - \log p_\theta(x|z) - \log p(z)] + \log p_\theta(x)
\]

**Step 4:** Rearrange:
\[
\log p_\theta(x) = \mathbb{E}_q[\log p_\theta(x|z)] + \mathbb{E}_q[\log p(z) - \log q_\phi(z|x)] + D_{\text{KL}}(q \| p_\theta(z|x))
\]

**Step 5:** Recognize the ELBO:
\[
\log p_\theta(x) = \underbrace{\mathbb{E}_q[\log p_\theta(x|z)] - D_{\text{KL}}(q \| p)}_{\mathcal{L}} + D_{\text{KL}}(q \| p_\theta(z|x))
\]

Since \(D_{\text{KL}} \geq 0\):
\[
\boxed{\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x)}
\]

---

### Method 3: Importance Sampling

**Starting point:** Express marginal likelihood as importance-weighted expectation.
\[
p_\theta(x) = \int p_\theta(x|z) p(z) \, dz = \int q_\phi(z|x) \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} \, dz
\]

**Step 1:** Define importance weight:
\[
w(z) = \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)}
\]

**Step 2:** Then:
\[
p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}[w(z)]
\]

**Step 3:** Taking log and applying Jensen's:
\[
\log p_\theta(x) = \log \mathbb{E}_q[w(z)] \geq \mathbb{E}_q[\log w(z)] = \mathcal{L}
\]

This view connects to IWAE: with K samples, we get:
\[
\mathcal{L}_K = \mathbb{E}_{z^{(1:K)}}\left[\log \frac{1}{K}\sum_{k=1}^K w(z^{(k)})\right]
\]

---

## A.2 KL Divergence for Gaussians

### Univariate Case

**Goal:** Derive \(D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1))\)

**Definition:**
\[
D_{\text{KL}}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} \, dz = \mathbb{E}_q[\log q(z)] - \mathbb{E}_q[\log p(z)]
\]

**Term 1:** \(\mathbb{E}_q[\log q(z)]\) (negative entropy of q)

\[
q(z) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(z-\mu)^2}{2\sigma^2}\right)
\]

\[
\log q(z) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2}
\]

Taking expectation:
\[
\mathbb{E}_q[\log q(z)] = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\underbrace{\mathbb{E}_q[(z-\mu)^2]}_{=\sigma^2}
\]

\[
\mathbb{E}_q[\log q(z)] = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2}
\]

**Term 2:** \(\mathbb{E}_q[\log p(z)]\) (cross-entropy)

\[
p(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)
\]

\[
\log p(z) = -\frac{1}{2}\log(2\pi) - \frac{z^2}{2}
\]

Taking expectation:
\[
\mathbb{E}_q[\log p(z)] = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\underbrace{\mathbb{E}_q[z^2]}_{=\sigma^2 + \mu^2}
\]

Using \(\mathbb{E}[z^2] = \text{Var}(z) + (\mathbb{E}[z])^2 = \sigma^2 + \mu^2\):
\[
\mathbb{E}_q[\log p(z)] = -\frac{1}{2}\log(2\pi) - \frac{\sigma^2 + \mu^2}{2}
\]

**Combining:**
\begin{align}
D_{\text{KL}} &= \mathbb{E}_q[\log q] - \mathbb{E}_q[\log p] \\
&= \left(-\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2}\right) - \left(-\frac{1}{2}\log(2\pi) - \frac{\sigma^2 + \mu^2}{2}\right) \\
&= -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{1}{2} + \frac{1}{2}\log(2\pi) + \frac{\sigma^2}{2} + \frac{\mu^2}{2} \\
&= -\frac{1}{2}\log(\sigma^2) + \frac{\sigma^2}{2} + \frac{\mu^2}{2} - \frac{1}{2}
\end{align}

\[
\boxed{D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2}\left(\mu^2 + \sigma^2 - \log \sigma^2 - 1\right)}
\]

### Multivariate Case (Diagonal Covariance)

For independent dimensions:
\[
D_{\text{KL}}(\mathcal{N}(\mu, \text{diag}(\sigma^2)) \| \mathcal{N}(0, I)) = \sum_{j=1}^d D_{\text{KL}}(\mathcal{N}(\mu_j, \sigma_j^2) \| \mathcal{N}(0, 1))
\]

\[
\boxed{= \frac{1}{2}\sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)}
\]

---

## A.3 Reparameterization Gradient

### The Problem

We want: \(\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)]\)

Where \(q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \sigma^2_\phi(x))\)

**Naive approach fails:**
\[
\nabla_\phi \int q_\phi(z|x) f(z) \, dz
\]
The gradient w.r.t. \(\phi\) acts on the distribution \(q_\phi\), not a simple deterministic function.

### Reparameterization Solution

**Step 1:** Express \(z\) as a deterministic function of \(\phi\) and independent noise \(\epsilon\):
\[
z = g_\phi(\epsilon, x) = \mu_\phi(x) + \sigma_\phi(x) \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

**Step 2:** Rewrite expectation:
\[
\mathbb{E}_{q_\phi(z|x)}[f(z)] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[f(g_\phi(\epsilon, x))]
\]

**Step 3:** Now \(p(\epsilon)\) doesn't depend on \(\phi\), so:
\[
\nabla_\phi \mathbb{E}_{\epsilon}[f(g_\phi(\epsilon, x))] = \mathbb{E}_{\epsilon}[\nabla_\phi f(g_\phi(\epsilon, x))]
\]

**Step 4:** By chain rule:
\[
\nabla_\phi f(g_\phi(\epsilon, x)) = \nabla_z f(z)|_{z=g_\phi} \cdot \nabla_\phi g_\phi(\epsilon, x)
\]

Where:
\[
\nabla_\phi g_\phi = \nabla_\phi (\mu_\phi + \sigma_\phi \cdot \epsilon) = \nabla_\phi \mu_\phi + \epsilon \cdot \nabla_\phi \sigma_\phi
\]

**Monte Carlo estimate:**
\[
\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] \approx \frac{1}{L}\sum_{\ell=1}^L \nabla_\phi f(\mu_\phi + \sigma_\phi \cdot \epsilon^{(\ell)})
\]

This is just standard backpropagation through the computation \(z = \mu + \sigma \cdot \epsilon\).

---

## A.4 Gaussian Log-Likelihood

### Single Observation

For \(x \in \mathbb{R}^D\) and parameters \(\mu, \sigma^2 \in \mathbb{R}^D\) (diagonal covariance):

\[
\log p(x | \mu, \sigma^2) = \log \prod_{i=1}^D \mathcal{N}(x_i; \mu_i, \sigma_i^2)
\]

\[
= \sum_{i=1}^D \log \mathcal{N}(x_i; \mu_i, \sigma_i^2)
\]

\[
= \sum_{i=1}^D \left[-\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma_i^2) - \frac{(x_i - \mu_i)^2}{2\sigma_i^2}\right]
\]

\[
= -\frac{D}{2}\log(2\pi) - \frac{1}{2}\sum_{i=1}^D \log(\sigma_i^2) - \frac{1}{2}\sum_{i=1}^D \frac{(x_i - \mu_i)^2}{\sigma_i^2}
\]

### Fixed Variance Case

If \(\sigma_i^2 = \sigma^2\) for all \(i\):

\[
\log p(x | \mu, \sigma^2) = -\frac{D}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^D (x_i - \mu_i)^2
\]

\[
= -\frac{D}{2}\log(2\pi\sigma^2) - \frac{\|x - \mu\|^2}{2\sigma^2}
\]

### Connection to MSE

Maximizing log-likelihood = Minimizing negative log-likelihood.

Ignoring constants w.r.t. \(\mu\):
\[
-\log p(x|\mu, \sigma^2) \propto \frac{\|x - \mu\|^2}{2\sigma^2}
\]

With \(\sigma^2 = 1\): Minimizing NLL ≡ Minimizing \(\frac{1}{2}\|x - \mu\|^2\) ≡ Minimizing MSE (up to factor of 2).

---

## A.5 Binary Cross-Entropy Derivation

### Bernoulli Likelihood

For binary data \(x_i \in \{0, 1\}\) with parameter \(p_i \in (0, 1)\):

\[
p(x_i | p_i) = p_i^{x_i} (1-p_i)^{1-x_i}
\]

\[
\log p(x_i | p_i) = x_i \log p_i + (1-x_i) \log(1-p_i)
\]

### For Image with D Pixels

\[
\log p(x | p) = \sum_{i=1}^D [x_i \log p_i + (1-x_i) \log(1-p_i)]
\]

The **negative** of this is binary cross-entropy:
\[
\text{BCE}(x, p) = -\sum_{i=1}^D [x_i \log p_i + (1-x_i) \log(1-p_i)]
\]

### With Logits

If \(p_i = \sigma(\ell_i)\) where \(\sigma\) is sigmoid and \(\ell_i\) is the logit:

\[
\text{BCE}(x, \ell) = -\sum_{i=1}^D [x_i \log \sigma(\ell_i) + (1-x_i) \log(1-\sigma(\ell_i))]
\]

Using \(\log(1-\sigma(\ell)) = -\ell - \log(1+e^{-\ell}) = \log\sigma(-\ell)\):

\[
\text{BCE}(x, \ell) = \sum_{i=1}^D [\ell_i - x_i \ell_i + \log(1 + e^{-\ell_i})]
\]

This is what `F.binary_cross_entropy_with_logits` computes (numerically stable).

---

## A.6 General KL Between Two Gaussians

For \(\mathcal{N}(\mu_1, \Sigma_1)\) and \(\mathcal{N}(\mu_2, \Sigma_2)\):

\[
D_{\text{KL}}(\mathcal{N}_1 \| \mathcal{N}_2) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1)\right]
\]

**Special case** (both diagonal, \(\Sigma = \text{diag}(\sigma^2)\)):

\[
D_{\text{KL}} = \frac{1}{2}\sum_{j=1}^d \left[\log\frac{\sigma_{2,j}^2}{\sigma_{1,j}^2} - 1 + \frac{\sigma_{1,j}^2}{\sigma_{2,j}^2} + \frac{(\mu_{2,j}-\mu_{1,j})^2}{\sigma_{2,j}^2}\right]
\]

**For VAE KL** (\(\mu_1 = \mu\), \(\sigma_1 = \sigma\), \(\mu_2 = 0\), \(\sigma_2 = 1\)):

\[
D_{\text{KL}} = \frac{1}{2}\sum_{j=1}^d \left[-\log\sigma_j^2 - 1 + \sigma_j^2 + \mu_j^2\right]
\]

Which matches our earlier derivation.


