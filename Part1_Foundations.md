# Part 1: Foundations and Mathematical Background

## Table of Contents
- [1.1 What is a Generative Model?](#11-what-is-a-generative-model)
- [1.2 The Data Distribution vs Model Distribution](#12-the-data-distribution-vs-model-distribution)
- [1.3 Latent Variable Models](#13-latent-variable-models)
- [1.4 Probability Distributions Review](#14-probability-distributions-review)
- [1.5 Maximum Likelihood Estimation](#15-maximum-likelihood-estimation)
- [1.6 The Intractability Problem](#16-the-intractability-problem)
- [1.7 Recap](#17-recap)

---

## 1.1 What is a Generative Model?

### The Core Question

A **generative model** answers a fundamental question: *Given a collection of data (images, text, audio), can we build a machine that creates new, plausible examples that look like they came from the same source?*

More formally, suppose we have a dataset \(\mathcal{D} = \{x^{(1)}, x^{(2)}, \ldots, x^{(N)}\}\) where each \(x^{(i)}\) is a data point (say, an image represented as a vector of pixel values). These data points are assumed to be drawn independently from some unknown **true data distribution** \(p_{\text{data}}(x)\).

Our goal is to learn a **model distribution** \(p_\theta(x)\), parameterized by \(\theta\), such that:
1. \(p_\theta(x)\) assigns high probability to data points that "look real"
2. We can **sample** from \(p_\theta(x)\) to generate new examples
3. We can **evaluate** \(p_\theta(x)\) to assess how likely a given point is

### Why Generative Models Matter

Generative models are not just about creating pretty pictures. They represent:

1. **Density Estimation:** Understanding the structure of data
2. **Unsupervised Learning:** Learning representations without labels
3. **Data Augmentation:** Creating synthetic training data
4. **Anomaly Detection:** Low \(p_\theta(x)\) indicates outliers
5. **Compression:** Efficient encoding requires understanding data structure
6. **Scientific Modeling:** Understanding the generative process of phenomena

### Types of Generative Models

| Model Type | Examples | Key Idea |
|------------|----------|----------|
| Explicit Density, Tractable | Autoregressive (PixelCNN, GPT), Normalizing Flows | Directly compute \(p_\theta(x)\) |
| Explicit Density, Approximate | VAEs, Boltzmann Machines | Lower bound on \(p_\theta(x)\) |
| Implicit Density | GANs, Score Matching | Learn to sample without explicit density |

VAEs fall into the "Explicit Density, Approximate" category—we cannot compute \(p_\theta(x)\) exactly, but we can optimize a lower bound on it.

### What Does "Lower Bound" Mean Intuitively?

When we say VAEs optimize a **lower bound** on \(p_\theta(x)\), imagine you want to measure the height of a mountain, but clouds obscure the peak. You can't see the true height, but you can always see *at least* up to the cloud line. The cloud line is your lower bound—it's guaranteed to be less than or equal to the true height.

Similarly, for VAEs:
- The true quantity \(\log p_\theta(x)\) (how likely our model thinks the data is) is impossible to compute directly
- We derive a quantity called the **ELBO** (Evidence Lower Bound) that is always ≤ \(\log p_\theta(x)\)
- When we maximize the ELBO, we push this "floor" upward, which necessarily pushes up the true likelihood too

**Why is this useful?** If we keep raising the floor (ELBO), the ceiling (true log-likelihood) must also rise. Even though we never directly touch \(\log p_\theta(x)\), maximizing its lower bound still improves our model.

---

## 1.2 The Data Distribution vs Model Distribution

### A Critical Conceptual Distinction

**This concept causes endless confusion, so let's be extremely clear.**

#### The True Data Distribution \(p_{\text{data}}(x)\)

This is a **philosophical object**. We never observe it directly. We only see samples from it (our dataset).

Think of it this way: imagine there's a magical oracle that generates images of cats. Every image of a cat that could possibly exist has some probability under this oracle's distribution. Some images (a normal photo of a tabby cat) have high probability. Others (a cat with 17 legs) have near-zero probability.

We don't have access to this oracle. We only have a finite collection of images \(\mathcal{D}\) that someone sampled from the oracle.

#### The Empirical Distribution \(\hat{p}_{\text{data}}(x)\)

Given our finite dataset, we can define:
\[
\hat{p}_{\text{data}}(x) = \frac{1}{N} \sum_{i=1}^{N} \delta(x - x^{(i)})
\]

**What is the Dirac Delta Function?**

The Dirac delta \(\delta(x)\) is a mathematical object (technically a "distribution," not a function) with these properties:
- \(\delta(x) = 0\) everywhere except at \(x = 0\)
- At \(x = 0\), it's "infinitely tall" but "infinitely narrow"
- Its total integral equals 1: \(\int_{-\infty}^{\infty} \delta(x) dx = 1\)

Think of it as the limit of increasingly narrow, tall spikes. A Gaussian \(\mathcal{N}(0, \sigma^2)\) becomes a delta function as \(\sigma \to 0\): all probability concentrates at a single point.

**Why use it here?**

When we write \(\delta(x - x^{(i)})\), we're placing a "spike" of probability exactly at data point \(x^{(i)}\). The empirical distribution says: *"The probability is concentrated entirely on the exact data points we observed, nowhere else."*

This is useful because:
1. It's the most "honest" representation—we only have evidence for these exact points
2. When we take expectations over \(\hat{p}_{\text{data}}\), we simply average over our dataset:
   \[
   \mathbb{E}_{x \sim \hat{p}_{\text{data}}}[f(x)] = \frac{1}{N}\sum_{i=1}^{N} f(x^{(i)})
   \]

For practical purposes, when we say "data distribution," we usually mean this empirical distribution.

#### The Model Distribution \(p_\theta(x)\)

This is what we're learning. It's a parametric family of distributions—a mathematical formula with adjustable knobs (\(\theta\)).

**Example:** If we model \(p_\theta(x)\) as a Gaussian, then \(\theta = (\mu, \Sigma)\) and:
\[
p_\theta(x) = \mathcal{N}(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)
\]

For VAEs, \(p_\theta(x)\) is defined through a more complex generative process involving latent variables.

### The Training Objective

We want to adjust \(\theta\) so that \(p_\theta(x)\) becomes "close to" \(p_{\text{data}}(x)\). 

The standard measure of "closeness" between two distributions is the **KL divergence**:
\[
D_{\text{KL}}(p_{\text{data}} \| p_\theta) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\log \frac{p_{\text{data}}(x)}{p_\theta(x)}\right]
\]

Minimizing this is equivalent to maximizing the **expected log-likelihood**:
\[
\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)] \approx \frac{1}{N}\sum_{i=1}^{N} \log p_\theta(x^{(i)})
\]

**Key insight:** We don't need to know \(p_{\text{data}}(x)\) to minimize the KL divergence—we only need samples from it (our dataset) and the ability to evaluate \(\log p_\theta(x)\).

---

## 1.3 Latent Variable Models

### The Idea of Latent Variables

Real data (like images) lives in a high-dimensional space, but the "interesting" variation often lies on a much lower-dimensional **manifold**.

**Intuition:** Consider images of faces:
- Each 64×64 grayscale image is a point in \(\mathbb{R}^{4096}\)
- But the space of "valid faces" is much smaller
- Faces vary by identity, pose, expression, lighting—perhaps a few hundred dimensions
- Most random points in \(\mathbb{R}^{4096}\) don't look like faces at all

**Latent variables** \(z\) capture this low-dimensional structure. We imagine that:
1. Nature first samples a low-dimensional "code" \(z\) from some simple prior \(p(z)\)
2. Then nature generates the high-dimensional observation \(x\) from \(p_\theta(x|z)\)

### The Generative Story

A **latent variable model** posits the following generative process:

```
1. Sample z ~ p(z)           # Draw from the prior (e.g., standard normal)
2. Sample x ~ p_θ(x|z)       # Draw from the conditional (decoder)
```

The joint distribution is:
\[
p_\theta(x, z) = p(z) \cdot p_\theta(x|z)
\]

The **marginal likelihood** (the probability of observing \(x\)) is obtained by integrating out \(z\):
\[
p_\theta(x) = \int p(z) \cdot p_\theta(x|z) \, dz
\]

This integral is the source of both the power and the difficulty of VAEs.

### Why Latent Variables Are Powerful

1. **Compression:** \(z\) is a compact representation of \(x\)
2. **Disentanglement:** Different dimensions of \(z\) might correspond to interpretable factors
3. **Interpolation:** We can smoothly interpolate between data points in \(z\)-space
4. **Conditional generation:** Fix some aspects of \(z\), vary others
5. **Semantic manipulation:** Vector arithmetic in latent space (e.g., "face + smile")

### The Decoder: \(p_\theta(x|z)\)

The decoder defines a distribution over \(x\) given \(z\). In neural network terms:
- Input: latent code \(z \in \mathbb{R}^{d_z}\)
- Output: parameters of a distribution over \(x \in \mathbb{R}^{d_x}\)

**Crucial point:** The decoder neural network does NOT output an image directly. It outputs **parameters of a probability distribution** over images.

**Example (Gaussian decoder):**
```python
class Decoder(nn.Module):
    def forward(self, z):
        # z: (batch_size, latent_dim)
        h = self.network(z)                    # (batch_size, hidden_dim)
        mu_x = self.mu_layer(h)                # (batch_size, image_dim) - mean
        log_var_x = self.log_var_layer(h)      # (batch_size, image_dim) - log variance
        return mu_x, log_var_x
        # These are PARAMETERS of p(x|z) = N(x; mu_x, exp(log_var_x))
```

We will elaborate on this extensively in Part 2.

---

## 1.4 Probability Distributions Review

### Gaussian (Normal) Distribution

The most important distribution in VAEs. For a single variable:
\[
\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
\]

The **log probability** is:
\[
\log \mathcal{N}(x; \mu, \sigma^2) = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}
\]

**What is a Multivariate Gaussian?**

A **multivariate Gaussian** (or multivariate normal) is the extension of the bell curve to multiple dimensions. Instead of a single variable \(x\), you have a vector \(x = [x_1, x_2, \ldots, x_d]\).

**Intuition:** Imagine a 2D case. A univariate Gaussian is a bell curve (1D bump). A bivariate Gaussian is a 3D "hill" or "blob" when viewed from above—it looks like an ellipse of contour lines centered at the mean \(\mu\).

- The **mean** \(\mu = [\mu_1, \mu_2, \ldots, \mu_d]\) is the center of the blob
- The **covariance matrix** \(\Sigma\) controls the shape: how stretched or rotated the ellipse is

**Diagonal covariance** (what VAEs typically use) means each dimension is independent:
- \(\Sigma = \text{diag}(\sigma_1^2, \sigma_2^2, \ldots, \sigma_d^2)\)
- The ellipse axes align with the coordinate axes (no rotation)
- The probability factorizes: \(p(x) = p(x_1) \cdot p(x_2) \cdot \ldots \cdot p(x_d)\)

For a **multivariate** Gaussian with diagonal covariance (independent dimensions):
\[
\log \mathcal{N}(x; \mu, \text{diag}(\sigma^2)) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\sum_{i=1}^{d}\log(\sigma_i^2) - \frac{1}{2}\sum_{i=1}^{d}\frac{(x_i-\mu_i)^2}{\sigma_i^2}
\]

This is simply the sum of \(d\) independent univariate Gaussian log-probabilities.

### Numerical Example

Let's compute a concrete log-likelihood. Suppose we have a tiny "image" with 3 pixels: \(x = [0.2, 0.5, 0.9]\).

Our decoder outputs:
- \(\mu = [0.3, 0.4, 0.8]\)
- \(\sigma^2 = [0.01, 0.01, 0.01]\) (i.e., \(\sigma = 0.1\) for each pixel)

The log-likelihood is:
\[
\log p(x|\mu, \sigma^2) = -\frac{3}{2}\log(2\pi) - \frac{1}{2}(3 \times \log(0.01)) - \frac{1}{2}\left[\frac{(0.2-0.3)^2}{0.01} + \frac{(0.5-0.4)^2}{0.01} + \frac{(0.9-0.8)^2}{0.01}\right]
\]

Computing step by step:
- \(\log(2\pi) \approx 1.8379\), so \(-\frac{3}{2}\log(2\pi) \approx -2.7568\)
- \(\log(0.01) = \log(10^{-2}) = -2\ln(10) \approx -4.6052\), so \(-\frac{1}{2}(3 \times -4.6052) \approx 6.9078\)
- Squared errors: \(0.01 + 0.01 + 0.01 = 0.03\)
- Divided by variance: \(\frac{0.03}{0.01} = 3\)
- So \(-\frac{1}{2}(3) = -1.5\)

Total: \(-2.7568 + 6.9078 - 1.5 = 2.651\)

This positive log-likelihood indicates this reconstruction is reasonably good given our variance assumption.

### Bernoulli Distribution

For binary data (e.g., black/white pixels):
\[
\text{Bernoulli}(x; p) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}
\]

Log probability:
\[
\log \text{Bernoulli}(x; p) = x \log(p) + (1-x) \log(1-p)
\]

This is the **binary cross-entropy**. When the decoder outputs logits \(\ell\), we have \(p = \sigma(\ell) = \frac{1}{1+e^{-\ell}}\).

For an image with \(d\) binary pixels:
\[
\log p(x|\ell) = \sum_{i=1}^{d} \left[x_i \log(\sigma(\ell_i)) + (1-x_i)\log(1-\sigma(\ell_i))\right]
\]

This is equivalent to the negative of `F.binary_cross_entropy_with_logits(logits, target, reduction='sum')`.

### KL Divergence

The KL divergence measures how one distribution \(q\) differs from a reference distribution \(p\):
\[
D_{\text{KL}}(q \| p) = \mathbb{E}_{x \sim q}\left[\log \frac{q(x)}{p(x)}\right] = \mathbb{E}_{x \sim q}[\log q(x)] - \mathbb{E}_{x \sim q}[\log p(x)]
\]

**Why is it an expectation?**

The KL divergence asks: *"On average, how surprised would I be if I thought data came from \(p\), but it actually came from \(q\)?"*

Here's the intuition:
1. We sample points from \(q\) (where the data actually comes from)
2. For each point, we compute \(\log \frac{q(x)}{p(x)}\)—the log-ratio of probabilities
3. We average this over all samples

**Why this formula makes sense:**
- If \(q(x) > p(x)\) at some point, then \(\log \frac{q(x)}{p(x)} > 0\): distribution \(q\) puts more mass here than \(p\) expected
- If \(q(x) < p(x)\), then \(\log \frac{q(x)}{p(x)} < 0\): \(q\) puts less mass than \(p\) expected
- We weight these by \(q(x)\) (the expectation under \(q\)) because we care about regions where \(q\) actually puts probability

**Simple analogy:** Imagine you're a weather forecaster. \(p\) is your predicted rain probability, \(q\) is the actual weather. KL divergence measures how "wrong" your predictions were, on average, weighted by what actually happened.

Properties:
- \(D_{\text{KL}}(q \| p) \geq 0\) always (Gibbs' inequality)
- \(D_{\text{KL}}(q \| p) = 0\) if and only if \(q = p\) almost everywhere
- NOT symmetric: \(D_{\text{KL}}(q \| p) \neq D_{\text{KL}}(p \| q)\) in general

---

## 1.5 Maximum Likelihood Estimation

### The MLE Principle

**Understanding argmax and argmin**

Before diving into MLE, let's clarify what \(\arg\max\) and \(\arg\min\) mean:

- \(\max_\theta f(\theta)\) returns the **maximum value** of function \(f\)
- \(\arg\max_\theta f(\theta)\) returns the **argument** (input) \(\theta\) that achieves that maximum

**Simple example:** Consider \(f(\theta) = -(\theta - 3)^2 + 10\). This is a downward parabola.
- \(\max_\theta f(\theta) = 10\) (the highest point on the curve)
- \(\arg\max_\theta f(\theta) = 3\) (the \(\theta\) value where the maximum occurs)

Similarly, \(\arg\min_\theta g(\theta)\) returns the \(\theta\) that minimizes \(g\).

**Example:** For \(g(\theta) = (\theta - 5)^2\):
- \(\min_\theta g(\theta) = 0\)
- \(\arg\min_\theta g(\theta) = 5\)

Now, the MLE principle:

Given data \(\mathcal{D} = \{x^{(1)}, \ldots, x^{(N)}\}\), find parameters \(\theta\) that maximize the likelihood of observing this data:
\[
\theta^* = \arg\max_\theta \prod_{i=1}^{N} p_\theta(x^{(i)})
\]

Equivalently, maximize the log-likelihood:
Logarithm is an increasing function. (x grows if and only if log(x) grows)
\[
\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log p_\theta(x^{(i)})
\]

Or minimize the negative log-likelihood:
\[
\theta^* = \arg\min_\theta -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(x^{(i)})
\]

### Why Log?

1. **Numerical stability:** Products of many small probabilities underflow; sums of logs don't
2. **Decomposition:** Log turns products into sums, enabling mini-batch training
3. **Interpretation:** Negative log-likelihood is related to **bits** needed to encode data

**The Bits Interpretation (Information Theory Connection)**

In information theory, the **entropy** of a distribution measures the average number of bits needed to encode samples from it. The key insight: \(-\log_2 p(x)\) gives the number of bits needed to encode an event with probability \(p(x)\).

- If \(p(x) = 1\) (certain event): \(-\log_2(1) = 0\) bits needed
- If \(p(x) = 0.5\): \(-\log_2(0.5) = 1\) bit needed
- If \(p(x) = 0.25\): \(-\log_2(0.25) = 2\) bits needed
- If \(p(x) = 0.001\) (rare event): \(-\log_2(0.001) \approx 10\) bits needed

**Why this matters for VAEs:**

When we minimize negative log-likelihood \(-\log p_\theta(x)\), we're essentially asking: *"How many bits does our model need to encode this data?"*

- A good model assigns high probability to real data → fewer bits → lower loss
- A bad model assigns low probability → many bits → higher loss

Using natural log (\(\ln\)) instead of \(\log_2\) gives units of "nats" instead of bits (1 nat ≈ 1.44 bits), but the intuition is identical.

### Connection to Cross-Entropy

The negative log-likelihood averaged over the dataset:
\[
-\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(x^{(i)}) \approx -\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)] = H(p_{\text{data}}, p_\theta)
\]

This is the **cross-entropy** between the data distribution and model distribution.

---

## 1.6 The Intractability Problem

### What Does "Intractable" Mean?

In mathematics and computer science, a problem is **intractable** when it cannot be solved exactly in a reasonable amount of time or with finite resources. There are different flavors:

1. **Computationally intractable:** The exact solution exists but would take exponentially long to compute (e.g., brute-force search over all possibilities)

2. **Analytically intractable:** No closed-form mathematical expression exists. You can't write down a formula—the integral or sum has no nice solution.

3. **Practically intractable:** The solution might exist theoretically, but computing it requires resources we don't have (infinite samples, infinite precision, etc.)

In VAEs, we face **analytical intractability**: the integral defining \(p_\theta(x)\) has no closed-form solution for neural network decoders.

### Why Can't We Just Maximize \(\log p_\theta(x)\)?

For latent variable models:
\[
p_\theta(x) = \int p(z) p_\theta(x|z) \, dz
\]

**Problem 1: The integral is intractable.** 
For most interesting models (neural network decoders), this integral has no closed form. We cannot compute \(p_\theta(x)\) exactly.

**Problem 2: Even if we could compute it, gradients are hard.**
We'd need:
\[
\nabla_\theta \log p_\theta(x) = \nabla_\theta \log \int p(z) p_\theta(x|z) \, dz
\]
The gradient doesn't pass through the integral easily.

**Problem 3: The posterior is intractable.**
Given data \(x\), what's the distribution over latent codes that could have generated it?
\[
p_\theta(z|x) = \frac{p(z) p_\theta(x|z)}{p_\theta(x)}
\]
This requires \(p_\theta(x)\), which we can't compute.

### Attempted Solutions and Why They Fail

**Monte Carlo estimation:** 
\[
p_\theta(x) \approx \frac{1}{L}\sum_{\ell=1}^{L} p_\theta(x|z^{(\ell)}), \quad z^{(\ell)} \sim p(z)
\]

This estimator has extremely high variance! Most samples from the prior \(p(z)\) produce \(z\) values that don't match \(x\) at all, giving near-zero \(p_\theta(x|z)\).

**Example:** Imagine trying to generate a specific face. Random samples from a standard normal are almost never the right latent code for that face.

### The VAE Solution: Variational Inference

VAEs solve this through **variational inference**:
1. Introduce an **approximate posterior** \(q_\phi(z|x)\) (the encoder)
2. Derive a **lower bound** on \(\log p_\theta(x)\) called the ELBO
3. Maximize this lower bound instead

The genius is that both the bound and its gradients are tractable.

---

## 1.7 Recap

Let's restate the key ideas from multiple perspectives:

### Mathematical Perspective
- We want to maximize \(\log p_\theta(x)\) (log-likelihood of data)
- This is intractable for latent variable models
- We'll maximize a lower bound (ELBO) instead

### Intuitive Perspective
- Data lies on a low-dimensional manifold in high-dimensional space
- Latent variables capture this low-dimensional structure
- We need both a way to go from latent to data (decoder) and data to latent (encoder)

### Computational Perspective
- The marginal likelihood integral has no closed form
- Monte Carlo estimation from the prior has astronomical variance
- We need importance sampling with a good proposal distribution—the encoder

### Diagrams to Draw

**Diagram 1: The Generative Process**
```
   p(z)          p_θ(x|z)
┌────────┐      ┌────────┐
│ Prior  │─────▶│Decoder │─────▶ x (observed data)
│  N(0,I)│  z   │  (NN)  │
└────────┘      └────────┘
```

**Diagram 2: The Inference Problem**
```
x (observed) ──?──▶ z (latent)

We want p_θ(z|x), but it's intractable.
Solution: Learn q_φ(z|x) to approximate it.
```

**Diagram 3: Data vs Model Distribution**
```
True data distribution p_data(x):
  [scattered points representing real images]
  
Model distribution p_θ(x):
  [a learned density we want to match p_data]
  
Training: minimize KL(p_data || p_θ)
         = maximize E_{p_data}[log p_θ(x)]
```

---

**Next:** In Part 2, we derive the ELBO, explain the reparameterization trick, and show exactly how neural networks become probability distributions.


