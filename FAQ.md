# Frequently Asked Questions (FAQ)

## Fundamental Questions

### Q1: Is ELBO maximized or minimized? Why do we implement negative ELBO?

**Answer:** The ELBO is **maximized**. It's a lower bound on log p(x), and we want high likelihood.

However, in deep learning frameworks, we typically **minimize** a loss function. Therefore:
- We implement **negative ELBO** as the loss
- Minimizing (-ELBO) = Maximizing ELBO

```python
# We MAXIMIZE this:
ELBO = E[log p(x|z)] - KL(q||p)

# We MINIMIZE this (what we implement):
loss = -ELBO = -E[log p(x|z)] + KL(q||p)
     = reconstruction_loss + KL_loss
```

The reconstruction loss is negative log-likelihood (e.g., BCE), which is positive. The KL is always non-negative. So our loss is a positive number we minimize.

---

### Q2: Why use a standard normal N(0, I) as the prior?

**Answer:** Several reasons:

1. **Simplicity:** Easy to sample from, KL has closed form
2. **Regularization:** Prevents encoder from mapping to arbitrary regions
3. **Generation:** Known distribution to sample from at test time
4. **Independence:** Encourages latent dimensions to be independent

**Alternatives exist:**
- Learned prior (VampPrior)
- Mixture of Gaussians
- Autoregressive prior over discrete codes (VQ-VAE)

The standard normal is a **convention**, not a requirement. But it works well and keeps things simple.

---

### Q3: Does VAE sample noise at generation time?

**Answer:** **Yes, always.** To generate:

```python
# Sample from PRIOR (not encoder)
z = torch.randn(n_samples, latent_dim)  # This IS the noise

# Decode
x = decoder(z)
```

The randomness comes from sampling z ~ N(0, I). Each sample gives a different output.

**During training:** We sample from the **encoder** (approximate posterior), not the prior:
```python
z = mu + sigma * epsilon  # epsilon is the noise
```

---

### Q4: Why are VAE samples often blurry?

**Answer:** Multiple related reasons:

1. **Gaussian decoder averages modes:** If the true distribution is multimodal, the Gaussian mean falls in between, appearing blurry.

2. **Pixel independence assumption:** We model pixels independently:
   \[p(x|z) = \prod_i p(x_i|z)\]
   This ignores correlations between neighboring pixels.

3. **KL regularization:** The KL term prevents the encoder from being too precise, introducing uncertainty that manifests as blur.

4. **Mode coverage:** VAEs try to cover all modes (unlike GANs which may miss some), leading to averaging effects.

**Solutions:**
- Use β < 1 (less regularization)
- Autoregressive decoders
- Perceptual losses (feature matching)
- Better architectures (hierarchical VAEs)

---

### Q5: What is "real" when training a VAE?

**Answer:** Let's be precise:

| Term | What It Is | "Real"? |
|------|-----------|---------|
| \(x\) | Training data | Yes, observed from reality |
| \(p_{\text{data}}(x)\) | True data distribution | Exists but never observed directly |
| \(\hat{p}_{\text{data}}(x)\) | Empirical distribution | Approximation using finite samples |
| \(p_\theta(x)\) | Model distribution | Learned approximation |
| \(z\) | Latent variable | Invented/assumed, not observed |
| \(p(z)\) | Prior | Chosen by us (e.g., N(0,I)) |
| \(p_\theta(z\|x)\) | True posterior | Exists mathematically, intractable |
| \(q_\phi(z\|x)\) | Approximate posterior | Learned approximation |

**Philosophically:**
- Data \(x\) is real (observed)
- Latent \(z\) is a useful fiction we impose
- Distributions are mathematical objects we use to model reality
- The "true" data distribution is unknowable—we only see samples

---

## Technical Questions

### Q6: How do I evaluate likelihood for discrete image pixels?

**Answer:** This is tricky because:
- Images have discrete pixels (0-255)
- Continuous density p(x|z) can be > 1 (it's density, not probability)
- Comparing log-likelihoods across models requires care

**Options:**

1. **Dequantization:** Add uniform noise [0, 1/256) to each pixel before training. This converts discrete to continuous.

2. **Discretized logistic:** Model each pixel as:
   \[P(x=k) = \sigma((k+0.5-\mu)/s) - \sigma((k-0.5-\mu)/s)\]
   Integrate the density over the pixel's bin.

3. **Bernoulli for binarized data:** Threshold at 0.5, use binary cross-entropy.

4. **Bits per dimension:** Report in bits: 
   \[\text{bpd} = -\frac{\text{ELBO}}{d \cdot \ln 2}\]
   where d = number of pixels. Lower is better.

---

### Q7: What's the difference between p(x|z) and p_θ(x|z)?

**Answer:** 

- **\(p(x|z)\):** The "true" conditional distribution (if the generative model were correct)
- **\(p_\theta(x|z)\):** Our parametric model of this distribution

We always mean \(p_\theta(x|z)\) in practice—the decoder neural network with parameters \(\theta\).

Similarly:
- \(p(z|x)\): True posterior (intractable)
- \(p_\theta(z|x)\): True posterior under our model (still intractable)
- \(q_\phi(z|x)\): Our learned approximation (tractable)

---

### Q8: Can I use VAE for classification/discrimination?

**Answer:** Yes, but it's not the primary use case.

**Approaches:**

1. **Encoder features:** Train VAE, use encoder's μ as features for downstream classifier.

2. **Semi-supervised VAE:** Add classifier on latent space, train jointly.

3. **M2 model (Kingma et al.):** Combine generative and discriminative objectives.

**Generally:** Pure discriminative models (CNNs) are better for classification. VAEs shine for generation, representation learning, and density estimation.

---

### Q9: How does the number of samples L affect training?

**Answer:** In the reconstruction term:
\[\mathbb{E}_{q(z|x)}[\log p(x|z)] \approx \frac{1}{L}\sum_{\ell=1}^L \log p(x|z^{(\ell)})\]

**L=1 (most common):**
- One sample per data point per step
- High variance estimator but unbiased
- Computationally cheap
- Works surprisingly well due to mini-batching

**L>1:**
- Lower variance gradient estimates
- More computation per step
- Often not worth the cost
- IWAE uses L>1 for tighter bounds

**Why L=1 works:**
- Averaging over mini-batch already reduces variance
- Reparameterization enables low-variance gradients
- Many updates over training average out noise

---

### Q10: What happens if I don't use the reparameterization trick?

**Answer:** You'd need the **score function estimator** (REINFORCE):
\[\nabla_\phi \mathbb{E}_q[f(z)] = \mathbb{E}_q[f(z) \nabla_\phi \log q(z|x)]\]

**Problems:**
- Extremely high variance
- Requires many samples to get stable gradients
- Training is slow and unstable

**The reparameterization trick:**
- Moves randomness outside the parameters
- Enables direct backpropagation
- Much lower variance
- Makes VAE training practical

Without reparameterization, VAE training was essentially infeasible before 2013.

---

## Architectural Questions

### Q11: How do I choose the latent dimension?

**Answer:** Rules of thumb:

| Dataset | Typical Latent Dim |
|---------|-------------------|
| MNIST | 10-50 |
| CIFAR-10 | 64-256 |
| CelebA 64x64 | 128-512 |
| Complex images | 256-2048 |

**Too small:** Underfitting, poor reconstruction
**Too large:** Waste computation, possible collapse (some dims unused)

**How to find:**
1. Start with ~32-64 for simple data
2. Monitor active dimensions (KL per dim > 0.1)
3. If all dimensions active, try larger
4. If many inactive, could use smaller

---

### Q12: Convolutional vs fully-connected VAE?

**Answer:**

| Architecture | Use Case |
|-------------|----------|
| **Fully-connected** | Small images (MNIST), tabular data, any 1D data |
| **Convolutional** | Images, any data with spatial structure |

**For images larger than ~32x32, always use convolutional.**

Convolutional benefits:
- Parameter sharing (fewer parameters)
- Translation equivariance
- Captures local structure
- Scales to larger images

---

### Q13: Should encoder and decoder be symmetric?

**Answer:** Not necessarily.

**Symmetric (common):**
- Encoder: 784→400→200→latent
- Decoder: latent→200→400→784

**Asymmetric (can help):**
- Larger decoder: Better reconstruction
- Larger encoder: Better posterior approximation

**In practice:** Start symmetric, adjust based on what matters more for your task.

---

## Troubleshooting Questions

### Q14: My KL is always 0. What's wrong?

**Answer:** **Posterior collapse.** The encoder outputs μ≈0, σ²≈1 for all inputs, matching the prior.

**Check:**
```python
# Print encoder outputs
print(f"mu: mean={mu.mean():.3f}, std={mu.std():.3f}")
print(f"sigma²: mean={log_var.exp().mean():.3f}")
```

**Causes and fixes:**

| Cause | Fix |
|-------|-----|
| Decoder too strong | Reduce decoder capacity |
| KL dominates early | Use KL annealing |
| Learning rate too high | Lower to 1e-4 or 1e-5 |
| Latent too large | Reduce latent dimension |

**KL annealing code:**
```python
beta = min(1.0, epoch / warmup_epochs)
loss = recon_loss + beta * kl_loss
```

---

### Q15: My loss goes to NaN. Help!

**Answer:** Common causes:

1. **log(0):** Ensure probabilities are > 0
   ```python
   # Use BCE with logits
   loss = F.binary_cross_entropy_with_logits(logits, x)
   ```

2. **Exploding variance:**
   ```python
   log_var = torch.clamp(log_var, min=-10, max=10)
   ```

3. **Gradient explosion:**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

4. **Learning rate too high:**
   ```python
   optimizer = Adam(model.parameters(), lr=1e-4)  # Try lower
   ```

**Debug:** Print intermediate values to find where NaN first appears.

---

### Q16: Reconstructions look identical for different inputs. Why?

**Answer:** This is **posterior collapse**—the decoder ignores z.

**Verification:**
```python
# Encode different inputs
mu1, _ = model.encode(x1)
mu2, _ = model.encode(x2)
print(f"Latent distance: {(mu1 - mu2).norm()}")  # If ~0, collapsed
```

**If latents are similar for different inputs**, the encoder isn't learning meaningful representations.

**Fixes:** Same as Q14 (KL annealing, weaker decoder, etc.)

---

## Advanced Questions

### Q17: How does VAE relate to autoencoders?

**Answer:**

| Aspect | Autoencoder | VAE |
|--------|-------------|-----|
| Latent | Deterministic point | Distribution |
| Loss | Reconstruction only | ELBO (recon + KL) |
| Sampling | Not straightforward | Sample from prior |
| Latent space | May have gaps | Regularized, smooth |
| Theory | None specific | Variational inference |

**VAE = Probabilistic/Regularized Autoencoder**

The KL term is what makes VAE different—it regularizes the latent space for sampling.

---

### Q18: Can VAE latent space be interpreted?

**Answer:** Sometimes, especially with modifications:

1. **Standard VAE:** Latent dimensions often entangled (mixtures of factors)

2. **β-VAE:** Higher β encourages disentanglement—single dimension per factor

3. **Supervised approaches:** CVAE, factor VAE explicitly encourage interpretability

**How to check:**
- Latent traversal: Vary one dimension, observe changes
- Correlation with known factors (if labels available)
- Disentanglement metrics (MIG, DCI, etc.)

---

### Q19: How do diffusion models compare to VAEs?

**Answer:**

| Aspect | VAE | Diffusion |
|--------|-----|-----------|
| **Architecture** | Encoder + Decoder | Just decoder (UNet typically) |
| **Latent space** | Learned, lower-dim | Fixed (noise), same-dim as data |
| **Training** | Single forward pass | Predict noise at random timestep |
| **Sampling** | Single decode | Iterative denoising (many steps) |
| **Quality** | Good, often blurry | State-of-the-art |
| **Speed** | Fast | Slow (100-1000 steps) |
| **Likelihood** | Lower bound (ELBO) | Lower bound (similar math!) |

**Connection:** Diffusion models can be viewed as hierarchical VAEs with specific structure. The math involves similar ELBO-like bounds.

*Note: As of 2023-2024, diffusion models dominate image generation. Verify current literature for latest developments.*

---

### Q20: When should I NOT use a VAE?

**Answer:**

**Don't use VAE when:**
- You only need classification (use discriminative model)
- You need highest quality samples (consider diffusion/GAN)
- Data is simple enough for explicit density (use flow/autoregressive)
- You don't need generation or latent space (simpler autoencoder suffices)

**Do use VAE when:**
- You need fast sampling
- You want a meaningful latent space
- You need density estimation
- You want semi-supervised learning
- You need a simple, well-understood generative model
- You're doing representation learning

---

## Quick Reference

| Question | Short Answer |
|----------|--------------|
| Maximize or minimize ELBO? | Maximize (minimize -ELBO) |
| Why N(0,I) prior? | Simple, enables sampling |
| Sample noise at generation? | Yes, z ~ N(0,I) |
| Why blurry? | Gaussian averages modes |
| KL always 0? | Posterior collapse |
| Loss is NaN? | Numerical issues, clamp values |
| Good latent dim? | 20-64 for simple data |
| Conv vs FC? | Conv for images |
| VAE vs diffusion? | Diffusion better quality, VAE faster |


