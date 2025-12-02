# Self-Assessment Quiz: Variational Autoencoders

## Instructions
- Answer all 20 questions
- Write your answers before checking the solutions
- Each question tests a key concept from the course
- Mix of conceptual, mathematical, and practical questions

---

## Questions

### Conceptual Understanding

**Q1.** What is the main problem that VAEs solve compared to simpler latent variable models?

---

**Q2.** In the ELBO equation \(\mathcal{L} = \mathbb{E}_q[\log p(x|z)] - D_{KL}(q \| p)\), what happens if the KL term is zero? Why is this usually bad?

---

**Q3.** Why do we need the reparameterization trick? What would happen if we tried to backpropagate through a direct sampling operation?

---

**Q4.** Explain in one sentence why VAE samples are often blurrier than GAN samples.

---

**Q5.** What is "posterior collapse" and name two ways to prevent it.

---

### Mathematical Understanding

**Q6.** Write the closed-form KL divergence between \(q = \mathcal{N}(\mu, \sigma^2)\) and \(p = \mathcal{N}(0, 1)\) for a single dimension.

---

**Q7.** If \(\log p_\theta(x) = -100\) nats and the ELBO is \(-110\) nats, what is the KL divergence between the approximate and true posterior?

---

**Q8.** In the reparameterization \(z = \mu + \sigma \cdot \epsilon\), what distribution does \(\epsilon\) follow?

---

**Q9.** For a Gaussian decoder with fixed variance \(\sigma^2 = 1\), show that maximizing log-likelihood is equivalent to minimizing MSE (up to constants).

---

**Q10.** What is the relationship between the ELBO and the marginal likelihood? Write the equation that connects them.

---

### Implementation

**Q11.** Why do we parameterize the encoder output as \(\log \sigma^2\) instead of \(\sigma\)?

---

**Q12.** In PyTorch, what is wrong with this loss computation?
```python
recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
```
(Assume `x_recon` comes directly from the decoder's linear output layer)

---

**Q13.** How do you sample new images from a trained VAE at test time? Write the two key steps.

---

**Q14.** What's the typical shape of the encoder output tensors for a batch of 32 images with latent dimension 20?

---

**Q15.** In the loss function, should we sum or average over (a) pixels/dimensions and (b) the batch?

---

### Advanced Topics

**Q16.** What is β-VAE and how does β > 1 affect the model?

---

**Q17.** How does VQ-VAE differ from standard VAE in terms of the latent space?

---

**Q18.** In IWAE, why does using K > 1 samples give a tighter bound on \(\log p(x)\)?

---

**Q19.** Name one advantage and one disadvantage of using an autoregressive decoder in a VAE.

---

**Q20.** What does "amortized inference" mean in the context of VAEs?

---

---

# Solutions

### Q1. Solution
**The intractability of the posterior.** VAEs solve the problem that \(p_\theta(z|x)\) is intractable to compute directly. By introducing an approximate posterior \(q_\phi(z|x)\) and optimizing the ELBO, we can train the model without ever computing the true posterior.

---

### Q2. Solution
If KL = 0, then \(q_\phi(z|x) = p(z)\) for all \(x\). This means the encoder ignores the input and always outputs the prior (e.g., \(\mu = 0, \sigma^2 = 1\)). This is **posterior collapse** - the decoder must reconstruct without any information from the input, so the model fails to learn meaningful representations.

---

### Q3. Solution
We need the reparameterization trick because **gradients cannot flow through stochastic sampling**. If we sample \(z \sim \mathcal{N}(\mu, \sigma^2)\) directly, we can't compute \(\nabla_\phi\) because the sampling operation has no gradient. Reparameterization moves the randomness to a fixed distribution (\(\epsilon \sim \mathcal{N}(0,1)\)), making \(z = \mu + \sigma \cdot \epsilon\) a deterministic function of the parameters, enabling standard backpropagation.

---

### Q4. Solution
VAEs use a Gaussian decoder that averages over possible outputs, producing the mean of multiple modes rather than sharp samples from individual modes.

---

### Q5. Solution
**Posterior collapse** is when the encoder outputs the prior (\(\mu \approx 0, \sigma^2 \approx 1\)) for all inputs, making KL ≈ 0 and leaving the decoder to reconstruct without latent information.

**Two prevention methods:**
1. **KL annealing/warm-up:** Start with β=0, gradually increase to 1
2. **Free bits:** Enforce a minimum KL per dimension

(Other valid answers: weaker decoder, smaller latent dimension, different architecture)

---

### Q6. Solution
\[
D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2}\left(\mu^2 + \sigma^2 - \log \sigma^2 - 1\right)
\]

---

### Q7. Solution
From the equation: \(\log p(x) = \mathcal{L} + D_{KL}(q \| p_\theta(z|x))\)

Therefore: \(D_{KL} = \log p(x) - \mathcal{L} = -100 - (-110) = \mathbf{10}\) **nats**

---

### Q8. Solution
\(\epsilon \sim \mathcal{N}(0, I)\) - a **standard normal distribution** (zero mean, unit variance, independent dimensions).

---

### Q9. Solution
For Gaussian decoder with fixed \(\sigma^2 = 1\):
\[
\log p(x|z) = -\frac{D}{2}\log(2\pi) - \frac{1}{2}\sum_i (x_i - \mu_i)^2
\]

Maximizing this means minimizing \(\frac{1}{2}\sum_i (x_i - \mu_i)^2 = \frac{1}{2}\|x - \mu\|^2\), which is **half the MSE**. The constants don't affect optimization. ✓

---

### Q10. Solution
\[
\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + D_{KL}(q_\phi(z|x) \| p_\theta(z|x))
\]

Since KL ≥ 0, we have \(\log p_\theta(x) \geq \mathcal{L}\) (ELBO is a **lower bound** on marginal log-likelihood).

---

### Q11. Solution
1. **Numerical stability:** \(\sigma\) must be positive, but neural networks output unbounded values. With \(\log \sigma^2\), any real output is valid.
2. **Stable gradients:** Very small or large variances are handled better in log-space.
3. **Direct computation:** The KL formula uses \(\log \sigma^2\) directly.

---

### Q12. Solution
**Problem:** `binary_cross_entropy` expects inputs in [0,1], but decoder outputs logits (unbounded).

**Fix:** Use `binary_cross_entropy_with_logits`:
```python
recon = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum')
```
This applies sigmoid internally and is numerically stable.

---

### Q13. Solution
```python
# Step 1: Sample from PRIOR (not encoder!)
z = torch.randn(n_samples, latent_dim)

# Step 2: Decode
x_generated = torch.sigmoid(decoder(z))  # sigmoid for visualization
```

Key insight: At generation time, we use the **prior** p(z), not the encoder.

---

### Q14. Solution
- `mu`: shape `(32, 20)` - mean for each sample in batch
- `log_var`: shape `(32, 20)` - log variance for each sample

Both tensors have shape `(batch_size, latent_dim)`.

---

### Q15. Solution
- **(a) Pixels/dimensions:** **Sum** (each pixel contributes to the total likelihood)
- **(b) Batch:** **Mean** (so loss scale doesn't depend on batch size)

Correct: `recon.sum(dim=-1).mean()` or equivalently divide the total sum by batch_size.

---

### Q16. Solution
**β-VAE** modifies the objective: \(\mathcal{L}_\beta = \mathbb{E}_q[\log p(x|z)] - \beta \cdot D_{KL}\)

With **β > 1:**
- Stronger regularization on the latent space
- Encourages **disentanglement** (each dimension captures one factor)
- May sacrifice reconstruction quality
- Pushes encoder outputs closer to prior

---

### Q17. Solution
**VQ-VAE uses discrete latent codes** from a learned codebook, while standard VAE uses continuous Gaussian latents.

Key differences:
- No KL divergence term (different objective)
- Latent space: Discrete indices vs. continuous vectors
- Training: Straight-through estimator for gradients
- Avoids posterior collapse by design

---

### Q18. Solution
IWAE uses the bound: \(\mathcal{L}_K = \mathbb{E}[\log \frac{1}{K}\sum_k w_k]\) where \(w_k\) are importance weights.

By Jensen's inequality and the law of large numbers, as K increases:
\[\mathcal{L}_1 \leq \mathcal{L}_2 \leq ... \leq \mathcal{L}_K \leq \log p(x)\]

More samples = tighter bound because we get a better Monte Carlo estimate of the marginal likelihood.

---

### Q19. Solution
**Advantage:** Models pixel dependencies → **sharper, more coherent samples**

**Disadvantage:** Risk of **posterior collapse** - the powerful decoder can ignore z and reconstruct purely from autoregressive context.

(Other valid advantages: better likelihood scores. Other disadvantages: slower sampling due to sequential generation)

---

### Q20. Solution
**Amortized inference** means using a neural network (the encoder) to directly predict posterior parameters \(q_\phi(z|x)\) for any input \(x\) in a single forward pass.

**Contrast with:** Non-amortized variational inference optimizes \(q(z)\) separately for each data point, which is much slower.

**Benefit:** After training, inference is instant - just run the encoder network.

---

## Scoring Guide

| Score | Level |
|-------|-------|
| 18-20 | Expert - Ready to do research |
| 15-17 | Proficient - Solid understanding |
| 12-14 | Competent - Good foundation, review weak areas |
| 9-11 | Developing - Re-read core sections |
| <9 | Beginning - Start from Part 1 again |

---

## Key Concepts to Review If You Missed Questions

| Questions | Topic to Review |
|-----------|----------------|
| 1, 5, 10 | ELBO derivation, bound interpretation |
| 2, 5, 16 | Posterior collapse, β-VAE |
| 3, 8, 9 | Reparameterization, decoder distributions |
| 6, 7, 9 | Mathematical derivations |
| 11-15 | Implementation details |
| 17-20 | Advanced topics and extensions |


