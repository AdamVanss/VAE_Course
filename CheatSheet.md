# VAE Cheat Sheet (One-Page Reference)

## Core Equations

### ELBO (Evidence Lower Bound)
```
𝓛(θ,φ;x) = 𝔼_q[log p_θ(x|z)] - D_KL(q_φ(z|x) ‖ p(z))
          = Reconstruction    -  Regularization
```

### Fundamental Inequality
```
log p_θ(x) ≥ 𝓛(θ,φ;x)
log p_θ(x) = 𝓛(θ,φ;x) + D_KL(q_φ(z|x) ‖ p_θ(z|x))
```

### Reparameterization Trick
```
z = μ + σ ⊙ ε,  where ε ~ 𝒩(0, I)
```

### KL Divergence (Gaussian to Standard Normal)
```
D_KL(𝒩(μ,σ²) ‖ 𝒩(0,I)) = ½ Σⱼ(μⱼ² + σⱼ² - log σⱼ² - 1)
```

---

## PyTorch Code Snippets

### Reparameterization
```python
def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + std * eps
```

### KL Loss
```python
kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

### Reconstruction Loss (Bernoulli)
```python
recon = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum')
```

### Reconstruction Loss (Gaussian, fixed variance)
```python
recon = F.mse_loss(x_recon, x, reduction='sum') * 0.5
```

### Full Loss
```python
loss = (recon_loss + kl_loss) / batch_size
```

---

## Architecture Template

```
ENCODER: x → [FC → ReLU]×n → (μ, log σ²)
         ↓
REPARAM: z = μ + σ ⊙ ε
         ↓
DECODER: z → [FC → ReLU]×n → x̂ (logits)
```

---

## Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Latent dim | 10-512 | 20-64 for MNIST |
| Hidden units | 256-1024 | Match data complexity |
| Learning rate | 1e-4 to 1e-3 | Start with 1e-3 |
| Batch size | 64-256 | 128 is common |
| β (KL weight) | 0.1-10 | 1 for standard VAE |

---

## Debugging Quick Reference

| Problem | First Fix |
|---------|-----------|
| KL → 0 | KL annealing |
| NaN loss | Clamp log_var, lower LR |
| Blurry output | Lower β, bigger decoder |
| Same reconstructions | Check encoder output variance |

### KL Annealing
```python
beta = min(1.0, epoch / warmup_epochs)
loss = recon + beta * kl
```

---

## Key Distributions

| Symbol | Distribution | VAE Role |
|--------|-------------|----------|
| p(z) | 𝒩(0, I) | Prior |
| q_φ(z\|x) | 𝒩(μ_φ(x), σ²_φ(x)) | Encoder |
| p_θ(x\|z) | Bernoulli or 𝒩 | Decoder |

---

## Variants Quick Guide

| Variant | Key Change | Use Case |
|---------|-----------|----------|
| β-VAE | β > 1 on KL | Disentanglement |
| CVAE | Condition on labels | Controlled generation |
| VQ-VAE | Discrete latents | Compression |
| IWAE | K samples, tighter bound | Better density |

---

## Evaluation Metrics

| Metric | Compute | Better |
|--------|---------|--------|
| ELBO | -loss | Higher |
| Recon | BCE or MSE | Lower |
| KL | Closed-form | Balanced |
| FID | pytorch-fid | Lower |

---

## Training Checklist

- [ ] Use log_var (not σ directly)
- [ ] BCE with logits (not after sigmoid)
- [ ] Sum over dims, mean over batch
- [ ] Set seeds for reproducibility
- [ ] Monitor KL per dimension
- [ ] Save checkpoints
- [ ] Use gradient clipping if unstable

---

## Generation vs Training

**Training:** Sample z from encoder q_φ(z|x)
```python
z = reparameterize(mu, log_var)  # Uses input x
```

**Generation:** Sample z from prior p(z)
```python
z = torch.randn(n_samples, latent_dim)  # No input needed
```

---

## Common Equations in Code Form

```python
# ELBO components
E_q[log p(x|z)] ≈ -BCE(x, decoder(z))
D_KL = 0.5 * sum(mu² + exp(log_var) - log_var - 1)

# Loss (negative ELBO)
loss = BCE + KL

# Gaussian log-likelihood
log_p = -0.5 * (log(2π) + log(σ²) + (x-μ)²/σ²)
```

---

**Remember:** VAE = Encoder + Reparameterize + Decoder + (Recon + KL) Loss


