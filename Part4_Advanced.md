# Part 4: Advanced Topics and Extensions

## Table of Contents
- [4.1 β-VAE and Disentanglement](#41-β-vae-and-disentanglement)
- [4.2 Conditional VAE (CVAE)](#42-conditional-vae-cvae)
- [4.3 VQ-VAE (Vector Quantized)](#43-vq-vae-vector-quantized)
- [4.4 Hierarchical VAEs](#44-hierarchical-vaes)
- [4.5 Importance Weighted Autoencoders (IWAE)](#45-importance-weighted-autoencoders-iwae)
- [4.6 VAEs with Autoregressive Decoders](#46-vaes-with-autoregressive-decoders)
- [4.7 Flow-Based Posteriors](#47-flow-based-posteriors)
- [4.8 VampPrior](#48-vampprior)
- [4.9 Relation to Other Generative Models](#49-relation-to-other-generative-models)
- [4.10 Recap](#410-recap)

---

## 4.1 β-VAE and Disentanglement

### Motivation

The standard VAE balances reconstruction and KL equally. But what if we want:
- **More disentanglement:** Each latent dimension captures one independent factor
- **Better reconstruction:** Sacrifice some regularity for sharper outputs

### β-VAE Objective

Higgins et al. (2017) proposed scaling the KL term:
\[
\mathcal{L}_{\beta} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{\text{KL}}(q_\phi(z|x) \| p(z))
\]

- **β > 1:** Stronger regularization, encourages disentanglement
- **β < 1:** Weaker regularization, better reconstruction
- **β = 1:** Standard VAE

### Why Does β > 1 Help Disentanglement?

Intuition: A stronger KL penalty pushes the encoder to:
1. Use fewer latent dimensions effectively (others collapse to prior)
2. Make dimensions more independent (matching the factorial prior)
3. Align latent axes with data-generating factors

### Implementation

```python
def beta_vae_loss(x_recon, x, mu, log_var, beta=4.0):
    """
    β-VAE loss with adjustable KL weight.
    """
    batch_size = x.size(0)
    
    # Reconstruction
    recon_loss = F.binary_cross_entropy_with_logits(
        x_recon, x, reduction='sum'
    ) / batch_size
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    
    # β-weighted total
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss
```

### Disentanglement Metrics

**1. β-VAE Metric (Higgins et al.):** Train a linear classifier to predict which factor varies in pairs of images using latent difference vectors.

**2. FactorVAE Metric:** Similar but uses majority voting over dimensions.

**3. Mutual Information Gap (MIG):** Measures gap between top two latent dimensions' mutual information with each factor.

**4. DCI Disentanglement:** Measures how well each latent captures exactly one factor.

*Note: Verify current literature for latest metrics and benchmarks.*

### KL Annealing / Warm-up

A common training trick: gradually increase β from 0 to target value:
\[
\beta_t = \min\left(1, \frac{t}{T_{\text{warmup}}}\right) \cdot \beta_{\text{target}}
\]

```python
def get_beta(epoch, warmup_epochs=10, target_beta=4.0):
    """Linear KL annealing schedule."""
    return min(1.0, epoch / warmup_epochs) * target_beta
```

**Why annealing helps:**
- Early training: Focus on reconstruction, learn useful features
- Later training: Add regularization, shape latent space
- Prevents posterior collapse (latent dimensions becoming unused)

---

## 4.2 Conditional VAE (CVAE)

### Motivation

Standard VAE: Generate random samples from the data distribution.
CVAE: Generate samples **conditioned on some attribute** (class label, text, etc.)

### Architecture

Condition both encoder and decoder on auxiliary information \(c\):
- **Encoder:** \(q_\phi(z|x, c)\)
- **Decoder:** \(p_\theta(x|z, c)\)
- **Prior:** Can be \(p(z)\) or learned \(p_\theta(z|c)\)

### ELBO for CVAE

\[
\mathcal{L} = \mathbb{E}_{q_\phi(z|x,c)}[\log p_\theta(x|z,c)] - D_{\text{KL}}(q_\phi(z|x,c) \| p(z|c))
\]

With standard prior \(p(z|c) = p(z) = \mathcal{N}(0, I)\):
\[
\mathcal{L} = \mathbb{E}_{q_\phi(z|x,c)}[\log p_\theta(x|z,c)] - D_{\text{KL}}(q_\phi(z|x,c) \| \mathcal{N}(0, I))
\]

### Implementation

```python
class CVAE(nn.Module):
    """
    Conditional VAE that conditions on class labels.
    """
    
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super().__init__()
        
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Encoder: input + one-hot class -> latent distribution
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: latent + one-hot class -> reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x, c):
        """
        Args:
            x: Input, shape (batch, input_dim)
            c: Class labels, shape (batch,) - integers
        """
        # One-hot encode class
        c_onehot = F.one_hot(c, self.num_classes).float()  # (batch, num_classes)
        
        # Concatenate input and class
        xc = torch.cat([x, c_onehot], dim=1)  # (batch, input_dim + num_classes)
        
        h = self.encoder(xc)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z, c):
        """
        Args:
            z: Latent code, shape (batch, latent_dim)
            c: Class labels, shape (batch,)
        """
        c_onehot = F.one_hot(c, self.num_classes).float()
        zc = torch.cat([z, c_onehot], dim=1)  # (batch, latent_dim + num_classes)
        return self.decoder(zc)
    
    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, c)
        return x_recon, mu, log_var
    
    def sample(self, c, num_samples=1):
        """Generate samples conditioned on class c."""
        device = next(self.parameters()).device
        c = c.expand(num_samples) if c.dim() == 0 else c
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return torch.sigmoid(self.decode(z, c))
```

### Use Cases

- **Image generation conditioned on class:** Generate digit "7"
- **Image-to-image translation:** Style transfer with content preservation
- **Text-conditioned generation:** Generate images from captions
- **Attribute manipulation:** Change specific attributes while keeping others

---

## 4.3 VQ-VAE (Vector Quantized)

### Motivation

Standard VAEs use continuous latent spaces. VQ-VAE (van den Oord et al., 2017) uses **discrete** latent codes from a learned codebook.

**Advantages:**
- Avoids posterior collapse
- Enables autoregressive priors over discrete codes
- Better compression properties

### Architecture

```
x → Encoder → z_e → Quantize → z_q → Decoder → x̂
                       ↑
                  Codebook E
                 (K embeddings)
```

1. **Encoder** outputs continuous \(z_e \in \mathbb{R}^{D}\)
2. **Codebook** contains K embedding vectors \(e_k \in \mathbb{R}^{D}\)
3. **Quantization** finds nearest codebook vector:
   \[
   z_q = e_k \quad \text{where} \quad k = \arg\min_j \|z_e - e_j\|^2
   \]
4. **Decoder** reconstructs from \(z_q\)

### Loss Function

\[
\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{Reconstruction}} + \underbrace{\|\text{sg}[z_e] - e\|^2}_{\text{Codebook loss}} + \beta \underbrace{\|z_e - \text{sg}[e]\|^2}_{\text{Commitment loss}}
\]

where \(\text{sg}[\cdot]\) is the stop-gradient operator.

### Straight-Through Estimator

The quantization operation is non-differentiable. We use the **straight-through estimator**:
- **Forward:** Use quantized \(z_q\)
- **Backward:** Pass gradients to \(z_e\) as if no quantization occurred

### Implementation

```python
class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with codebook learning.
    """
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        
        self.num_embeddings = num_embeddings  # K
        self.embedding_dim = embedding_dim    # D
        self.commitment_cost = commitment_cost
        
        # Codebook: K x D
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z_e):
        """
        Args:
            z_e: Encoder output, shape (batch, D) or (batch, H, W, D)
        
        Returns:
            z_q: Quantized latent, same shape as z_e
            loss: VQ loss (codebook + commitment)
            encoding_indices: Which codebook entry was selected
        """
        # Flatten if spatial
        input_shape = z_e.shape
        flat_z_e = z_e.view(-1, self.embedding_dim)  # (N, D)
        
        # Compute distances to all codebook entries
        # ||z_e - e||^2 = ||z_e||^2 + ||e||^2 - 2*z_e·e
        distances = (
            torch.sum(flat_z_e ** 2, dim=1, keepdim=True)    # (N, 1)
            + torch.sum(self.embedding.weight ** 2, dim=1)    # (K,)
            - 2 * torch.matmul(flat_z_e, self.embedding.weight.t())  # (N, K)
        )
        
        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)  # (N,)
        
        # Quantize: look up codebook entries
        z_q = self.embedding(encoding_indices)  # (N, D)
        z_q = z_q.view(input_shape)  # Reshape back
        
        # Losses
        # Codebook loss: move codebook toward encoder outputs
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        # Commitment loss: move encoder outputs toward codebook
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: copy gradients from z_q to z_e
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, vq_loss, encoding_indices


class VQVAE(nn.Module):
    """
    Simple VQ-VAE for images.
    """
    
    def __init__(self, input_channels=1, hidden_dim=128, 
                 num_embeddings=512, embedding_dim=64):
        super().__init__()
        
        # Encoder: image -> continuous latent
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 4, 2, 1),  # 28->14
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),      # 14->7
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, 3, 1, 1),   # 7->7
        )
        
        # Vector quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        
        # Decoder: quantized latent -> image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim, 4, 2, 1),  # 7->14
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),     # 14->28
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_channels, 3, 1, 1),          # 28->28
        )
    
    def forward(self, x):
        # Encode
        z_e = self.encoder(x)                    # (batch, D, H, W)
        z_e = z_e.permute(0, 2, 3, 1)           # (batch, H, W, D)
        
        # Quantize
        z_q, vq_loss, _ = self.vq(z_e)
        
        # Decode
        z_q = z_q.permute(0, 3, 1, 2)           # (batch, D, H, W)
        x_recon = self.decoder(z_q)
        
        return x_recon, vq_loss
```

---

## 4.4 Hierarchical VAEs

### Motivation

Single-layer VAEs may not capture complex, multi-scale structure. **Hierarchical VAEs** use multiple layers of latent variables.

### Architecture

```
x → z₁ → z₂ → ... → zL
        ↑
    Each layer conditions on the one below
```

**Generative model:**
\[
p(x, z_{1:L}) = p(z_L) \prod_{\ell=1}^{L-1} p(z_\ell | z_{\ell+1}) \cdot p(x|z_1)
\]

**Inference model:**
\[
q(z_{1:L}|x) = q(z_1|x) \prod_{\ell=2}^{L} q(z_\ell | z_{\ell-1}, x)
\]

### ELBO for Hierarchical VAE

\[
\mathcal{L} = \mathbb{E}_q[\log p(x|z_1)] - \sum_{\ell=1}^{L} D_{\text{KL}}(q(z_\ell | \cdot) \| p(z_\ell | \cdot))
\]

### Notable Architectures

1. **Ladder VAE:** Uses deterministic skip connections
2. **NVAE:** (Vahdat & Kautz, 2020) State-of-the-art image generation
3. **Very Deep VAE:** Uses residual connections throughout

*Verify current literature for latest hierarchical VAE architectures.*

### Simple Two-Level Example

```python
class HierarchicalVAE(nn.Module):
    """
    Two-level hierarchical VAE.
    """
    
    def __init__(self, input_dim, hidden_dim, z1_dim, z2_dim):
        super().__init__()
        
        # Bottom-up encoder
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu1 = nn.Linear(hidden_dim, z1_dim)
        self.fc_logvar1 = nn.Linear(hidden_dim, z1_dim)
        
        self.encoder2 = nn.Sequential(
            nn.Linear(z1_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu2 = nn.Linear(hidden_dim, z2_dim)
        self.fc_logvar2 = nn.Linear(hidden_dim, z2_dim)
        
        # Top-down decoder
        self.decoder2 = nn.Sequential(
            nn.Linear(z2_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu1_prior = nn.Linear(hidden_dim, z1_dim)
        self.fc_logvar1_prior = nn.Linear(hidden_dim, z1_dim)
        
        self.decoder1 = nn.Sequential(
            nn.Linear(z1_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Bottom-up: infer z1 and z2 from x
        h1 = self.encoder1(x)
        mu1_q, logvar1_q = self.fc_mu1(h1), self.fc_logvar1(h1)
        z1 = self.reparameterize(mu1_q, logvar1_q)
        
        h2 = self.encoder2(z1)
        mu2_q, logvar2_q = self.fc_mu2(h2), self.fc_logvar2(h2)
        z2 = self.reparameterize(mu2_q, logvar2_q)
        
        # Top-down: generate z1 prior from z2, then x from z1
        h2_dec = self.decoder2(z2)
        mu1_p, logvar1_p = self.fc_mu1_prior(h2_dec), self.fc_logvar1_prior(h2_dec)
        
        x_recon = self.decoder1(z1)
        
        # Losses
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, reduction='sum'
        ) / batch_size
        
        # KL for z2: standard prior N(0,I)
        kl2 = -0.5 * torch.sum(
            1 + logvar2_q - mu2_q.pow(2) - logvar2_q.exp()
        ) / batch_size
        
        # KL for z1: prior from z2
        kl1 = self.gaussian_kl(mu1_q, logvar1_q, mu1_p, logvar1_p) / batch_size
        
        loss = recon_loss + kl1 + kl2
        
        return loss, recon_loss, kl1, kl2
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def gaussian_kl(self, mu1, logvar1, mu2, logvar2):
        """KL(N(mu1, var1) || N(mu2, var2))"""
        return 0.5 * torch.sum(
            logvar2 - logvar1 - 1 
            + (logvar1.exp() + (mu1 - mu2).pow(2)) / logvar2.exp()
        )
```

---

## 4.5 Importance Weighted Autoencoders (IWAE)

### Motivation

The standard ELBO uses a single sample from \(q(z|x)\). IWAE uses multiple samples to get a tighter bound.

### IWAE Objective

\[
\mathcal{L}_K = \mathbb{E}_{z^{(1)}, \ldots, z^{(K)} \sim q(z|x)}\left[\log \frac{1}{K}\sum_{k=1}^{K} \frac{p(x,z^{(k)})}{q(z^{(k)}|x)}\right]
\]

**Key property:**
\[
\mathcal{L}_1 = \text{ELBO} \leq \mathcal{L}_2 \leq \cdots \leq \mathcal{L}_K \leq \log p(x)
\]

More samples = tighter bound.

### Implementation

```python
def iwae_loss(model, x, K=5):
    """
    Importance Weighted Autoencoder loss.
    
    Args:
        model: VAE model
        x: Input batch, shape (batch_size, input_dim)
        K: Number of importance samples
    """
    batch_size = x.size(0)
    
    # Encode once to get distribution parameters
    mu, log_var = model.encode(x)  # (batch, latent_dim)
    
    # Repeat for K samples
    mu = mu.unsqueeze(1).repeat(1, K, 1)       # (batch, K, latent)
    log_var = log_var.unsqueeze(1).repeat(1, K, 1)  # (batch, K, latent)
    
    # Sample K times
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    z = mu + std * eps  # (batch, K, latent)
    
    # Flatten for decoding
    z_flat = z.view(batch_size * K, -1)  # (batch*K, latent)
    x_recon_flat = model.decode(z_flat)  # (batch*K, input_dim)
    x_recon = x_recon_flat.view(batch_size, K, -1)  # (batch, K, input_dim)
    
    # Expand x for comparison
    x_expanded = x.unsqueeze(1).repeat(1, K, 1)  # (batch, K, input_dim)
    
    # Log probabilities
    # log p(x|z)
    log_p_x_given_z = -F.binary_cross_entropy_with_logits(
        x_recon, x_expanded, reduction='none'
    ).sum(dim=-1)  # (batch, K)
    
    # log p(z) = log N(z; 0, I)
    log_p_z = -0.5 * torch.sum(z.pow(2) + np.log(2 * np.pi), dim=-1)  # (batch, K)
    
    # log q(z|x)
    log_q_z_given_x = -0.5 * torch.sum(
        log_var + ((z - mu) / std).pow(2) + np.log(2 * np.pi), dim=-1
    )  # (batch, K)
    
    # Log importance weights
    log_w = log_p_x_given_z + log_p_z - log_q_z_given_x  # (batch, K)
    
    # IWAE objective: log mean of importance weights
    # Use logsumexp for numerical stability: log(1/K * sum(exp(log_w)))
    iwae_elbo = torch.logsumexp(log_w, dim=1) - np.log(K)  # (batch,)
    
    # Negative ELBO as loss (we maximize ELBO = minimize -ELBO)
    loss = -iwae_elbo.mean()
    
    return loss
```

### Trade-offs

**Pros:**
- Tighter bound on log-likelihood
- Better density estimation

**Cons:**
- Computational cost: K forward passes per batch
- Gradients may have higher variance
- May not improve sample quality

---

## 4.6 VAEs with Autoregressive Decoders

### Motivation

Factorized decoders (independent pixels) produce blurry outputs. **Autoregressive decoders** model pixel dependencies:
\[
p(x|z) = \prod_{i=1}^{D} p(x_i | x_{<i}, z)
\]

### The KL Collapse Problem

With powerful autoregressive decoders, the model may ignore \(z\) entirely (posterior collapse). The decoder can reconstruct perfectly from just the autoregressive context.

### Solutions

1. **Weak decoder:** Limit the autoregressive receptive field
2. **Strong KL:** Use β > 1
3. **Free bits:** Ensure minimum KL per dimension
4. **Discrete latents:** VQ-VAE with autoregressive prior

### Example: VAE with PixelCNN Decoder

```python
class MaskedConv2d(nn.Conv2d):
    """
    Masked convolution for autoregressive modeling.
    """
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0
        
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class PixelCNNDecoder(nn.Module):
    """
    Simple PixelCNN decoder conditioned on z.
    """
    def __init__(self, latent_dim, hidden_dim=64, out_channels=1):
        super().__init__()
        
        # Project z to spatial feature map
        self.fc = nn.Linear(latent_dim, 7 * 7 * hidden_dim)
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
        )
        
        # Autoregressive layers
        self.ar_layers = nn.Sequential(
            MaskedConv2d('A', hidden_dim + out_channels, hidden_dim, 7, 1, 3),
            nn.ReLU(),
            MaskedConv2d('B', hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(),
            MaskedConv2d('B', hidden_dim, out_channels, 3, 1, 1),
        )
    
    def forward(self, z, x=None):
        """
        Args:
            z: Latent code (batch, latent_dim)
            x: Input image for autoregressive conditioning (batch, 1, 28, 28)
        """
        # Get spatial features from z
        h = self.fc(z).view(-1, 64, 7, 7)
        h = self.upsample(h)  # (batch, 64, 28, 28)
        
        # Concatenate with input for autoregressive
        if x is not None:
            h = torch.cat([h, x], dim=1)  # (batch, 65, 28, 28)
        
        return self.ar_layers(h)
```

---

## 4.7 Flow-Based Posteriors

### Motivation

Gaussian posteriors may be too simple. **Normalizing flows** transform a simple distribution into a complex one.

### Idea

Apply a sequence of invertible transformations:
\[
z_K = f_K \circ f_{K-1} \circ \cdots \circ f_1(z_0), \quad z_0 \sim \mathcal{N}(0, I)
\]

The density of \(z_K\) is:
\[
\log q(z_K|x) = \log q(z_0) - \sum_{k=1}^{K} \log \left|\det \frac{\partial f_k}{\partial z_{k-1}}\right|
\]

### Planar Flow Example

\[
f(z) = z + u \cdot h(w^T z + b)
\]

where \(u, w \in \mathbb{R}^D\), \(b \in \mathbb{R}\), and \(h\) is a nonlinearity.

```python
class PlanarFlow(nn.Module):
    """
    Single planar flow transformation.
    """
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim))
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(1))
    
    def forward(self, z):
        """
        Args:
            z: Input, shape (batch, dim)
        Returns:
            z_new: Transformed, shape (batch, dim)
            log_det: Log determinant of Jacobian, shape (batch,)
        """
        # f(z) = z + u * tanh(w^T z + b)
        wzb = torch.mv(z, self.w) + self.b  # (batch,)
        tanh_wzb = torch.tanh(wzb)          # (batch,)
        
        z_new = z + self.u.unsqueeze(0) * tanh_wzb.unsqueeze(1)  # (batch, dim)
        
        # Log determinant
        # det = 1 + u^T * psi, where psi = (1 - tanh^2(w^T z + b)) * w
        psi = (1 - tanh_wzb.pow(2)).unsqueeze(1) * self.w.unsqueeze(0)  # (batch, dim)
        log_det = torch.log(torch.abs(1 + torch.mv(psi, self.u)) + 1e-8)  # (batch,)
        
        return z_new, log_det


class FlowEncoder(nn.Module):
    """
    Encoder with normalizing flow for flexible posterior.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, n_flows=4):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        self.flows = nn.ModuleList([PlanarFlow(latent_dim) for _ in range(n_flows)])
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        
        # Sample z0 from base distribution
        std = torch.exp(0.5 * log_var)
        z0 = mu + std * torch.randn_like(std)
        
        # Apply flow transformations
        z = z0
        sum_log_det = 0
        for flow in self.flows:
            z, log_det = flow(z)
            sum_log_det += log_det
        
        # Log q(z_K|x) = log q(z_0|x) - sum(log det)
        log_q_z0 = -0.5 * torch.sum(log_var + ((z0 - mu) / std).pow(2) + np.log(2*np.pi), dim=1)
        log_q_zK = log_q_z0 - sum_log_det
        
        return z, mu, log_var, log_q_zK
```

---

## 4.8 VampPrior

### Motivation

The standard \(\mathcal{N}(0, I)\) prior may not match the true aggregate posterior. **VampPrior** (Tomczak & Welling, 2018) uses a mixture of variational posteriors.

### Definition

\[
p(z) = \frac{1}{K}\sum_{k=1}^{K} q_\phi(z | u_k)
\]

where \(\{u_k\}_{k=1}^K\) are learnable **pseudo-inputs**.

### ELBO with VampPrior

\[
\mathcal{L} = \mathbb{E}_q[\log p(x|z)] - D_{\text{KL}}\left(q(z|x) \| \frac{1}{K}\sum_k q(z|u_k)\right)
\]

The KL term no longer has a closed form—use Monte Carlo estimation.

### Implementation

```python
class VampPriorVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_pseudo_inputs=100):
        super().__init__()
        
        # Standard encoder/decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Pseudo-inputs: learnable parameters
        self.pseudo_inputs = nn.Parameter(torch.randn(n_pseudo_inputs, input_dim) * 0.01)
        
        self.n_pseudo_inputs = n_pseudo_inputs
        self.latent_dim = latent_dim
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)
    
    def log_vamp_prior(self, z):
        """
        Compute log p(z) under VampPrior.
        
        Args:
            z: Latent codes, shape (batch, latent_dim)
        Returns:
            log_p_z: Log prior probability, shape (batch,)
        """
        # Get posterior parameters for each pseudo-input
        mu_pseudo, log_var_pseudo = self.encode(torch.sigmoid(self.pseudo_inputs))
        # mu_pseudo: (K, latent_dim), log_var_pseudo: (K, latent_dim)
        
        # Compute log q(z|u_k) for each pseudo-input
        # Expand z: (batch, 1, latent) and pseudo: (1, K, latent)
        z_exp = z.unsqueeze(1)                      # (batch, 1, latent)
        mu_exp = mu_pseudo.unsqueeze(0)             # (1, K, latent)
        log_var_exp = log_var_pseudo.unsqueeze(0)  # (1, K, latent)
        
        # log N(z; mu, var) = -0.5 * (log(2pi) + log_var + (z-mu)^2/var)
        log_q = -0.5 * torch.sum(
            np.log(2 * np.pi) + log_var_exp + 
            (z_exp - mu_exp).pow(2) / log_var_exp.exp(),
            dim=-1
        )  # (batch, K)
        
        # log p(z) = log(1/K * sum_k q(z|u_k)) = logsumexp(log_q) - log(K)
        log_p_z = torch.logsumexp(log_q, dim=1) - np.log(self.n_pseudo_inputs)
        
        return log_p_z
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(log_var)
        x_recon = self.decoder(z)
        
        # Reconstruction loss
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, reduction='sum'
        ) / x.size(0)
        
        # KL approximation: E_q[log q(z|x) - log p(z)]
        log_q_z = -0.5 * torch.sum(
            np.log(2 * np.pi) + log_var + 1, dim=1
        )  # Uses (z-mu)/std = eps ~ N(0,1)
        log_p_z = self.log_vamp_prior(z)
        
        kl_loss = (log_q_z - log_p_z).mean()
        
        return recon_loss + kl_loss, recon_loss, kl_loss
```

---

## 4.9 Relation to Other Generative Models

### VAEs vs GANs

| Aspect | VAE | GAN |
|--------|-----|-----|
| **Training** | Stable (max likelihood) | Unstable (adversarial) |
| **Likelihood** | Tractable lower bound | No density |
| **Sample Quality** | Often blurry | Sharp but mode collapse risk |
| **Latent Space** | Smooth, good for interpolation | May have gaps |
| **Inference** | Encoder maps x → z | Requires optimization |

### VAEs vs Normalizing Flows

| Aspect | VAE | Flow |
|--------|-----|------|
| **Likelihood** | Lower bound | Exact |
| **Flexibility** | Gaussian encoder, any decoder | Invertible transformations |
| **Computation** | Cheap sampling | Expensive for some architectures |
| **Latent space** | Lower-dimensional | Same dimension as data |

### VAEs vs Diffusion Models

| Aspect | VAE | Diffusion |
|--------|-----|-----------|
| **Process** | Single encode/decode | Many denoising steps |
| **Sample Quality** | Moderate | State-of-the-art |
| **Speed** | Fast | Slow (many steps) |
| **Training** | ELBO | Denoising score matching |
| **Latent** | Learned | Fixed (noise schedule) |

*Note: Diffusion models (2020-2023) have largely surpassed VAEs for image generation quality. Verify current literature for latest comparisons.*

---

## 4.10 Recap

### Extensions Summary

| Extension | Key Idea | Use Case |
|-----------|----------|----------|
| **β-VAE** | Scale KL weight | Disentanglement |
| **CVAE** | Condition on labels | Controlled generation |
| **VQ-VAE** | Discrete latents | Compression, avoids collapse |
| **Hierarchical** | Multiple latent layers | Complex data |
| **IWAE** | Multiple samples | Tighter bounds |
| **AR decoder** | Model pixel dependencies | Sharper outputs |
| **Flow posterior** | Flexible q(z\|x) | Better inference |
| **VampPrior** | Mixture prior | Match aggregate posterior |

### When to Use What

- **Simple VAE:** Baseline, fast prototyping
- **β-VAE:** Need interpretable latent factors
- **CVAE:** Have labels, want control
- **VQ-VAE:** Images, video, need discrete tokens
- **Hierarchical:** Very complex data, need multi-scale
- **IWAE:** Need better density estimates

---

**Next:** Part 5 covers practical training details and debugging.


