# Part 5: Practical Training Guide

## Table of Contents
- [5.1 Initialization and Architecture Choices](#51-initialization-and-architecture-choices)
- [5.2 Numerical Stability](#52-numerical-stability)
- [5.3 Common Failure Modes](#53-common-failure-modes)
- [5.4 Debugging Checklist](#54-debugging-checklist)
- [5.5 Evaluation Metrics](#55-evaluation-metrics)
- [5.6 Reproducibility and Experiment Logging](#56-reproducibility-and-experiment-logging)
- [5.7 Learning Path and Study Schedule](#57-learning-path-and-study-schedule)
- [5.8 Recap](#58-recap)

---

## 5.1 Initialization and Architecture Choices

### Weight Initialization

**Default PyTorch/TensorFlow initialization usually works.** However, for VAEs specifically:

```python
def init_weights(m):
    """
    Custom initialization for VAE.
    """
    if isinstance(m, nn.Linear):
        # Xavier uniform is good for tanh/sigmoid
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # He initialization for ReLU
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Apply
model.apply(init_weights)

# Special: initialize log_var bias to small negative value
# This starts with small variance, preventing initial collapse
model.fc_log_var.bias.data.fill_(-1.0)  # σ² ≈ 0.37
```

### Architecture Guidelines

| Hyperparameter | Typical Range | Notes |
|----------------|---------------|-------|
| **Latent dim** | 2-512 | Start with 20-64 for MNIST, 128-256 for complex images |
| **Hidden layers** | 2-4 | More for larger images |
| **Hidden units** | 256-1024 | Depends on data complexity |
| **Activation** | ReLU, LeakyReLU | ReLU common, LeakyReLU helps with dead neurons |
| **Output activation** | None (logits) | Apply sigmoid only for visualization |

### Encoder/Decoder Symmetry

**Common practice:** Make decoder mirror encoder architecture.

```python
# Encoder: 784 -> 400 -> 200 -> latent
# Decoder: latent -> 200 -> 400 -> 784
```

**But asymmetry can help:**
- Stronger decoder: Better reconstruction
- Stronger encoder: Better posterior approximation

### Convolutional VAE Guidelines

For images, use convolutional layers:

```python
# Encoder (downsampling)
Conv2d(3, 32, 4, stride=2, padding=1)   # 64 -> 32
Conv2d(32, 64, 4, stride=2, padding=1)  # 32 -> 16
Conv2d(64, 128, 4, stride=2, padding=1) # 16 -> 8
Flatten()
Linear(128*8*8, latent_dim*2)  # mu and log_var

# Decoder (upsampling)
Linear(latent_dim, 128*8*8)
Unflatten(-1, (128, 8, 8))
ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 8 -> 16
ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # 16 -> 32
ConvTranspose2d(32, 3, 4, stride=2, padding=1)    # 32 -> 64
```

---

## 5.2 Numerical Stability

### The Log-Variance Parameterization

**Never** parameterize variance directly. Use log-variance:

```python
# BAD: variance can be negative or zero
var = self.fc_var(h)  # Could output negative!
std = torch.sqrt(var)  # NaN!

# GOOD: log-variance is unconstrained
log_var = self.fc_log_var(h)  # Can be any real number
std = torch.exp(0.5 * log_var)  # Always positive
```

### Clamping Log-Variance

Extreme log-variance values cause issues:

```python
def safe_log_var(log_var, min_val=-20, max_val=20):
    """
    Clamp log variance to prevent numerical issues.
    
    - Very negative: σ² → 0, can cause division by zero
    - Very positive: σ² → ∞, exploding gradients
    """
    return torch.clamp(log_var, min=min_val, max=max_val)
```

### Stable Log-Sum-Exp

For IWAE and mixture distributions:

```python
# BAD: numerical overflow
log_weights = ...  # Large values
weights_sum = torch.sum(torch.exp(log_weights))
result = torch.log(weights_sum)

# GOOD: logsumexp is stable
result = torch.logsumexp(log_weights, dim=-1)
```

### Epsilon in Logs

When computing log of values that might be zero:

```python
# BAD
log_prob = x * torch.log(p) + (1-x) * torch.log(1-p)  # log(0) = -inf

# GOOD: use binary_cross_entropy_with_logits
log_prob = -F.binary_cross_entropy_with_logits(logits, x, reduction='none')

# Or add small epsilon
eps = 1e-8
log_prob = x * torch.log(p + eps) + (1-x) * torch.log(1-p + eps)
```

---

## 5.3 Common Failure Modes

### 1. Posterior Collapse

**Symptoms:**
- KL divergence → 0 across training
- Encoder outputs nearly constant μ ≈ 0, σ ≈ 1 for all inputs
- Reconstructions identical regardless of input
- Decoder ignores z, reconstructs from nothing

**Causes:**
- Decoder too powerful (autoregressive)
- Latent dimension too large
- KL weight too high early in training
- Learning rate too high

**Diagnosis:**
```python
def check_posterior_collapse(model, data_loader, device, threshold=0.1):
    """
    Check if posterior is collapsing.
    """
    model.eval()
    kl_per_dim = []
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            mu, log_var = model.encode(x)
            
            # KL per dimension: 0.5 * (mu^2 + exp(log_var) - log_var - 1)
            kl_dim = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1)
            kl_per_dim.append(kl_dim.mean(0))  # Average over batch
    
    kl_per_dim = torch.stack(kl_per_dim).mean(0)  # Average over batches
    
    # Count "active" dimensions (KL > threshold)
    active_dims = (kl_per_dim > threshold).sum().item()
    
    print(f"KL per dimension: {kl_per_dim}")
    print(f"Active dimensions: {active_dims} / {len(kl_per_dim)}")
    
    return active_dims, kl_per_dim
```

**Solutions:**
1. **KL annealing:** Start with β=0, gradually increase
   ```python
   beta = min(1.0, epoch / warmup_epochs)
   loss = recon_loss + beta * kl_loss
   ```

2. **Free bits:** Ensure minimum KL per dimension
   ```python
   free_bits = 0.5
   kl_per_dim = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1)
   kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
   kl_loss = kl_per_dim.sum(dim=1).mean()
   ```

3. **Weaker decoder:** Reduce capacity
4. **Smaller latent dimension**
5. **Lower learning rate**

### 2. Blurry Reconstructions

**Symptoms:**
- Reconstructions look smoothed/averaged
- Fine details missing
- Samples lack sharpness

**Causes:**
- Gaussian decoder with fixed variance averages over modes
- Factorized likelihood ignores pixel correlations
- Underfitting reconstruction term

**Solutions:**
1. **Use MSE with lower weight or learned variance**
2. **β < 1 (sacrifice some regularization)**
3. **Larger/deeper decoder**
4. **Autoregressive decoder** (but watch for posterior collapse)
5. **Perceptual loss** (add VGG feature matching)

### 3. Mode Collapse

**Symptoms:**
- Generated samples all look similar
- Low diversity
- Latent space has "holes"

**Causes:**
- KL too low (poor coverage of prior)
- Dataset imbalance
- Poor encoder

**Solutions:**
1. **Increase KL weight (β > 1)**
2. **Use VampPrior**
3. **Better data augmentation**

### 4. Unstable Training

**Symptoms:**
- Loss oscillates wildly
- NaN losses
- Gradient explosions

**Causes:**
- Learning rate too high
- Numerical instability in log/exp operations
- Extreme variance values

**Solutions:**
```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Clamp log_var
log_var = torch.clamp(log_var, min=-10, max=10)

# Use batch normalization
self.encoder = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.BatchNorm1d(hidden_dim),
    nn.ReLU(),
    ...
)
```

---

## 5.4 Debugging Checklist

### Pre-Training Checks

```python
def pre_training_checks(model, train_loader, device):
    """
    Run before training to catch common issues.
    """
    model.to(device)
    x, _ = next(iter(train_loader))
    x = x.to(device)
    
    # 1. Forward pass works
    try:
        x_recon, mu, log_var = model(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {x_recon.shape}")
        print(f"  Latent shape: {mu.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # 2. Output ranges
    print(f"\n  x_recon range: [{x_recon.min():.3f}, {x_recon.max():.3f}]")
    print(f"  mu range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"  log_var range: [{log_var.min():.3f}, {log_var.max():.3f}]")
    
    # 3. Loss computes
    try:
        loss, recon, kl = vae_loss(x_recon, x, mu, log_var)
        print(f"\n✓ Loss computation successful")
        print(f"  Total: {loss.item():.3f}, Recon: {recon.item():.3f}, KL: {kl.item():.3f}")
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False
    
    # 4. Backward pass works
    try:
        loss.backward()
        print(f"✓ Backward pass successful")
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return False
    
    # 5. Check for NaN gradients
    has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"✗ NaN gradient in {name}")
            has_nan = True
    if not has_nan:
        print(f"✓ No NaN gradients")
    
    # 6. Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {n_params:,}")
    
    return True
```

### During Training Monitoring

```python
def training_monitor(epoch, batch_idx, loss, recon, kl, mu, log_var, 
                     prev_loss=None, log_every=100):
    """
    Monitor training progress and detect issues.
    """
    if batch_idx % log_every != 0:
        return
    
    alerts = []
    
    # Check for NaN
    if torch.isnan(loss):
        alerts.append("⚠️ LOSS IS NaN!")
    
    # Check for KL collapse
    avg_kl_per_dim = kl / mu.size(1)
    if avg_kl_per_dim < 0.01:
        alerts.append(f"⚠️ KL very low ({avg_kl_per_dim:.4f}) - possible collapse")
    
    # Check for exploding variance
    max_var = log_var.exp().max().item()
    if max_var > 100:
        alerts.append(f"⚠️ Large variance ({max_var:.1f}) - instability")
    
    # Check for loss spike
    if prev_loss is not None and loss.item() > 2 * prev_loss:
        alerts.append(f"⚠️ Loss spike: {prev_loss:.1f} -> {loss.item():.1f}")
    
    # Print status
    print(f"Epoch {epoch} Batch {batch_idx} | Loss: {loss.item():.2f} | "
          f"Recon: {recon.item():.2f} | KL: {kl.item():.2f}")
    
    for alert in alerts:
        print(alert)
    
    return loss.item()
```

### Post-Training Diagnostics

```python
def post_training_diagnostics(model, test_loader, device):
    """
    Run after training to assess model quality.
    """
    model.eval()
    
    all_mu = []
    all_log_var = []
    all_recon_errors = []
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_recon, mu, log_var = model(x)
            
            all_mu.append(mu.cpu())
            all_log_var.append(log_var.cpu())
            
            recon_error = (x - torch.sigmoid(x_recon)).pow(2).sum(dim=1)
            all_recon_errors.append(recon_error.cpu())
    
    all_mu = torch.cat(all_mu, dim=0)
    all_log_var = torch.cat(all_log_var, dim=0)
    all_recon_errors = torch.cat(all_recon_errors, dim=0)
    
    print("=" * 50)
    print("POST-TRAINING DIAGNOSTICS")
    print("=" * 50)
    
    # Latent statistics
    print("\n1. LATENT SPACE STATISTICS")
    print(f"   μ mean: {all_mu.mean():.4f} (should be ~0)")
    print(f"   μ std: {all_mu.std():.4f}")
    print(f"   σ² mean: {all_log_var.exp().mean():.4f} (should be ~1 if good coverage)")
    
    # Per-dimension activity
    kl_per_dim = 0.5 * (all_mu.pow(2) + all_log_var.exp() - all_log_var - 1).mean(0)
    active = (kl_per_dim > 0.1).sum().item()
    print(f"\n2. LATENT DIMENSION ACTIVITY")
    print(f"   Active dimensions: {active} / {all_mu.size(1)}")
    print(f"   KL range: [{kl_per_dim.min():.4f}, {kl_per_dim.max():.4f}]")
    
    # Reconstruction quality
    print(f"\n3. RECONSTRUCTION QUALITY")
    print(f"   Mean squared error: {all_recon_errors.mean():.4f}")
    print(f"   Std: {all_recon_errors.std():.4f}")
    
    return {
        'mu_mean': all_mu.mean().item(),
        'mu_std': all_mu.std().item(),
        'var_mean': all_log_var.exp().mean().item(),
        'active_dims': active,
        'mse': all_recon_errors.mean().item(),
    }
```

---

## 5.5 Evaluation Metrics

### ELBO (Evidence Lower Bound)

The training objective. Higher is better.

```python
def compute_elbo(model, data_loader, device, n_samples=1):
    """
    Compute ELBO on a dataset.
    """
    model.eval()
    total_elbo = 0
    n_points = 0
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            batch_size = x.size(0)
            
            if n_samples == 1:
                x_recon, mu, log_var = model(x)
                recon = -F.binary_cross_entropy_with_logits(
                    x_recon, x, reduction='none'
                ).sum(dim=1)
                kl = 0.5 * torch.sum(
                    mu.pow(2) + log_var.exp() - log_var - 1, dim=1
                )
                elbo = recon - kl
            else:
                # Monte Carlo estimate with multiple samples
                elbo = torch.zeros(batch_size).to(device)
                for _ in range(n_samples):
                    x_recon, mu, log_var = model(x)
                    recon = -F.binary_cross_entropy_with_logits(
                        x_recon, x, reduction='none'
                    ).sum(dim=1)
                    kl = 0.5 * torch.sum(
                        mu.pow(2) + log_var.exp() - log_var - 1, dim=1
                    )
                    elbo += (recon - kl) / n_samples
            
            total_elbo += elbo.sum().item()
            n_points += batch_size
    
    return total_elbo / n_points
```

### IWAE Bound

A tighter bound using importance weighting:

```python
def compute_iwae_bound(model, data_loader, device, K=50):
    """
    Compute IWAE bound (K samples).
    """
    model.eval()
    total_iwae = 0
    n_points = 0
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            batch_size = x.size(0)
            
            mu, log_var = model.encode(x)
            
            log_weights = []
            for _ in range(K):
                z = model.reparameterize(mu, log_var)
                x_recon = model.decode(z)
                
                # log p(x|z)
                log_p_x_z = -F.binary_cross_entropy_with_logits(
                    x_recon, x, reduction='none'
                ).sum(dim=1)
                
                # log p(z)
                log_p_z = -0.5 * torch.sum(z.pow(2) + np.log(2*np.pi), dim=1)
                
                # log q(z|x)
                std = torch.exp(0.5 * log_var)
                log_q_z = -0.5 * torch.sum(
                    log_var + ((z - mu) / std).pow(2) + np.log(2*np.pi), dim=1
                )
                
                log_weights.append(log_p_x_z + log_p_z - log_q_z)
            
            log_weights = torch.stack(log_weights, dim=1)  # (batch, K)
            iwae = torch.logsumexp(log_weights, dim=1) - np.log(K)
            
            total_iwae += iwae.sum().item()
            n_points += batch_size
    
    return total_iwae / n_points
```

### FID Score (Fréchet Inception Distance)

Measures quality and diversity of generated samples. Lower is better.

```python
# FID requires pretrained Inception network
# Use pytorch-fid package for proper implementation
# pip install pytorch-fid

# Command line:
# python -m pytorch_fid real_images/ generated_images/

# Or programmatically (simplified):
def compute_fid(model, real_images, device, n_samples=10000):
    """
    Note: This is a simplified sketch. Use pytorch-fid for proper computation.
    """
    from pytorch_fid import fid_score
    
    # Generate samples
    model.eval()
    samples = []
    batch_size = 100
    
    with torch.no_grad():
        for _ in range(n_samples // batch_size):
            z = torch.randn(batch_size, model.latent_dim).to(device)
            x = model.decode(z)
            samples.append(torch.sigmoid(x))
    
    samples = torch.cat(samples, dim=0)
    
    # Save to directory and compute FID
    # ... (see pytorch-fid documentation)
    
    return fid_score
```

### Reconstruction Error

```python
def compute_reconstruction_error(model, data_loader, device, metric='mse'):
    """
    Compute reconstruction error on dataset.
    """
    model.eval()
    total_error = 0
    n_points = 0
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            x_recon, _, _ = model(x)
            x_recon = torch.sigmoid(x_recon)
            
            if metric == 'mse':
                error = F.mse_loss(x_recon, x, reduction='sum')
            elif metric == 'mae':
                error = F.l1_loss(x_recon, x, reduction='sum')
            elif metric == 'bce':
                error = F.binary_cross_entropy(x_recon, x, reduction='sum')
            
            total_error += error.item()
            n_points += x.size(0)
    
    return total_error / n_points
```

---

## 5.6 Reproducibility and Experiment Logging

### Setting Seeds

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set all random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Experiment Configuration

```python
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class ExperimentConfig:
    """
    Configuration for VAE experiment.
    """
    # Model
    input_dim: int = 784
    hidden_dim: int = 400
    latent_dim: int = 20
    
    # Training
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    
    # VAE specific
    beta: float = 1.0
    kl_warmup_epochs: int = 0
    
    # Reproducibility
    seed: int = 42
    
    # Meta
    experiment_name: str = "vae_mnist"
    notes: str = ""
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls(**json.load(f))
```

### Experiment Logger

```python
import os
from datetime import datetime

class ExperimentLogger:
    """
    Simple experiment logging.
    """
    
    def __init__(self, config, log_dir='experiments'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = os.path.join(log_dir, f"{config.experiment_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.config = config
        self.metrics = []
        
        # Save config
        config.save(os.path.join(self.exp_dir, 'config.json'))
        
        # Log file
        self.log_file = open(os.path.join(self.exp_dir, 'training.log'), 'w')
        self._log(f"Experiment: {config.experiment_name}")
        self._log(f"Started: {timestamp}")
        self._log(f"Config: {config.to_dict()}")
    
    def _log(self, message):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()
    
    def log_epoch(self, epoch, train_loss, test_loss, train_recon, train_kl, 
                  test_recon, test_kl, lr=None):
        """Log metrics for one epoch."""
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_recon': train_recon,
            'train_kl': train_kl,
            'test_recon': test_recon,
            'test_kl': test_kl,
            'lr': lr,
        }
        self.metrics.append(metrics)
        
        self._log(f"Epoch {epoch:3d} | Train: {train_loss:.2f} | Test: {test_loss:.2f} | "
                  f"Recon: {train_recon:.2f} | KL: {train_kl:.2f}")
    
    def save_model(self, model, optimizer, epoch, filename='checkpoint.pt'):
        """Save model checkpoint."""
        path = os.path.join(self.exp_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config.to_dict(),
            'metrics': self.metrics,
        }, path)
        self._log(f"Saved checkpoint to {path}")
    
    def save_metrics(self):
        """Save metrics to JSON."""
        path = os.path.join(self.exp_dir, 'metrics.json')
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def close(self):
        """Close logger."""
        self.save_metrics()
        self.log_file.close()
```

### Complete Training Script with Logging

```python
def train_with_logging(config):
    """
    Complete training pipeline with logging.
    """
    # Set seed
    set_seed(config.seed)
    
    # Initialize logger
    logger = ExperimentLogger(config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger._log(f"Using device: {device}")
    
    # Data
    train_loader, test_loader = get_mnist_loaders(config.batch_size)
    
    # Model
    model = VAE(config.input_dim, config.hidden_dim, config.latent_dim).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training loop
    for epoch in range(1, config.epochs + 1):
        # Get beta for this epoch (annealing)
        if config.kl_warmup_epochs > 0:
            beta = min(1.0, epoch / config.kl_warmup_epochs) * config.beta
        else:
            beta = config.beta
        
        # Train
        model.train()
        train_loss, train_recon, train_kl = 0, 0, 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            x_recon, mu, log_var = model(x)
            loss, recon, kl = beta_vae_loss(x_recon, x, mu, log_var, beta)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
        
        n_train = len(train_loader)
        train_loss /= n_train
        train_recon /= n_train
        train_kl /= n_train
        
        # Evaluate
        model.eval()
        test_loss, test_recon, test_kl = 0, 0, 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_recon, mu, log_var = model(x)
                loss, recon, kl = beta_vae_loss(x_recon, x, mu, log_var, beta)
                test_loss += loss.item()
                test_recon += recon.item()
                test_kl += kl.item()
        
        n_test = len(test_loader)
        test_loss /= n_test
        test_recon /= n_test
        test_kl /= n_test
        
        # Log
        logger.log_epoch(epoch, train_loss, test_loss, 
                        train_recon, train_kl, test_recon, test_kl)
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            logger.save_model(model, optimizer, epoch, f'checkpoint_epoch{epoch}.pt')
    
    # Final save
    logger.save_model(model, optimizer, config.epochs, 'final_model.pt')
    logger.close()
    
    return model
```

---

## 5.7 Learning Path and Study Schedule

### 12-Week Mastery Schedule

#### Weeks 1-2: Foundations
- **Reading:** Kingma & Welling (2013), Deep Learning Book Ch. 20
- **Topics:** Probability review, MLE, latent variables, intractability
- **Coding:** Implement log-likelihood for simple distributions
- **Exercise:** Derive MLE for Gaussian, compute KL divergence examples

#### Weeks 3-4: Core VAE Theory
- **Reading:** Tutorial papers (Doersch 2016, Kingma 2019 thesis)
- **Topics:** ELBO derivation (all three ways), interpretation of terms
- **Coding:** Implement ELBO computation from scratch
- **Exercise:** Derive ELBO, prove KL ≥ 0, show ELBO is lower bound

#### Weeks 5-6: Implementation
- **Reading:** PyTorch/TensorFlow documentation
- **Topics:** Reparameterization, encoder/decoder architecture
- **Coding:** Full VAE implementation, training loop
- **Exercise:** Train on MNIST, visualize latent space, debug common issues

#### Weeks 7-8: Advanced Training
- **Reading:** β-VAE paper, posterior collapse literature
- **Topics:** KL annealing, β-VAE, debugging, evaluation
- **Coding:** Implement β-VAE, add monitoring
- **Exercise:** Experiment with β values, analyze disentanglement

#### Weeks 9-10: Extensions
- **Reading:** CVAE, VQ-VAE, IWAE papers
- **Topics:** Conditional generation, discrete latents, tighter bounds
- **Coding:** Implement CVAE and VQ-VAE
- **Exercise:** Compare bounds, generate conditional samples

#### Weeks 11-12: Advanced Topics & Project
- **Reading:** Hierarchical VAEs, flows, recent papers
- **Topics:** State-of-the-art architectures, relation to diffusion
- **Coding:** Custom project (your choice of application)
- **Project:** Apply VAE to new dataset, write report

### Weekly Time Commitment

| Activity | Hours/Week |
|----------|------------|
| Reading papers/tutorials | 3-4 |
| Lectures/videos | 2-3 |
| Coding exercises | 4-5 |
| Experiments | 3-4 |
| Review/notes | 1-2 |
| **Total** | **13-18** |

---

## 5.8 Recap

### Critical Training Tips

1. **Always use log-variance parameterization**
2. **Start with KL annealing for complex data**
3. **Monitor KL per dimension for collapse**
4. **Use gradient clipping for stability**
5. **Save checkpoints frequently**
6. **Set random seeds for reproducibility**

### Key Metrics to Track

| Metric | What It Tells You |
|--------|------------------|
| Total loss | Overall training progress |
| Reconstruction loss | How well model explains data |
| KL divergence | How much latent is regularized |
| KL per dimension | Which dimensions are active |
| Validation loss | Generalization |

### Quick Debugging Reference

| Problem | First Thing to Try |
|---------|-------------------|
| NaN loss | Lower learning rate, clamp log_var |
| KL = 0 | Add KL annealing |
| Blurry reconstructions | Lower β, bigger decoder |
| Poor samples | Check latent coverage |
| Unstable training | Gradient clipping |

---

**End of Course Main Content**

See Appendices for:
- Complete derivations
- All code files
- Exercise solutions
- LaTeX source
- French translation


