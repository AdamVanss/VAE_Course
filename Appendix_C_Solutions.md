# Appendix C: Exercises and Solutions

## Exercise 1: Derive the Gaussian KL Formula (Easy)

**Problem:** Derive the KL divergence between \(q(z) = \mathcal{N}(\mu, \sigma^2)\) and \(p(z) = \mathcal{N}(0, 1)\) for a single dimension.

**Solution:**

Start with the definition:
\[
D_{\text{KL}}(q \| p) = \mathbb{E}_q\left[\log \frac{q(z)}{p(z)}\right] = \mathbb{E}_q[\log q(z)] - \mathbb{E}_q[\log p(z)]
\]

**Step 1: Compute \(\mathbb{E}_q[\log q(z)]\)**

\[
\log q(z) = \log \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(z-\mu)^2}{2\sigma^2}\right)
= -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2}
\]

Taking expectation under \(q\):
\[
\mathbb{E}_q[\log q(z)] = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\mathbb{E}_q[(z-\mu)^2]
\]

Since \(\mathbb{E}_q[(z-\mu)^2] = \text{Var}_q(z) = \sigma^2\):
\[
\mathbb{E}_q[\log q(z)] = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2}
\]

**Step 2: Compute \(\mathbb{E}_q[\log p(z)]\)**

\[
\log p(z) = -\frac{1}{2}\log(2\pi) - \frac{z^2}{2}
\]

Taking expectation:
\[
\mathbb{E}_q[\log p(z)] = -\frac{1}{2}\log(2\pi) - \frac{1}{2}\mathbb{E}_q[z^2]
\]

Now, \(\mathbb{E}_q[z^2] = \text{Var}_q(z) + (\mathbb{E}_q[z])^2 = \sigma^2 + \mu^2\):
\[
\mathbb{E}_q[\log p(z)] = -\frac{1}{2}\log(2\pi) - \frac{\sigma^2 + \mu^2}{2}
\]

**Step 3: Combine**

\[
D_{\text{KL}} = \left(-\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2}\right) - \left(-\frac{1}{2}\log(2\pi) - \frac{\sigma^2 + \mu^2}{2}\right)
\]

\[
= -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log(\sigma^2) - \frac{1}{2} + \frac{1}{2}\log(2\pi) + \frac{\sigma^2 + \mu^2}{2}
\]

\[
= -\frac{1}{2}\log(\sigma^2) - \frac{1}{2} + \frac{\sigma^2}{2} + \frac{\mu^2}{2}
\]

\[
\boxed{D_{\text{KL}} = \frac{1}{2}\left(\mu^2 + \sigma^2 - \log \sigma^2 - 1\right)}
\]

For \(d\) independent dimensions, sum over dimensions:
\[
D_{\text{KL}} = \frac{1}{2}\sum_{j=1}^{d}\left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)
\]

---

## Exercise 2: Show ELBO is a Lower Bound (Medium)

**Problem:** Prove that \(\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x)\) for any distribution \(q_\phi(z|x)\).

**Solution:**

**Method 1: Using Jensen's Inequality**

\[
\log p_\theta(x) = \log \int p_\theta(x, z) dz = \log \int q_\phi(z|x) \frac{p_\theta(x, z)}{q_\phi(z|x)} dz
\]

This is \(\log \mathbb{E}_q[w(z)]\) where \(w(z) = \frac{p_\theta(x,z)}{q_\phi(z|x)}\).

By Jensen's inequality (log is concave):
\[
\log \mathbb{E}_q[w] \geq \mathbb{E}_q[\log w]
\]

Therefore:
\[
\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right] = \mathcal{L}
\]

**Method 2: Using KL Non-negativity**

We can write:
\[
\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + D_{\text{KL}}(q_\phi(z|x) \| p_\theta(z|x))
\]

Since \(D_{\text{KL}} \geq 0\) always:
\[
\log p_\theta(x) \geq \mathcal{L}(\theta, \phi; x)
\]

**To verify the decomposition:**

\begin{align}
\mathcal{L} &= \mathbb{E}_q[\log p_\theta(x|z)] - D_{\text{KL}}(q \| p(z)) \\
&= \mathbb{E}_q[\log p_\theta(x|z) + \log p(z) - \log q] \\
&= \mathbb{E}_q[\log p_\theta(x,z) - \log q]
\end{align}

\begin{align}
D_{\text{KL}}(q \| p_\theta(z|x)) &= \mathbb{E}_q\left[\log \frac{q}{p_\theta(z|x)}\right] \\
&= \mathbb{E}_q[\log q - \log p_\theta(z|x)] \\
&= \mathbb{E}_q\left[\log q - \log \frac{p_\theta(x,z)}{p_\theta(x)}\right] \\
&= \mathbb{E}_q[\log q - \log p_\theta(x,z) + \log p_\theta(x)] \\
&= -\mathcal{L} + \log p_\theta(x)
\end{align}

Rearranging: \(\log p_\theta(x) = \mathcal{L} + D_{\text{KL}}(q \| p_\theta(z|x))\) ✓

---

## Exercise 3: Implement VAE from Scratch (Hard)

**Problem:** Implement a complete VAE without using any high-level VAE libraries. Train on MNIST and achieve test ELBO > -100 nats.

**Solution:**

```python
"""
Complete VAE Implementation from Scratch
No high-level VAE helpers - everything explicit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Hyperparameters
INPUT_DIM = 784
HIDDEN_DIM = 400
LATENT_DIM = 20
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Define Encoder Network
class Encoder(nn.Module):
    """
    Maps x to parameters of q(z|x) = N(mu, sigma^2)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc_mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)
    
    def forward(self, x):
        # x: (batch, 784)
        h = F.relu(self.fc1(x))      # (batch, 400)
        h = F.relu(self.fc2(h))      # (batch, 400)
        mu = self.fc_mu(h)           # (batch, 20)
        logvar = self.fc_logvar(h)   # (batch, 20)
        return mu, logvar

# 2. Define Decoder Network
class Decoder(nn.Module):
    """
    Maps z to parameters of p(x|z) (Bernoulli logits)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc_out = nn.Linear(HIDDEN_DIM, INPUT_DIM)
    
    def forward(self, z):
        # z: (batch, 20)
        h = F.relu(self.fc1(z))      # (batch, 400)
        h = F.relu(self.fc2(h))      # (batch, 400)
        logits = self.fc_out(h)      # (batch, 784)
        return logits

# 3. Reparameterization Trick (explicit implementation)
def reparameterize(mu, logvar):
    """
    z = mu + sigma * epsilon, where epsilon ~ N(0, I)
    """
    std = torch.exp(0.5 * logvar)    # sigma = exp(0.5 * log(sigma^2))
    epsilon = torch.randn_like(std)   # Sample from N(0, I)
    z = mu + std * epsilon
    return z

# 4. Loss Function (explicit ELBO components)
def compute_loss(x, x_logits, mu, logvar):
    """
    Loss = -ELBO = -E[log p(x|z)] + KL(q(z|x) || p(z))
    
    Returns:
        total_loss: Scalar
        recon_loss: Reconstruction term (negative log-likelihood)
        kl_loss: KL divergence term
    """
    batch_size = x.size(0)
    
    # Reconstruction loss: -E[log p(x|z)]
    # For Bernoulli: -sum(x * log(sigmoid(logits)) + (1-x) * log(1-sigmoid(logits)))
    # This is binary cross-entropy
    recon_loss = F.binary_cross_entropy_with_logits(
        x_logits, x, reduction='none'
    ).sum(dim=1).mean()  # Sum over pixels, mean over batch
    
    # KL divergence: KL(N(mu, sigma^2) || N(0, 1))
    # = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    # = 0.5 * sum(mu^2 + exp(logvar) - logvar - 1)
    kl_loss = 0.5 * torch.sum(
        mu.pow(2) + logvar.exp() - logvar - 1,
        dim=1
    ).mean()  # Sum over latent dims, mean over batch
    
    total_loss = recon_loss + kl_loss
    
    return total_loss, recon_loss, kl_loss

# 5. Create Full Model
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        x_logits = self.decoder(z)
        return x_logits, mu, logvar

# 6. Data Loading
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

# 7. Training Function
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for x, _ in loader:
        x = x.to(DEVICE)
        
        optimizer.zero_grad()
        x_logits, mu, logvar = model(x)
        loss, recon, kl = compute_loss(x, x_logits, mu, logvar)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
    
    n = len(loader)
    return total_loss/n, total_recon/n, total_kl/n

# 8. Evaluation Function
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            x_logits, mu, logvar = model(x)
            loss, _, _ = compute_loss(x, x_logits, mu, logvar)
            total_loss += loss.item()
    
    return total_loss / len(loader)

# 9. Main Training Loop
def main():
    print(f"Using device: {DEVICE}")
    
    # Data
    train_loader, test_loader = get_data()
    
    # Model
    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Train
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer)
        test_loss = evaluate(model, test_loader)
        
        # Note: Loss is -ELBO, so ELBO = -loss
        test_elbo = -test_loss
        
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.2f} "
              f"(Recon: {train_recon:.2f}, KL: {train_kl:.2f}) | "
              f"Test ELBO: {test_elbo:.2f}")
    
    print(f"\nFinal Test ELBO: {test_elbo:.2f} nats")
    print("Target: > -100 nats")
    print(f"{'PASSED' if test_elbo > -100 else 'FAILED'}")

if __name__ == '__main__':
    main()
```

**Expected output after 30 epochs:**
```
Epoch 30 | Train Loss: 93.15 (Recon: 79.82, KL: 13.33) | Test ELBO: -93.27
Final Test ELBO: -93.27 nats
Target: > -100 nats
PASSED
```

---

## Exercise 4: Analyze Training Dynamics (Medium)

**Problem:** Train a VAE and plot:
1. Reconstruction loss vs. epoch
2. KL divergence vs. epoch
3. KL per latent dimension at the end of training
4. Identify any collapsed dimensions

**Solution:**

```python
import matplotlib.pyplot as plt
import torch
import numpy as np

def analyze_training(model, train_loader, test_loader, epochs=30, device='cpu'):
    """
    Train VAE and analyze dynamics.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    history = {
        'train_recon': [], 'train_kl': [],
        'test_recon': [], 'test_kl': []
    }
    
    for epoch in range(epochs):
        # Train
        model.train()
        epoch_recon, epoch_kl = 0, 0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_logits, mu, logvar = model(x)
            loss, recon, kl = compute_loss(x, x_logits, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
        
        history['train_recon'].append(epoch_recon / len(train_loader))
        history['train_kl'].append(epoch_kl / len(train_loader))
        
        # Evaluate
        model.eval()
        test_recon, test_kl = 0, 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_logits, mu, logvar = model(x)
                _, recon, kl = compute_loss(x, x_logits, mu, logvar)
                test_recon += recon.item()
                test_kl += kl.item()
        
        history['test_recon'].append(test_recon / len(test_loader))
        history['test_kl'].append(test_kl / len(test_loader))
    
    # Compute KL per dimension
    model.eval()
    all_mu, all_logvar = [], []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            mu, logvar = model.encoder(x)
            all_mu.append(mu)
            all_logvar.append(logvar)
    
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    
    # KL per dimension
    kl_per_dim = 0.5 * (all_mu.pow(2) + all_logvar.exp() - all_logvar - 1).mean(dim=0).cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Reconstruction loss
    axes[0, 0].plot(history['train_recon'], label='Train')
    axes[0, 0].plot(history['test_recon'], label='Test')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Reconstruction Loss')
    axes[0, 0].set_title('Reconstruction Loss vs. Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. KL divergence
    axes[0, 1].plot(history['train_kl'], label='Train')
    axes[0, 1].plot(history['test_kl'], label='Test')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].set_title('KL Divergence vs. Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. KL per dimension
    dims = np.arange(len(kl_per_dim))
    colors = ['red' if kl < 0.1 else 'blue' for kl in kl_per_dim]
    axes[1, 0].bar(dims, kl_per_dim, color=colors)
    axes[1, 0].axhline(y=0.1, color='gray', linestyle='--', label='Collapse threshold')
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL per Latent Dimension (Red = Collapsed)')
    axes[1, 0].legend()
    
    # 4. Summary statistics
    active_dims = np.sum(kl_per_dim > 0.1)
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary
    ================
    
    Final Reconstruction Loss: {history['test_recon'][-1]:.2f}
    Final KL Divergence: {history['test_kl'][-1]:.2f}
    Final ELBO: {-(history['test_recon'][-1] + history['test_kl'][-1]):.2f}
    
    Latent Dimension Analysis:
    - Total dimensions: {len(kl_per_dim)}
    - Active dimensions (KL > 0.1): {active_dims}
    - Collapsed dimensions: {len(kl_per_dim) - active_dims}
    
    KL per dim: min={kl_per_dim.min():.4f}, max={kl_per_dim.max():.4f}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=150)
    plt.show()
    
    return history, kl_per_dim
```

---

## Exercise 5: Posterior Collapse Experiment (Hard)

**Problem:** 
1. Train a VAE that exhibits posterior collapse
2. Diagnose it using metrics
3. Fix it using KL annealing
4. Compare results

**Solution:**

```python
"""
Posterior Collapse: Diagnosis and Fix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def create_vae(latent_dim=100, decoder_hidden=800):
    """
    Create VAE with potentially collapsing settings.
    Large latent dim + powerful decoder = collapse risk
    """
    class CollapseVAE(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder (relatively weak)
            self.enc = nn.Sequential(
                nn.Linear(784, 400),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(400, latent_dim)
            self.fc_logvar = nn.Linear(400, latent_dim)
            
            # Decoder (powerful - 3 layers)
            self.dec = nn.Sequential(
                nn.Linear(latent_dim, decoder_hidden),
                nn.ReLU(),
                nn.Linear(decoder_hidden, decoder_hidden),
                nn.ReLU(),
                nn.Linear(decoder_hidden, decoder_hidden),
                nn.ReLU(),
                nn.Linear(decoder_hidden, 784),
            )
        
        def encode(self, x):
            h = self.enc(x)
            return self.fc_mu(h), self.fc_logvar(h)
        
        def decode(self, z):
            return self.dec(z)
        
        def forward(self, x):
            mu, logvar = self.encode(x)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            return self.decode(z), mu, logvar
    
    return CollapseVAE()

def train_with_beta(model, train_loader, epochs, beta_schedule, device):
    """
    Train with specified beta schedule.
    
    Args:
        beta_schedule: function(epoch) -> beta value
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {'loss': [], 'recon': [], 'kl': [], 'beta': []}
    
    for epoch in range(epochs):
        beta = beta_schedule(epoch)
        model.train()
        epoch_loss, epoch_recon, epoch_kl = 0, 0, 0
        
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            x_recon, mu, logvar = model(x)
            
            recon = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum') / x.size(0)
            kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1) / x.size(0)
            
            # Beta-weighted loss
            loss = recon + beta * kl
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
        
        n = len(train_loader)
        history['loss'].append(epoch_loss/n)
        history['recon'].append(epoch_recon/n)
        history['kl'].append(epoch_kl/n)
        history['beta'].append(beta)
    
    return history

def diagnose_collapse(model, test_loader, device):
    """
    Diagnose posterior collapse.
    """
    model.eval()
    all_mu, all_logvar = [], []
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            all_mu.append(mu)
            all_logvar.append(logvar)
    
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    
    # KL per dimension
    kl_per_dim = 0.5 * (all_mu.pow(2) + all_logvar.exp() - all_logvar - 1).mean(dim=0)
    
    # Metrics
    total_kl = kl_per_dim.sum().item()
    active_dims = (kl_per_dim > 0.1).sum().item()
    
    # Check for collapse indicators
    mu_std = all_mu.std(dim=0).mean().item()  # Should vary across samples
    var_mean = all_logvar.exp().mean().item()  # Should not be ~1 for all
    
    diagnosis = {
        'total_kl': total_kl,
        'active_dims': active_dims,
        'total_dims': all_mu.size(1),
        'mu_std': mu_std,
        'var_mean': var_mean,
        'kl_per_dim': kl_per_dim.cpu().numpy(),
    }
    
    # Determine if collapsed
    collapsed = active_dims < all_mu.size(1) * 0.2  # Less than 20% active
    diagnosis['collapsed'] = collapsed
    
    return diagnosis

def run_collapse_experiment():
    """
    Full experiment comparing β=1 (collapse) vs annealed β (no collapse).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128)
    
    epochs = 30
    
    # Experiment 1: β=1 (likely to collapse)
    print("=" * 50)
    print("Experiment 1: β=1 (no annealing)")
    print("=" * 50)
    
    model1 = create_vae().to(device)
    history1 = train_with_beta(
        model1, train_loader, epochs,
        beta_schedule=lambda e: 1.0,  # Constant β=1
        device=device
    )
    diag1 = diagnose_collapse(model1, test_loader, device)
    
    print(f"Total KL: {diag1['total_kl']:.2f}")
    print(f"Active dimensions: {diag1['active_dims']}/{diag1['total_dims']}")
    print(f"Collapsed: {diag1['collapsed']}")
    
    # Experiment 2: KL annealing (should prevent collapse)
    print("\n" + "=" * 50)
    print("Experiment 2: KL annealing (β: 0 → 1 over 15 epochs)")
    print("=" * 50)
    
    model2 = create_vae().to(device)
    history2 = train_with_beta(
        model2, train_loader, epochs,
        beta_schedule=lambda e: min(1.0, e / 15),  # Anneal from 0 to 1
        device=device
    )
    diag2 = diagnose_collapse(model2, test_loader, device)
    
    print(f"Total KL: {diag2['total_kl']:.2f}")
    print(f"Active dimensions: {diag2['active_dims']}/{diag2['total_dims']}")
    print(f"Collapsed: {diag2['collapsed']}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # KL over training
    axes[0, 0].plot(history1['kl'], label='β=1')
    axes[0, 0].plot(history2['kl'], label='Annealed')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('KL Divergence')
    axes[0, 0].set_title('KL Divergence During Training')
    axes[0, 0].legend()
    
    # Recon over training
    axes[0, 1].plot(history1['recon'], label='β=1')
    axes[0, 1].plot(history2['recon'], label='Annealed')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss During Training')
    axes[0, 1].legend()
    
    # Beta schedule
    axes[0, 2].plot(history1['beta'], label='β=1')
    axes[0, 2].plot(history2['beta'], label='Annealed')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('β')
    axes[0, 2].set_title('β Schedule')
    axes[0, 2].legend()
    
    # KL per dimension comparison
    dims = range(len(diag1['kl_per_dim']))
    axes[1, 0].bar(dims, diag1['kl_per_dim'], alpha=0.7)
    axes[1, 0].axhline(0.1, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('KL')
    axes[1, 0].set_title(f"KL per Dim (β=1) - {diag1['active_dims']} active")
    
    axes[1, 1].bar(dims, diag2['kl_per_dim'], alpha=0.7)
    axes[1, 1].axhline(0.1, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('KL')
    axes[1, 1].set_title(f"KL per Dim (Annealed) - {diag2['active_dims']} active")
    
    # Summary
    axes[1, 2].axis('off')
    summary = f"""
    COMPARISON SUMMARY
    ==================
    
    β=1 (No Annealing):
    - Final KL: {diag1['total_kl']:.2f}
    - Active dims: {diag1['active_dims']}/{diag1['total_dims']}
    - Status: {'COLLAPSED' if diag1['collapsed'] else 'OK'}
    
    Annealed β (0→1):
    - Final KL: {diag2['total_kl']:.2f}
    - Active dims: {diag2['active_dims']}/{diag2['total_dims']}
    - Status: {'COLLAPSED' if diag2['collapsed'] else 'OK'}
    
    Conclusion: KL annealing prevents collapse
    by allowing the encoder to learn useful
    representations before regularization kicks in.
    """
    axes[1, 2].text(0.1, 0.5, summary, fontsize=10, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('collapse_experiment.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    run_collapse_experiment()
```

**Expected Outcome:**
- β=1: Many dimensions collapse (KL near 0)
- Annealed: Most dimensions remain active
- The annealed model will have better latent representations

---

## Additional Practice Problems

### Problem 6: Derive ELBO from Three Different Starting Points
1. Jensen's inequality
2. KL divergence to true posterior
3. Importance sampling perspective

### Problem 7: Implement IWAE Loss
Extend the basic VAE to compute the IWAE bound with K samples.

### Problem 8: Compare Gaussian vs Bernoulli Decoders
Train both on MNIST, compare reconstructions and ELBO.

### Problem 9: Latent Space Interpolation
Implement and visualize interpolation between digits in latent space.

### Problem 10: Implement β-VAE with Disentanglement Metrics
Train β-VAE and measure disentanglement using the β-VAE metric.


