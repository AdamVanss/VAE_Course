"""
Minimal VAE Implementation in PyTorch
=====================================
A complete, self-contained VAE for MNIST that you can run directly.

Requirements:
    pip install torch torchvision matplotlib numpy

Run:
    python minimal_vae_pytorch.py

Expected output: Training progress, saved images of reconstructions and samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    'batch_size': 128,
    'epochs': 20,
    'learning_rate': 1e-3,
    'latent_dim': 20,
    'hidden_dim': 400,
    'input_dim': 784,  # 28x28 MNIST
    'seed': 42,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# MODEL
# =============================================================================
class VAE(nn.Module):
    """
    Variational Autoencoder with fully-connected layers.
    
    Encoder: x (784) -> h (400) -> h (400) -> mu, log_var (20 each)
    Decoder: z (20) -> h (400) -> h (400) -> x_recon (784)
    """
    
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decode(self, z):
        """Decode latent to reconstruction logits."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    @torch.no_grad()
    def sample(self, num_samples, device):
        """Generate samples from the prior."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return torch.sigmoid(self.decode(z))


# =============================================================================
# LOSS FUNCTION
# =============================================================================
def vae_loss(x_recon, x, mu, log_var):
    """
    VAE Loss = Reconstruction + KL Divergence
    
    Returns: total_loss, recon_loss, kl_loss (all scalars)
    """
    batch_size = x.size(0)
    
    # Reconstruction: Binary Cross Entropy
    recon_loss = F.binary_cross_entropy_with_logits(
        x_recon, x, reduction='sum'
    ) / batch_size
    
    # KL Divergence: KL(N(mu, sigma^2) || N(0, 1))
    kl_loss = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    ) / batch_size
    
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss


# =============================================================================
# DATA
# =============================================================================
def get_data_loaders(batch_size):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784
    ])
    
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# =============================================================================
# TRAINING
# =============================================================================
def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    for x, _ in train_loader:
        x = x.to(device)
        
        optimizer.zero_grad()
        x_recon, mu, log_var = model(x)
        loss, recon, kl = vae_loss(x_recon, x, mu, log_var)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
    
    n = len(train_loader)
    return total_loss/n, total_recon/n, total_kl/n


def evaluate(model, test_loader, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_recon, mu, log_var = model(x)
            loss, _, _ = vae_loss(x_recon, x, mu, log_var)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_results(model, test_loader, device, output_dir='.'):
    """Generate and save visualizations."""
    model.eval()
    
    # Reconstructions
    x, _ = next(iter(test_loader))
    x = x[:10].to(device)
    with torch.no_grad():
        x_recon, _, _ = model(x)
        x_recon = torch.sigmoid(x_recon)
    
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(x[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(x_recon[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_title('Original')
    axes[1, 0].set_title('Reconstructed')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reconstructions.png'), dpi=150)
    plt.close()
    
    # Samples from prior
    samples = model.sample(100, device)
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(samples[i*10+j].cpu().view(28, 28), cmap='gray')
            axes[i, j].axis('off')
    plt.suptitle('Samples from Prior')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples.png'), dpi=150)
    plt.close()
    
    print(f"Saved reconstructions.png and samples.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Set seed
    torch.manual_seed(CONFIG['seed'])
    
    print(f"VAE Training on MNIST")
    print(f"Device: {DEVICE}")
    print(f"Config: {CONFIG}")
    print("-" * 50)
    
    # Data
    train_loader, test_loader = get_data_loaders(CONFIG['batch_size'])
    
    # Model
    model = VAE(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        latent_dim=CONFIG['latent_dim']
    ).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training loop
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, DEVICE)
        test_loss = evaluate(model, test_loader, DEVICE)
        
        print(f"Epoch {epoch:2d} | Train: {train_loss:.1f} (R:{train_recon:.1f}, KL:{train_kl:.1f}) | "
              f"Test: {test_loss:.1f} | ELBO: {-test_loss:.1f}")
    
    # Save results
    visualize_results(model, test_loader, DEVICE)
    torch.save(model.state_dict(), 'vae_mnist.pt')
    print(f"\nSaved model to vae_mnist.pt")
    print("Training complete!")


if __name__ == '__main__':
    main()

