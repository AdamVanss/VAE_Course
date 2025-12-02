# Part 3: Implementation

## Table of Contents
- [3.1 PyTorch Implementation](#31-pytorch-implementation)
- [3.2 TensorFlow Implementation](#32-tensorflow-implementation)
- [3.3 Loss Functions Explained](#33-loss-functions-explained)
- [3.4 Training Loop](#34-training-loop)
- [3.5 Sampling and Reconstruction](#35-sampling-and-reconstruction)
- [3.6 Visualization Code](#36-visualization-code)
- [3.7 Recap](#37-recap)

---

## 3.1 PyTorch Implementation

### Complete Minimal VAE

```python
"""
Minimal VAE Implementation in PyTorch
=====================================
A complete, runnable VAE for MNIST.

Requirements:
    pip install torch torchvision matplotlib numpy

Run:
    python minimal_vae_pytorch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
LATENT_DIM = 20
HIDDEN_DIM = 400
INPUT_DIM = 784  # 28 * 28 for MNIST
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# MODEL DEFINITION
# =============================================================================

class VAE(nn.Module):
    """
    Variational Autoencoder with fully-connected layers.
    
    Architecture:
        Encoder: x (784) -> h (400) -> mu (20), log_var (20)
        Decoder: z (20) -> h (400) -> x_recon (784)
    """
    
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # =====================================================================
        # ENCODER
        # Maps input x to parameters of q(z|x) = N(mu, diag(sigma^2))
        # =====================================================================
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # (batch, 784) -> (batch, 400)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # (batch, 400) -> (batch, 400)
            nn.ReLU(),
        )
        
        # Output layers for mean and log-variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # (batch, 400) -> (batch, 20)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim) # (batch, 400) -> (batch, 20)
        
        # =====================================================================
        # DECODER
        # Maps latent z to parameters of p(x|z)
        # For Bernoulli: outputs logits for each pixel
        # =====================================================================
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # (batch, 20) -> (batch, 400)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # (batch, 400) -> (batch, 400)
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),   # (batch, 400) -> (batch, 784)
            # No activation - these are logits for Bernoulli
        )
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            mu: Mean of q(z|x), shape (batch_size, latent_dim)
            log_var: Log variance of q(z|x), shape (batch_size, latent_dim)
        """
        # x: (batch_size, 784)
        h = self.encoder(x)           # h: (batch_size, 400)
        mu = self.fc_mu(h)            # mu: (batch_size, 20)
        log_var = self.fc_log_var(h)  # log_var: (batch_size, 20)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        
        This allows gradients to flow through the sampling operation.
        
        Args:
            mu: Mean tensor, shape (batch_size, latent_dim)
            log_var: Log variance tensor, shape (batch_size, latent_dim)
            
        Returns:
            z: Sampled latent code, shape (batch_size, latent_dim)
        """
        # std = exp(0.5 * log_var) = exp(log(sigma)) = sigma
        std = torch.exp(0.5 * log_var)  # std: (batch_size, 20)
        
        # Sample epsilon from N(0, I)
        eps = torch.randn_like(std)      # eps: (batch_size, 20)
        
        # z = mu + sigma * epsilon
        z = mu + std * eps               # z: (batch_size, 20)
        
        return z
    
    def decode(self, z):
        """
        Decode latent code to reconstruction logits.
        
        Args:
            z: Latent code, shape (batch_size, latent_dim)
            
        Returns:
            logits: Reconstruction logits, shape (batch_size, input_dim)
        """
        # z: (batch_size, 20)
        logits = self.decoder(z)  # logits: (batch_size, 784)
        return logits
    
    def forward(self, x):
        """
        Full forward pass for training.
        
        Args:
            x: Input tensor, shape (batch_size, input_dim)
            
        Returns:
            x_recon: Reconstruction logits, shape (batch_size, input_dim)
            mu: Encoder mean, shape (batch_size, latent_dim)
            log_var: Encoder log variance, shape (batch_size, latent_dim)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def sample(self, num_samples):
        """
        Generate new samples from the model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            samples: Generated samples, shape (num_samples, input_dim)
        """
        # Sample from prior p(z) = N(0, I)
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
        
        # Decode to get reconstruction parameters
        logits = self.decode(z)
        
        # Apply sigmoid to get probabilities (for visualization)
        samples = torch.sigmoid(logits)
        
        return samples

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def vae_loss(x_recon, x, mu, log_var):
    """
    Compute the VAE loss = -ELBO = Reconstruction Loss + KL Divergence
    
    Args:
        x_recon: Reconstruction logits from decoder, shape (batch_size, input_dim)
        x: Original input, shape (batch_size, input_dim)
        mu: Encoder mean, shape (batch_size, latent_dim)
        log_var: Encoder log variance, shape (batch_size, latent_dim)
        
    Returns:
        loss: Total loss (scalar)
        recon_loss: Reconstruction loss (scalar)
        kl_loss: KL divergence loss (scalar)
    """
    batch_size = x.size(0)
    
    # =========================================================================
    # RECONSTRUCTION LOSS
    # For Bernoulli decoder: Binary Cross Entropy
    # E_q[log p(x|z)] ≈ -BCE(x, sigmoid(x_recon))
    # =========================================================================
    # Note: binary_cross_entropy_with_logits is numerically stable
    # It computes: -[x * log(sigmoid(logits)) + (1-x) * log(1-sigmoid(logits))]
    # Sum over all pixels, average over batch
    recon_loss = F.binary_cross_entropy_with_logits(
        x_recon,  # logits
        x,        # targets (original pixels in [0,1])
        reduction='sum'  # Sum over pixels and batch
    ) / batch_size  # Average over batch
    
    # =========================================================================
    # KL DIVERGENCE
    # KL(q(z|x) || p(z)) where q(z|x) = N(mu, sigma^2), p(z) = N(0, I)
    # = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    # = 0.5 * sum(mu^2 + exp(log_var) - log_var - 1)
    # =========================================================================
    # Sum over latent dimensions, average over batch
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    
    # Total loss
    loss = recon_loss + kl_loss
    
    return loss, recon_loss, kl_loss

# =============================================================================
# DATA LOADING
# =============================================================================

def get_mnist_loaders(batch_size=BATCH_SIZE):
    """
    Load MNIST dataset and create data loaders.
    
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    # Transform: convert to tensor and flatten
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] range
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten: (1, 28, 28) -> (784,)
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set >0 for faster loading on multi-core systems
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader

# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, train_loader, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: VAE model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        
    Returns:
        avg_loss: Average loss over epoch
        avg_recon: Average reconstruction loss
        avg_kl: Average KL divergence
    """
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        # data: (batch_size, 784), _: labels (unused)
        data = data.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, mu, log_var = model(data)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(x_recon, data, mu, log_var)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    
    # Compute averages
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    avg_recon = total_recon / n_batches
    avg_kl = total_kl / n_batches
    
    return avg_loss, avg_recon, avg_kl

def evaluate(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Args:
        model: VAE model
        test_loader: Test data loader
        device: Device to use
        
    Returns:
        avg_loss: Average loss
        avg_recon: Average reconstruction loss
        avg_kl: Average KL divergence
    """
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            x_recon, mu, log_var = model(data)
            loss, recon_loss, kl_loss = vae_loss(x_recon, data, mu, log_var)
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
    
    n_batches = len(test_loader)
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_reconstructions(model, test_loader, device, n_samples=10):
    """
    Visualize original images and their reconstructions.
    """
    model.eval()
    data, _ = next(iter(test_loader))
    data = data[:n_samples].to(device)
    
    with torch.no_grad():
        x_recon, _, _ = model(data)
        x_recon = torch.sigmoid(x_recon)  # Convert logits to probabilities
    
    # Plot
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(data[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original')
        
        # Reconstruction
        axes[1, i].imshow(x_recon[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstruction')
    
    plt.tight_layout()
    plt.savefig('reconstructions.png', dpi=150)
    plt.close()
    print("Saved reconstructions.png")

def visualize_samples(model, device, n_samples=100):
    """
    Visualize samples generated from the model.
    """
    model.eval()
    samples = model.sample(n_samples)
    
    # Plot in a grid
    n_rows = 10
    n_cols = 10
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            axes[i, j].imshow(samples[idx].cpu().view(28, 28), cmap='gray')
            axes[i, j].axis('off')
    
    plt.suptitle('Generated Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('samples.png', dpi=150)
    plt.close()
    print("Saved samples.png")

def visualize_latent_space(model, test_loader, device):
    """
    Visualize the 2D latent space (only works if latent_dim=2).
    For higher dimensions, we visualize the first two dimensions.
    """
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            latents.append(mu.cpu())
            labels.append(label)
    
    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    # Plot first two dimensions
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.5, s=1)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('Latent Space (first 2 dimensions)')
    plt.savefig('latent_space.png', dpi=150)
    plt.close()
    print("Saved latent_space.png")

def plot_training_curves(train_losses, test_losses, recon_losses, kl_losses):
    """
    Plot training curves.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(epochs, train_losses, label='Train')
    axes[0].plot(epochs, test_losses, label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss (-ELBO)')
    axes[0].legend()
    
    # Reconstruction loss
    axes[1].plot(epochs, recon_losses)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss (BCE)')
    
    # KL loss
    axes[2].plot(epochs, kl_losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.close()
    print("Saved training_curves.png")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"Hyperparameters: batch_size={BATCH_SIZE}, epochs={EPOCHS}, "
          f"latent_dim={LATENT_DIM}, lr={LEARNING_RATE}")
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = VAE().to(DEVICE)
    print(f"\nModel architecture:\n{model}")
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    train_losses = []
    test_losses = []
    recon_losses = []
    kl_losses = []
    
    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, DEVICE)
        
        # Evaluate
        test_loss, test_recon, test_kl = evaluate(model, test_loader, DEVICE)
        
        # Store metrics
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        recon_losses.append(train_recon)
        kl_losses.append(train_kl)
        
        # Print progress
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.2f} | "
              f"Test Loss: {test_loss:.2f} | "
              f"Recon: {train_recon:.2f} | "
              f"KL: {train_kl:.2f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_reconstructions(model, test_loader, DEVICE)
    visualize_samples(model, DEVICE)
    visualize_latent_space(model, test_loader, DEVICE)
    plot_training_curves(train_losses, test_losses, recon_losses, kl_losses)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
    }, 'vae_model.pt')
    print("\nSaved model to vae_model.pt")
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()
```

---

## 3.2 TensorFlow Implementation

### Complete Minimal VAE in TensorFlow/Keras

```python
"""
Minimal VAE Implementation in TensorFlow/Keras
==============================================
A complete, runnable VAE for MNIST.

Requirements:
    pip install tensorflow matplotlib numpy

Run:
    python minimal_vae_tensorflow.py
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
LATENT_DIM = 20
HIDDEN_DIM = 400
INPUT_DIM = 784

# =============================================================================
# SAMPLING LAYER (Reparameterization)
# =============================================================================

class Sampling(layers.Layer):
    """
    Custom layer that performs the reparameterization trick.
    
    Given mean and log_var, samples z = mean + std * epsilon
    where epsilon ~ N(0, I).
    """
    
    def call(self, inputs):
        """
        Args:
            inputs: tuple of (mean, log_var), each shape (batch, latent_dim)
            
        Returns:
            z: Sampled latent code, shape (batch, latent_dim)
        """
        mean, log_var = inputs
        
        # Get batch size and latent dimension
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        
        # Sample epsilon from N(0, I)
        epsilon = tf.random.normal(shape=(batch, dim))
        
        # Reparameterization: z = mean + std * epsilon
        # std = exp(0.5 * log_var)
        z = mean + tf.exp(0.5 * log_var) * epsilon
        
        return z

# =============================================================================
# VAE MODEL
# =============================================================================

class VAE(keras.Model):
    """
    Variational Autoencoder implemented as a Keras Model.
    """
    
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # =====================================================================
        # ENCODER
        # =====================================================================
        self.encoder = keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
        ], name='encoder_backbone')
        
        # Mean and log variance outputs
        self.fc_mean = layers.Dense(latent_dim, name='mean')
        self.fc_log_var = layers.Dense(latent_dim, name='log_var')
        
        # Sampling layer
        self.sampling = Sampling()
        
        # =====================================================================
        # DECODER
        # =====================================================================
        self.decoder = keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(input_dim),  # Output logits
        ], name='decoder')
        
        # Metrics trackers
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.recon_loss_tracker = keras.metrics.Mean(name='recon_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input, shape (batch, input_dim)
            
        Returns:
            mean: Shape (batch, latent_dim)
            log_var: Shape (batch, latent_dim)
            z: Sampled latent, shape (batch, latent_dim)
        """
        h = self.encoder(x)
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        z = self.sampling([mean, log_var])
        return mean, log_var, z
    
    def decode(self, z):
        """
        Decode latent code to reconstruction.
        
        Args:
            z: Latent code, shape (batch, latent_dim)
            
        Returns:
            logits: Reconstruction logits, shape (batch, input_dim)
        """
        return self.decoder(z)
    
    def call(self, x, training=None):
        """
        Full forward pass.
        """
        mean, log_var, z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mean, log_var
    
    def train_step(self, data):
        """
        Custom training step.
        """
        # data is just x (no labels)
        x = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            mean, log_var, z = self.encode(x)
            x_recon = self.decode(z)
            
            # Reconstruction loss (BCE)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(x, tf.sigmoid(x_recon)),
                    axis=-1
                )
            )
            
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + log_var - tf.square(mean) - tf.exp(log_var),
                    axis=-1
                )
            )
            
            # Total loss
            total_loss = recon_loss + kl_loss
        
        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        
        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            'loss': self.total_loss_tracker.result(),
            'recon_loss': self.recon_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        """
        Custom test step.
        """
        x = data
        mean, log_var, z = self.encode(x)
        x_recon = self.decode(z)
        
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(x, tf.sigmoid(x_recon)),
                axis=-1
            )
        )
        
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + log_var - tf.square(mean) - tf.exp(log_var),
                axis=-1
            )
        )
        
        total_loss = recon_loss + kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }
    
    def sample(self, num_samples):
        """
        Generate new samples from the model.
        """
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        logits = self.decode(z)
        return tf.sigmoid(logits)

# =============================================================================
# DATA LOADING
# =============================================================================

def get_mnist_data():
    """
    Load and preprocess MNIST data.
    """
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    
    # Normalize to [0, 1] and flatten
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    return x_train, x_test

# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_tf(model, x_test):
    """
    Generate visualizations.
    """
    # Reconstructions
    x_sample = x_test[:10]
    x_recon, _, _ = model(x_sample)
    x_recon = tf.sigmoid(x_recon).numpy()
    
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(x_sample[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(x_recon[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
    plt.savefig('tf_reconstructions.png', dpi=150)
    plt.close()
    
    # Samples
    samples = model.sample(100).numpy()
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(10):
        for j in range(10):
            axes[i, j].imshow(samples[i*10+j].reshape(28, 28), cmap='gray')
            axes[i, j].axis('off')
    plt.savefig('tf_samples.png', dpi=150)
    plt.close()
    print("Saved tf_reconstructions.png and tf_samples.png")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("TensorFlow VAE for MNIST")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Load data
    print("\nLoading data...")
    x_train, x_test = get_mnist_data()
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Create model
    model = VAE()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    
    # Build model by calling it once
    _ = model(x_train[:1])
    model.summary()
    
    # Train
    print("\nTraining...")
    history = model.fit(
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test,),
        verbose=1
    )
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_tf(model, x_test)
    
    # Save model
    model.save_weights('vae_tf_weights.h5')
    print("\nSaved model weights to vae_tf_weights.h5")
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()
```

---

## 3.3 Loss Functions Explained

### Reconstruction Loss: Three Common Choices

#### 1. Binary Cross-Entropy (Bernoulli Decoder)

For binary or grayscale images in [0, 1]:
\[
\mathcal{L}_{\text{recon}} = -\sum_{i=1}^{D} \left[ x_i \log \hat{x}_i + (1-x_i) \log(1-\hat{x}_i) \right]
\]

where \(\hat{x}_i = \sigma(\text{logit}_i)\) is the decoder output after sigmoid.

```python
# PyTorch
recon_loss = F.binary_cross_entropy_with_logits(logits, x, reduction='sum')

# Equivalently, with sigmoid:
# recon_loss = F.binary_cross_entropy(torch.sigmoid(logits), x, reduction='sum')
```

**When to use:** Binarized images, images treated as probabilities.

#### 2. Mean Squared Error (Gaussian Decoder with fixed variance)

For continuous data:
\[
\mathcal{L}_{\text{recon}} = \frac{1}{2\sigma^2}\sum_{i=1}^{D} (x_i - \mu_i)^2 + \frac{D}{2}\log(2\pi\sigma^2)
\]

With \(\sigma^2 = 1\), this simplifies to MSE (up to constants):
\[
\mathcal{L}_{\text{recon}} = \frac{1}{2}\sum_{i=1}^{D} (x_i - \mu_i)^2
\]

```python
# PyTorch
recon_loss = F.mse_loss(x_recon, x, reduction='sum') * 0.5

# Or simply (ignoring constants):
recon_loss = F.mse_loss(x_recon, x, reduction='sum')
```

**When to use:** Continuous data, when you want sharper reconstructions.

**Caution:** The implicit variance assumption affects the KL-reconstruction trade-off.

#### 3. Gaussian with Learned Variance

The decoder outputs both \(\mu\) and \(\log \sigma^2\):
\[
\mathcal{L}_{\text{recon}} = \sum_{i=1}^{D} \left[ \frac{1}{2}\log \sigma_i^2 + \frac{(x_i - \mu_i)^2}{2\sigma_i^2} \right] + \text{const}
\]

```python
def gaussian_nll(x, mu, log_var):
    """Negative log-likelihood for Gaussian."""
    return 0.5 * torch.sum(log_var + (x - mu).pow(2) / log_var.exp())
```

**When to use:** When reconstruction quality varies across the image (e.g., edges vs. flat regions).

### KL Divergence Loss

Always computed analytically for Gaussian encoder:

```python
def kl_divergence(mu, log_var):
    """
    KL(N(mu, sigma^2) || N(0, I))
    """
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
```

### Putting It Together

```python
def vae_loss(x_recon, x, mu, log_var, decoder_type='bernoulli'):
    """
    Complete VAE loss computation.
    
    Args:
        x_recon: Decoder output (logits for Bernoulli, mean for Gaussian)
        x: Original input
        mu, log_var: Encoder outputs
        decoder_type: 'bernoulli' or 'gaussian'
    """
    batch_size = x.size(0)
    
    # Reconstruction loss
    if decoder_type == 'bernoulli':
        recon = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum')
    elif decoder_type == 'gaussian':
        recon = F.mse_loss(x_recon, x, reduction='sum') * 0.5
    
    # KL divergence
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Average over batch
    loss = (recon + kl) / batch_size
    
    return loss
```

---

## 3.4 Training Loop

### Standard Training Loop with Monitoring

```python
def train_vae(model, train_loader, test_loader, epochs, device, lr=1e-3):
    """
    Full training loop with monitoring and early stopping.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    history = {
        'train_loss': [], 'test_loss': [],
        'train_recon': [], 'train_kl': [],
        'test_recon': [], 'test_kl': [],
    }
    
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(1, epochs + 1):
        # =====================================================================
        # TRAINING
        # =====================================================================
        model.train()
        train_loss, train_recon, train_kl = 0, 0, 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            x_recon, mu, log_var = model(data)
            loss, recon, kl = vae_loss(x_recon, data, mu, log_var)
            loss.backward()
            
            # Gradient clipping (optional, helps with stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
        
        n_train = len(train_loader)
        train_loss /= n_train
        train_recon /= n_train
        train_kl /= n_train
        
        # =====================================================================
        # EVALUATION
        # =====================================================================
        model.eval()
        test_loss, test_recon, test_kl = 0, 0, 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                x_recon, mu, log_var = model(data)
                loss, recon, kl = vae_loss(x_recon, data, mu, log_var)
                test_loss += loss.item()
                test_recon += recon.item()
                test_kl += kl.item()
        
        n_test = len(test_loader)
        test_loss /= n_test
        test_recon /= n_test
        test_kl /= n_test
        
        # =====================================================================
        # LOGGING AND CHECKPOINTING
        # =====================================================================
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_recon'].append(train_recon)
        history['train_kl'].append(train_kl)
        history['test_recon'].append(test_recon)
        history['test_kl'].append(test_kl)
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Print progress
        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.1f} (R:{train_recon:.1f}, KL:{train_kl:.1f}) | "
              f"Test: {test_loss:.1f} (R:{test_recon:.1f}, KL:{test_kl:.1f})")
        
        # Early stopping check
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    return history
```

---

## 3.5 Sampling and Reconstruction

### Sampling from the Prior

```python
def sample_from_prior(model, num_samples, device):
    """
    Generate new samples by sampling z from the prior and decoding.
    
    This is how we generate "new" data from the learned model.
    """
    model.eval()
    with torch.no_grad():
        # Sample from prior p(z) = N(0, I)
        z = torch.randn(num_samples, model.latent_dim).to(device)
        
        # Decode
        logits = model.decode(z)
        
        # For Bernoulli decoder: apply sigmoid to get probabilities
        samples = torch.sigmoid(logits)
        
        # Optionally, sample binary values:
        # samples = torch.bernoulli(torch.sigmoid(logits))
    
    return samples
```

### Reconstruction

```python
def reconstruct(model, x, device):
    """
    Reconstruct input through the VAE.
    
    Note: Due to stochastic sampling, each call gives slightly different results.
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        x_recon, mu, log_var = model(x)
        x_recon = torch.sigmoid(x_recon)
    return x_recon, mu
```

### Interpolation in Latent Space

```python
def interpolate(model, x1, x2, num_steps, device):
    """
    Interpolate between two images in latent space.
    
    This demonstrates the smoothness of the learned latent space.
    """
    model.eval()
    with torch.no_grad():
        # Encode both images (using mean, not sampling)
        mu1, _ = model.encode(x1.unsqueeze(0).to(device))
        mu2, _ = model.encode(x2.unsqueeze(0).to(device))
        
        # Linear interpolation in latent space
        interpolations = []
        for alpha in np.linspace(0, 1, num_steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            x_interp = torch.sigmoid(model.decode(z))
            interpolations.append(x_interp)
        
        interpolations = torch.cat(interpolations, dim=0)
    
    return interpolations
```

---

## 3.6 Visualization Code

### Complete Visualization Suite

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch

def plot_reconstructions(model, data_loader, device, n_samples=10, save_path='reconstructions.png'):
    """
    Plot original images alongside their reconstructions.
    """
    model.eval()
    data, labels = next(iter(data_loader))
    data = data[:n_samples].to(device)
    
    with torch.no_grad():
        x_recon, _, _ = model(data)
        x_recon = torch.sigmoid(x_recon)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(data[i].cpu().view(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Label: {labels[i].item()}', fontsize=8)
        
        # Reconstruction
        axes[1, i].imshow(x_recon[i].cpu().view(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Recon', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_samples_grid(model, device, grid_size=10, save_path='samples_grid.png'):
    """
    Plot a grid of samples generated from the prior.
    """
    model.eval()
    n_samples = grid_size * grid_size
    
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim).to(device)
        samples = torch.sigmoid(model.decode(z))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            axes[i, j].imshow(samples[idx].cpu().view(28, 28), cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
    
    plt.suptitle('Generated Samples from Prior', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_latent_space_2d(model, data_loader, device, save_path='latent_2d.png'):
    """
    Visualize the latent space using the first two dimensions.
    Color points by their class labels.
    """
    model.eval()
    all_mu = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            all_mu.append(mu.cpu())
            all_labels.append(labels)
    
    all_mu = torch.cat(all_mu, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        all_mu[:, 0], all_mu[:, 1], 
        c=all_labels, cmap='tab10', 
        alpha=0.5, s=2
    )
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel('z[0]', fontsize=12)
    plt.ylabel('z[1]', fontsize=12)
    plt.title('Latent Space (First 2 Dimensions)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_latent_tsne(model, data_loader, device, n_samples=5000, save_path='latent_tsne.png'):
    """
    Visualize the latent space using t-SNE for high-dimensional latent spaces.
    """
    model.eval()
    all_mu = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data = data.to(device)
            mu, _ = model.encode(data)
            all_mu.append(mu.cpu())
            all_labels.append(labels)
            
            if sum(m.size(0) for m in all_mu) >= n_samples:
                break
    
    all_mu = torch.cat(all_mu, dim=0)[:n_samples].numpy()
    all_labels = torch.cat(all_labels, dim=0)[:n_samples].numpy()
    
    print("Running t-SNE... (this may take a minute)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    mu_2d = tsne.fit_transform(all_mu)
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        mu_2d[:, 0], mu_2d[:, 1], 
        c=all_labels, cmap='tab10', 
        alpha=0.6, s=5
    )
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.title('Latent Space (t-SNE Visualization)', fontsize=14)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training curves for loss, reconstruction, and KL.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['test_loss'], 'r--', label='Test', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss (-ELBO)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction
    axes[1].plot(epochs, history['train_recon'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['test_recon'], 'r--', label='Test', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL
    axes[2].plot(epochs, history['train_kl'], 'b-', label='Train', linewidth=2)
    axes[2].plot(epochs, history['test_kl'], 'r--', label='Test', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title('KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_latent_traversal(model, device, dim_range=(-3, 3), n_steps=10, save_path='latent_traversal.png'):
    """
    Visualize what each latent dimension encodes by traversing it while fixing others.
    Only shows first 10 latent dimensions.
    """
    model.eval()
    n_dims = min(10, model.latent_dim)
    
    fig, axes = plt.subplots(n_dims, n_steps, figsize=(n_steps, n_dims))
    
    with torch.no_grad():
        for dim in range(n_dims):
            for step, val in enumerate(np.linspace(dim_range[0], dim_range[1], n_steps)):
                z = torch.zeros(1, model.latent_dim).to(device)
                z[0, dim] = val
                sample = torch.sigmoid(model.decode(z))
                
                axes[dim, step].imshow(sample[0].cpu().view(28, 28), cmap='gray', vmin=0, vmax=1)
                axes[dim, step].axis('off')
                
                if step == 0:
                    axes[dim, step].set_ylabel(f'z[{dim}]', fontsize=8)
    
    plt.suptitle('Latent Dimension Traversal', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def plot_interpolation(model, x1, x2, device, n_steps=10, save_path='interpolation.png'):
    """
    Show interpolation between two images in latent space.
    """
    model.eval()
    
    with torch.no_grad():
        mu1, _ = model.encode(x1.unsqueeze(0).to(device))
        mu2, _ = model.encode(x2.unsqueeze(0).to(device))
        
        fig, axes = plt.subplots(1, n_steps + 2, figsize=(2 * (n_steps + 2), 2))
        
        # Original images at ends
        axes[0].imshow(x1.cpu().view(28, 28), cmap='gray')
        axes[0].set_title('Start')
        axes[0].axis('off')
        
        axes[-1].imshow(x2.cpu().view(28, 28), cmap='gray')
        axes[-1].set_title('End')
        axes[-1].axis('off')
        
        # Interpolations
        for i, alpha in enumerate(np.linspace(0, 1, n_steps)):
            z = (1 - alpha) * mu1 + alpha * mu2
            interp = torch.sigmoid(model.decode(z))
            axes[i + 1].imshow(interp[0].cpu().view(28, 28), cmap='gray')
            axes[i + 1].axis('off')
    
    plt.suptitle('Latent Space Interpolation', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")
```

---

## 3.7 Recap

### Implementation Checklist

| Component | Key Points |
|-----------|------------|
| **Encoder** | Outputs μ and log(σ²), not σ directly |
| **Reparameterization** | z = μ + σ ⊙ ε, where ε ~ N(0,I) |
| **Decoder** | Outputs distribution parameters (logits or means) |
| **Reconstruction Loss** | BCE for Bernoulli, MSE for Gaussian |
| **KL Loss** | Closed-form for Gaussians |
| **Loss Averaging** | Sum over dimensions, mean over batch |

### Code Structure Summary

```
VAE
├── encode(x) -> mu, log_var
├── reparameterize(mu, log_var) -> z
├── decode(z) -> x_recon
├── forward(x) -> x_recon, mu, log_var
└── sample(n) -> generated_samples

Loss = BCE(x, x_recon) + KL(q||p)
```

### Common Pitfalls

1. **Forgetting sigmoid on decoder output** for visualization (but not in BCE loss with logits)
2. **Wrong reduction** in loss (should sum over pixels, mean over batch)
3. **Not using log_var** for numerical stability
4. **Computing KL incorrectly** (sign errors, missing terms)
5. **Detaching tensors** accidentally breaking gradient flow

---

**Next:** Part 4 covers advanced topics and extensions.


