"""
Variational Autoencoder (VAE) Algorithm
======================================

Complete implementation of Variational Autoencoder for unsupervised learning,
data generation, and representation learning with mathematical foundations
and practical applications.

Mathematical Foundation:
VAE Loss = Reconstruction Loss + KL Divergence
L(θ,φ) = E_q[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))

Where:
- q_φ(z|x): Encoder (recognition network)
- p_θ(x|z): Decoder (generative network)
- p(z): Prior distribution (typically N(0,I))

Applications:
- Data generation
- Dimensionality reduction
- Anomaly detection
- Feature learning
- Image synthesis
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class VAEConfig:
    """Configuration for VAE model."""
    input_dim: int = 784        # Input dimension (e.g., 28*28 for MNIST)
    hidden_dims: List[int] = None  # Hidden layer dimensions
    latent_dim: int = 20        # Latent space dimension
    learning_rate: float = 1e-3
    beta: float = 1.0           # Weight for KL divergence
    batch_size: int = 128
    num_epochs: int = 100
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [400, 200]


class Encoder(nn.Module):
    """
    Encoder network that maps input to latent distribution parameters.
    
    Maps x -> (μ, σ) of q(z|x)
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder_layers = nn.Sequential(*layers)
        
        # Latent distribution parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        h = self.encoder_layers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network that maps latent codes to reconstructions.
    
    Maps z -> x_reconstructed
    """
    
    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        # Build decoder layers (reverse of encoder)
        layers = []
        prev_dim = latent_dim
        
        # Reverse hidden dimensions
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # For images in [0,1] range
        
        self.decoder_layers = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            
        Returns:
            Reconstructed output [batch_size, output_dim]
        """
        return self.decoder_layers(z)


class VariationalAutoencoder(nn.Module):
    """
    Complete Variational Autoencoder implementation.
    
    Combines encoder and decoder with reparameterization trick
    for end-to-end training via gradient descent.
    """
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        self.encoder = Encoder(
            config.input_dim, 
            config.hidden_dims, 
            config.latent_dim
        )
        
        self.decoder = Decoder(
            config.latent_dim,
            config.hidden_dims,
            config.input_dim
        )
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,I)
        
        This allows gradients to flow through the sampling operation.
        
        Args:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log-variance of latent distribution [batch_size, latent_dim]
            
        Returns:
            Sampled latent codes [batch_size, latent_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate samples from the learned distribution.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples [num_samples, input_dim]
        """
        if device is None:
            device = next(self.parameters()).device
            
        with torch.no_grad():
            # Sample from prior p(z) ~ N(0, I)
            z = torch.randn(num_samples, self.config.latent_dim, device=device)
            
            # Decode to generate samples
            samples = self.decoder(z)
            
        return samples
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            Latent codes [batch_size, latent_dim]
        """
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to reconstructions.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            
        Returns:
            Reconstructions [batch_size, input_dim]
        """
        with torch.no_grad():
            return self.decoder(z)


def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor, 
                     beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate VAE loss with reconstruction and KL divergence terms.
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Latent distribution mean
        logvar: Latent distribution log-variance
        beta: Weight for KL divergence (β-VAE)
        
    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction loss (Binary Cross Entropy for images)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0,I)
    # KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_div
    
    return total_loss, recon_loss, kl_div


class VAETrainer:
    """
    Trainer class for Variational Autoencoder.
    
    Handles training loop, validation, and metrics tracking.
    """
    
    def __init__(self, model: VariationalAutoencoder, config: VAEConfig):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.training_history = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'val_loss': []
        }
        
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary with average losses
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            # Flatten images if needed
            data = data.view(data.size(0), -1)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = self.model(data)
            
            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss_function(
                recon_batch, data, mu, logvar, self.config.beta
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.view(data.size(0), -1)
                
                recon_batch, mu, logvar = self.model(data)
                
                loss, recon_loss, kl_loss = vae_loss_function(
                    recon_batch, data, mu, logvar, self.config.beta
                )
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                num_batches += 1
        
        return {
            'val_total_loss': total_loss / num_batches,
            'val_recon_loss': total_recon_loss / num_batches,
            'val_kl_loss': total_kl_loss / num_batches
        }
    
    def train(self, train_loader, val_loader=None) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            Training history dictionary
        """
        print(f"Starting VAE training for {self.config.num_epochs} epochs...")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            
            # Record history
            self.training_history['total_loss'].append(train_metrics['total_loss'])
            self.training_history['recon_loss'].append(train_metrics['recon_loss'])
            self.training_history['kl_loss'].append(train_metrics['kl_loss'])
            
            if val_loader is not None:
                self.training_history['val_loss'].append(val_metrics['val_total_loss'])
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: "
                      f"Loss={train_metrics['total_loss']:.2f}, "
                      f"Recon={train_metrics['recon_loss']:.2f}, "
                      f"KL={train_metrics['kl_loss']:.2f}")
                
                if val_loader is not None:
                    print(f"         Val Loss={val_metrics['val_total_loss']:.2f}")
        
        return self.training_history


def create_synthetic_data(n_samples: int = 1000, dim: int = 50) -> torch.Tensor:
    """
    Create synthetic data for VAE demonstration.
    
    Args:
        n_samples: Number of samples
        dim: Data dimension
        
    Returns:
        Synthetic dataset
    """
    # Create data with some underlying structure
    # Two clusters in latent space
    cluster1 = torch.randn(n_samples // 2, 2) + torch.tensor([2.0, 2.0])
    cluster2 = torch.randn(n_samples // 2, 2) + torch.tensor([-2.0, -2.0])
    latent = torch.cat([cluster1, cluster2], dim=0)
    
    # Project to higher dimensional space with non-linear transformation
    W = torch.randn(2, dim)
    data = torch.tanh(latent @ W) * 0.5 + 0.5  # Scale to [0,1]
    
    # Add some noise
    data += torch.randn_like(data) * 0.1
    data = torch.clamp(data, 0, 1)
    
    return data


def analyze_latent_space(model: VariationalAutoencoder, data: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Analyze the learned latent space.
    
    Args:
        model: Trained VAE model
        data: Input data
        
    Returns:
        Dictionary with latent space analysis
    """
    model.eval()
    
    with torch.no_grad():
        # Encode data to latent space
        mu, logvar = model.encoder(data)
        z = model.reparameterize(mu, logvar)
        
        # Calculate statistics
        latent_mean = torch.mean(z, dim=0)
        latent_std = torch.std(z, dim=0)
        latent_corr = torch.corrcoef(z.T)
        
        # Reconstruction quality
        recon = model.decoder(z)
        mse = F.mse_loss(recon, data, reduction='none').mean(dim=1)
        
    return {
        'latent_codes': z.numpy(),
        'latent_mean': latent_mean.numpy(),
        'latent_std': latent_std.numpy(),
        'latent_correlation': latent_corr.numpy(),
        'reconstruction_mse': mse.numpy()
    }


def vae_comprehensive_example():
    """Comprehensive example demonstrating VAE capabilities."""
    print("=== Variational Autoencoder Example ===")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = VAEConfig(
        input_dim=50,
        hidden_dims=[100, 50],
        latent_dim=10,
        learning_rate=1e-3,
        beta=1.0,
        batch_size=64,
        num_epochs=100
    )
    
    print(f"VAE Configuration:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Latent dim: {config.latent_dim}")
    print(f"  Beta: {config.beta}")
    print(f"  Learning rate: {config.learning_rate}")
    print()
    
    # Create synthetic data
    data = create_synthetic_data(n_samples=2000, dim=config.input_dim)
    
    # Split into train/validation
    train_data = data[:1600]
    val_data = data[1600:]
    
    print(f"Data:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
    print()
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_data, torch.zeros(len(train_data)))
    val_dataset = torch.utils.data.TensorDataset(val_data, torch.zeros(len(val_data)))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    
    # Initialize model and trainer
    model = VariationalAutoencoder(config)
    trainer = VAETrainer(model, config)
    
    print(f"Model Architecture:")
    print(f"  Encoder: {config.input_dim} -> {' -> '.join(map(str, config.hidden_dims))} -> {config.latent_dim}")
    print(f"  Decoder: {config.latent_dim} -> {' -> '.join(map(str, reversed(config.hidden_dims)))} -> {config.input_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Train the model
    history = trainer.train(train_loader, val_loader)
    
    # Analyze results
    print("\n=== Training Results ===")
    final_train_loss = history['total_loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    print(f"Final training loss: {final_train_loss:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final reconstruction loss: {history['recon_loss'][-1]:.4f}")
    print(f"Final KL loss: {history['kl_loss'][-1]:.4f}")
    print()
    
    # Latent space analysis
    print("=== Latent Space Analysis ===")
    
    latent_analysis = analyze_latent_space(model, data)
    
    print(f"Latent Statistics:")
    print(f"  Mean: {np.mean(latent_analysis['latent_mean']):.4f} ± {np.std(latent_analysis['latent_mean']):.4f}")
    print(f"  Std: {np.mean(latent_analysis['latent_std']):.4f} ± {np.std(latent_analysis['latent_std']):.4f}")
    print(f"  Dimension utilization: {np.sum(latent_analysis['latent_std'] > 0.1)}/{config.latent_dim} dimensions active")
    print()
    
    # Reconstruction quality
    recon_mse = latent_analysis['reconstruction_mse']
    print(f"Reconstruction Quality:")
    print(f"  Mean MSE: {np.mean(recon_mse):.6f}")
    print(f"  Std MSE: {np.std(recon_mse):.6f}")
    print(f"  Max MSE: {np.max(recon_mse):.6f}")
    print(f"  Min MSE: {np.min(recon_mse):.6f}")
    print()
    
    # Generation capabilities
    print("=== Generation Analysis ===")
    
    # Generate new samples
    generated_samples = model.sample(100)
    
    print(f"Generated Samples:")
    print(f"  Shape: {generated_samples.shape}")
    print(f"  Range: [{generated_samples.min():.3f}, {generated_samples.max():.3f}]")
    print(f"  Mean: {generated_samples.mean():.3f}")
    print(f"  Std: {generated_samples.std():.3f}")
    print()
    
    # Compare distributions
    original_mean = data.mean(dim=0)
    generated_mean = generated_samples.mean(dim=0)
    mean_diff = torch.abs(original_mean - generated_mean).mean()
    
    print(f"Distribution Comparison:")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Original data std: {data.std():.6f}")
    print(f"  Generated data std: {generated_samples.std():.6f}")
    print()
    
    # Interpolation in latent space
    print("=== Latent Space Interpolation ===")
    
    # Take two random points in latent space
    z1 = torch.randn(1, config.latent_dim)
    z2 = torch.randn(1, config.latent_dim)
    
    # Create interpolation
    alphas = torch.linspace(0, 1, 11)
    interpolated_samples = []
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            x_interp = model.decode(z_interp)
            interpolated_samples.append(x_interp)
    
    interpolated_samples = torch.cat(interpolated_samples, dim=0)
    
    print(f"Interpolation smoothness:")
    diffs = torch.diff(interpolated_samples, dim=0)
    smoothness = torch.mean(torch.norm(diffs, dim=1))
    print(f"  Average step size: {smoothness:.6f}")
    print()
    
    # β-VAE analysis (disentanglement)
    print("=== Disentanglement Analysis (β-VAE) ===")
    
    # Vary each latent dimension independently
    base_z = torch.zeros(1, config.latent_dim)
    variations = []
    
    with torch.no_grad():
        for dim in range(min(5, config.latent_dim)):  # Analyze first 5 dimensions
            z_varied = base_z.clone()
            
            # Vary this dimension from -3 to 3
            values = torch.linspace(-3, 3, 7)
            dim_variations = []
            
            for val in values:
                z_varied[0, dim] = val
                x_varied = model.decode(z_varied)
                dim_variations.append(x_varied)
            
            dim_variations = torch.cat(dim_variations, dim=0)
            
            # Calculate variation magnitude
            variation_magnitude = torch.std(dim_variations, dim=0).mean()
            variations.append(variation_magnitude.item())
            
            print(f"  Dimension {dim}: variation magnitude = {variation_magnitude:.6f}")
    
    print(f"  Average variation: {np.mean(variations):.6f}")
    print(f"  Variation std: {np.std(variations):.6f}")


if __name__ == "__main__":
    vae_comprehensive_example()