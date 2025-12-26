import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional

class LFVULoss(nn.Module):
    """
    Local Fraction of Variance Unexplained (LFVU) Loss
    Measures preservation of local neighborhood structure
    """
    def __init__(self, k: int = 10, alpha: float = 1.0):
        """
        Args:
            k: Number of nearest neighbors to consider
            alpha: Weight for LFVU loss
        """
        super().__init__()
        self.k = k
        self.alpha = alpha

    def compute_pairwise_distances(self, X: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances"""
        # X: [batch_size, dim]
        X_norm = (X ** 2).sum(dim=1, keepdim=True)
        distances = X_norm + X_norm.t() - 2 * torch.mm(X, X.t())
        distances = torch.clamp(distances, min=0.0)  # Ensure non-negative
        return torch.sqrt(distances + 1e-8)  # Add small epsilon for stability

    def normalize_distances(self, distances: torch.Tensor) -> torch.Tensor:
        """Normalize distances as in LFVU formula"""
        # distances: [batch_size, k] distances to k-nearest neighbors
        mean_dist = distances.mean(dim=1, keepdim=True)
        std_dist = distances.std(dim=1, keepdim=True) + 1e-8

        # Normalize: subtract mean and divide by std (like z-score)
        normalized = (distances - mean_dist) / std_dist
        return normalized

    def get_knn_distances(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get distances to k-nearest neighbors and their indices
        Returns:
            - knn_distances: [batch_size, k] distances to k-nearest neighbors
            - knn_indices: [batch_size, k] indices of k-nearest neighbors
        """
        batch_size = X.shape[0]

        # Compute pairwise distances
        pairwise_dist = self.compute_pairwise_distances(X)

        # Get k+1 neighbors (including self)
        k_neighbors = min(self.k + 1, batch_size)
        knn_dist, knn_idx = torch.topk(pairwise_dist, k=k_neighbors,
                                      dim=1, largest=False, sorted=True)

        # Remove self (distance 0) if present
        if k_neighbors > 1:
            knn_dist = knn_dist[:, 1:]  # Remove self
            knn_idx = knn_idx[:, 1:]    # Remove self
        else:
            knn_dist = torch.zeros((batch_size, 1), device=X.device)
            knn_idx = torch.zeros((batch_size, 1), device=X.device, dtype=torch.long)

        return knn_dist, knn_idx

    def forward(self, X_original: torch.Tensor, X_latent: torch.Tensor) -> torch.Tensor:
        """
        Compute LFVU loss

        Args:
            X_original: Original high-dimensional data [batch_size, input_dim]
            X_latent: Latent representation [batch_size, latent_dim]

        Returns:
            LFVU loss value
        """
        batch_size = X_original.shape[0]

        # Get k-nearest neighbors in original space
        d_dist, d_idx = self.get_knn_distances(X_original)

        # Get distances to same neighbors in latent space
        # We need to compute distances between each point and its k-nearest neighbors
        r_dist = []
        for i in range(batch_size):
            # Get indices of neighbors for point i
            neighbor_indices = d_idx[i]

            # Compute distances from point i to its neighbors in latent space
            point_latent = X_latent[i:i+1]  # [1, latent_dim]
            neighbors_latent = X_latent[neighbor_indices]  # [k, latent_dim]

            # Compute Euclidean distances
            dist = torch.norm(point_latent - neighbors_latent, dim=1)
            r_dist.append(dist)

        r_dist = torch.stack(r_dist)  # [batch_size, k]

        # Normalize distances
        d_norm = self.normalize_distances(d_dist)
        r_norm = self.normalize_distances(r_dist)

        # Compute LFVU for each point
        numerator = torch.sum((d_norm - r_norm) ** 2, dim=1)  # [batch_size]
        denominator = torch.sum((d_norm - d_norm.mean(dim=1, keepdim=True)) ** 2, dim=1)  # [batch_size]

        # Avoid division by zero
        denominator = torch.clamp(denominator, min=1e-8)

        # LFVU for each point
        lfvu_per_point = numerator / denominator

        # Average LFVU across batch
        lfvu_loss = lfvu_per_point.mean()

        return self.alpha * lfvu_loss


class Autoencoder(nn.Module):
    """Basic Autoencoder architecture"""
    def __init__(self, input_dim: int, latent_dim: int,
                 hidden_dims: list = [512, 256, 128]):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (reverse architecture)
        decoder_layers = []
        hidden_dims_rev = list(reversed(hidden_dims))
        prev_dim = latent_dim
        for hidden_dim in hidden_dims_rev:
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


class LFVU_AE(nn.Module):
    """
    LFVU-Autoencoder: Autoencoder with LFVU loss
    Similar to RTD-AE but using LFVU instead of RTD
    """
    def __init__(self, input_dim: int, latent_dim: int,
                 lfvu_k: int = 10, lfvu_weight: float = 1.0,
                 hidden_dims: list = [512, 256, 128]):
        super().__init__()

        self.autoencoder = Autoencoder(input_dim, latent_dim, hidden_dims)
        self.lfvu_loss = LFVULoss(k=lfvu_k, alpha=lfvu_weight)

    def forward(self, x):
        x_recon, z = self.autoencoder(x)
        return x_recon, z

    def compute_loss(self, x, x_recon, z,
                    recon_weight: float = 1.0,
                    lfvu_weight: float = 1.0) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss: reconstruction + LFVU

        Args:
            x: Original input
            x_recon: Reconstructed input
            z: Latent representation
            recon_weight: Weight for reconstruction loss
            lfvu_weight: Weight for LFVU loss

        Returns:
            total_loss, loss_dict
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)

        # LFVU loss
        lfvu_loss_value = self.lfvu_loss(x, z)

        # Combined loss
        total_loss = recon_weight * recon_loss + lfvu_weight * lfvu_loss_value

        # Return loss dictionary for monitoring
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'lfvu': lfvu_loss_value.item(),
            'recon_weighted': (recon_weight * recon_loss).item(),
            'lfvu_weighted': (lfvu_weight * lfvu_loss_value).item()
        }

        return total_loss, loss_dict


class LFVU_AETrainer:
    """Training wrapper for LFVU-AE"""
    def __init__(self, model, lr: float = 1e-3, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.75
        )

    def train_step(self, x_batch):
        self.model.train()
        self.optimizer.zero_grad()

        x_batch = x_batch.to(self.device)
        x_recon, z = self.model(x_batch)

        # Two-phase training like RTD-AE:
        # Phase 1: Only reconstruction (first few epochs)
        # Phase 2: Reconstruction + LFVU

        # For simplicity, we'll use combined loss with weights
        # In practice, you might want to implement the two-phase training
        # from the RTD-AE paper
        loss, loss_dict = self.model.compute_loss(
            x_batch, x_recon, z,
            recon_weight=1.0,
            lfvu_weight=1.0  # Can be adjusted or made 0 for phase 1
        )

        loss.backward()
        self.optimizer.step()

        return loss_dict

    def validate(self, x_batch):
        self.model.eval()
        with torch.no_grad():
            x_batch = x_batch.to(self.device)
            x_recon, z = self.model(x_batch)
            loss, loss_dict = self.model.compute_loss(
                x_batch, x_recon, z,
                recon_weight=1.0,
                lfvu_weight=1.0
            )
        return loss_dict

    def train(self, train_loader, val_loader, epochs: int,
              lfvu_start_epoch: int = 200):
        """
        Train with two-phase approach similar to RTD-AE

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Total training epochs
            lfvu_start_epoch: Epoch to start adding LFVU loss (like RTD-AE)
        """
        train_history = []
        val_history = []

        for epoch in range(epochs):
            # Phase-based training
            if epoch < lfvu_start_epoch:
                # Phase 1: Only reconstruction
                lfvu_weight = 0.0
            else:
                # Phase 2: Reconstruction + LFVU
                lfvu_weight = 0.05

            # Training
            epoch_train_losses = []
            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    x_batch = batch[0]
                else:
                    x_batch = batch

                # Update LFVU weight in model
                self.model.lfvu_loss.alpha = lfvu_weight

                loss_dict = self.train_step(x_batch)
                epoch_train_losses.append(loss_dict['total'])

            # Validation
            epoch_val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)):
                        x_batch = batch[0]
                    else:
                        x_batch = batch

                    val_dict = self.validate(x_batch)
                    epoch_val_losses.append(val_dict['total'])

            # Record history
            train_history.append(np.mean(epoch_train_losses))
            val_history.append(np.mean(epoch_val_losses))

            # Update learning rate
            self.scheduler.step(np.mean(epoch_val_losses))

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_history[-1]:.4f}, "
                      f"Train Recon Loss: {loss_dict['recon']:.4f}, "
                      f"Train Mean LFVU: {loss_dict['lfvu']:.4f}, \n"
                      f"Val Loss: {val_history[-1]:.4f}, "
                      f"Val Recon Loss: {val_dict['recon']:.4f}, "
                      f"Val Mean LFVU: {val_dict['lfvu']:.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                      f"LFVU Weight: {lfvu_weight}")

        return train_history, val_history


# Example usage
def create_lfvu_ae(input_dim=784, latent_dim=2, lfvu_k=10):
    """Create LFVU-AE model for MNIST-like data"""
    model = LFVU_AE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        lfvu_k=lfvu_k,
        lfvu_weight=1.0,
        hidden_dims=[512, 256, 128]
    )
    return model


def create_lfvu_ae_for_high_dim(input_dim=16384, latent_dim=16, lfvu_k=10):
    """Create LFVU-AE model for high-dimensional data (like COIL-20)"""
    model = LFVU_AE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        lfvu_k=lfvu_k,
        lfvu_weight=1.0,
        hidden_dims=[2048, 1024, 512, 256]  # Deeper for high-dim data
    )
    return model


# Training example
# if __name__ == "__main__":
#     # Example with synthetic data
#     batch_size = 64
#     input_dim = 100
#     latent_dim = 2

#     # Create synthetic data
#     X_train = torch.randn(1000, input_dim)
#     X_val = torch.randn(200, input_dim)

#     # Create data loaders
#     train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(X_train),
#         batch_size=batch_size,
#         shuffle=True
#     )

#     val_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(X_val),
#         batch_size=batch_size
#     )

#     # Create model
#     model = create_lfvu_ae(input_dim, latent_dim, lfvu_k=10)
#     trainer = LFVU_AETrainer(model, lr=1e-3, device='cuda')

#     # Train
#     train_history, val_history = trainer.train(
#         train_loader, val_loader,
#         epochs=100,
#         lfvu_start_epoch=50
#     )

#     print("Training complete!")