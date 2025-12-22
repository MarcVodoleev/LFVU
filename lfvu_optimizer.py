import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import typing as tp
import umap

from scipy.signal import correlate
from scipy.spatial import distance_matrix
from scipy.stats import wasserstein_distance
from scipy.stats import pearsonr
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class LFVUOptimizer: # TODO: add dimension reducer to arguments
    def __init__(self,
                 X: tp.Union[np.array, torch.tensor],
                 target_dim: int = 2,
                 k: int = 10,
                 lr : float = 0.01,
                 epochs : int = 100,
                 init_method: str = 'tsne',
                 initial_emb: tp.Optional[tp.Union[np.array, torch.tensor]] = None,
                 reducer = None,
                 device: str = 'auto'):
        """
        LFVU dimension reduction algorithm

        Parameters:
        -----------
        X : numpy array or torch tensor
            Initial data (N x n)
        target_dim : int
            Target dimension
        k : int
            Number of nearest neighbors for the algorithm
        lr : float
            Learning rate
        epochs : int
            Number of epochs
        init_method : str
            Method for initialization: 'pca', 'random', 'tsne', 'umap'
        device : str
            'cpu' or 'cuda'
        """
        if device == 'auto':
          device = "cuda" if torch.cuda.is_available() else "cpu"
        assert device in ["cpu", "cuda"], "Incorrect device name"
        self.device = device
        print(f"Using device {self.device}")

        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32).to(device)
        else:
            self.X = X.to(device)

        self.N, self.orig_dim = self.X.shape
        self.k = k
        self.target_dim = target_dim
        self.lr = lr
        self.epochs = epochs

        assert self.k < self.N, "k should be less than total number of samples."

        self.Y = self._initialize_Y(init_method, initial_emb)
        self.Y_0 = self.Y.detach().cpu().numpy()
        self.Y.requires_grad_(True)

        self.neighbors_X = self._compute_neighbors_X()

        self.loss_history = []
        self.kl_history = []
        self.emb_history = {}
        self.lfvu_history = {}

    def _initialize_Y(self, method, initial_emb=None):
        """Initialization of low-dimensional representation"""
        init_time = time.time()
        if method == 'pca':
            X_np = self.X.cpu().numpy()
            pca = PCA(n_components=self.target_dim)
            Y_init = pca.fit_transform(X_np)
            return torch.tensor(Y_init, dtype=torch.float32,
                               device=self.device, requires_grad=True)

        elif method == 'random':
            return torch.randn(self.N, self.target_dim,
                              device=self.device, requires_grad=True)

        elif method == 'tsne':
            X_np = self.X.cpu().numpy()
            tsne = TSNE(n_components=self.target_dim, perplexity=30,
                       random_state=42)
            Y_init = tsne.fit_transform(X_np)
            return torch.tensor(Y_init, dtype=torch.float32,
                               device=self.device, requires_grad=True)

        elif method == 'umap':
            X_np = self.X.cpu().numpy()
            reducer = umap.UMAP(n_components=self.target_dim, random_state=42)
            Y_init = reducer.fit_transform(X_np)
            return torch.tensor(Y_init, dtype=torch.float32,
                                device=self.device, requires_grad=True)

        elif method == "from_emb":
            return torch.tensor(initial_emb, dtype=torch.float32,
                                device=self.device, requires_grad=True)

        else:
            raise ValueError(f"Неизвестный метод инициализации: {method}")

        print(f"Initialization time: {time.time() - init_time:4f} s")

    def _compute_neighbors_X(self):
        """Finds k nearest neighbors in the initial space"""
        X_np = self.X.detach().cpu().numpy()
        knn = NearestNeighbors(n_neighbors=self.k+1)  # +1 to later exclude THE point
        knn.fit(X_np)
        distances, indices = knn.kneighbors(X_np)

        # Excluding THE point
        indices = indices[:, 1:self.k+1]
        distances = distances[:, 1:self.k+1]

        # Saving as torch tensor
        indices_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
        distances_tensor = torch.tensor(distances, device=self.device, dtype=torch.float32)

        return {'indices': indices_tensor, 'distances': distances_tensor}

    def compute_lfvu(self, Y):
        """
        Computes LFVU and gradients

        Returns:
        --------
        loss : torch tensor
            Mean LFVU
        LFVU_i : torch tensor
            LFVU for each point
        """
        N = self.N
        k = self.k

        neighbor_indices = self.neighbors_X['indices']
        d_original = self.neighbors_X['distances']

        # Computing low-dimensional distances
        Y_expanded = Y.unsqueeze(1).expand(-1, k, -1)
        Y_neighbors = Y[neighbor_indices]
        r = torch.norm(Y_expanded - Y_neighbors, dim=2)

        # Normalization of distances
        d_mean = d_original.mean(dim=1, keepdim=True)
        r_mean = r.mean(dim=1, keepdim=True)

        d_std = d_original.std(dim=1, keepdim=True) + 1e-8
        r_std = r.std(dim=1, keepdim=True) + 1e-8

        d_norm = (d_original - d_mean) / d_std
        r_norm = (r - r_mean) / r_std

        numerator = torch.sum((d_norm - r_norm) ** 2, dim=1)
        denominator = torch.sum((d_norm - d_norm.mean(dim=1, keepdim=True)) ** 2, dim=1) + 1e-8
        LFVU_i = numerator / denominator

        # Средний LFVU
        loss = torch.mean(LFVU_i)

        return loss, LFVU_i

    def optimize(self, verbose=True):
        """Optimizes Y for LFVU minimization"""
        optimizer = torch.optim.Adam([self.Y], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10)
        total_start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            optimizer.zero_grad()

            loss, lfvu = self.compute_lfvu(self.Y)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            self.loss_history.append(loss.item())

            epoch_time = time.time() - epoch_start_time

            if verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch:4d}/{self.epochs}: "
                      f"Loss = {loss.item():.6f}, "
                      f"Time = {epoch_time:.4f}s")

            if (epoch % 50 == 0 or epoch == self.epochs - 1):
                self.emb_history[epoch] = self.Y.detach().cpu().numpy()
                self.lfvu_history[epoch] = lfvu.detach().cpu().numpy()

        print(f"Total training time = {time.time() - total_start_time:.4f}s")

        return self.Y.detach()

    def get_embedding(self):
        """Returns final embedding"""
        return self.Y.detach().cpu().numpy()

    def plot_loss_history(self):
        """Visualizes loss history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('LFVU Loss')
        plt.title('LFVU loss history')
        plt.grid(True, alpha=0.3)
        plt.show()
