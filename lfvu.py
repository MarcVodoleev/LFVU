import numpy as np
import torch
import torch.nn as nn
import typing as tp
from sklearn.neighbors import NearestNeighbors

def compute_neighbors(X: torch.tensor | np.NDArray,
					  k: int) -> dict[str, torch.tensor]:
	"""Finds k nearest neighbors in the initial space"""
	if isinstance(X, torch.tensor):
		X_np = X.detach().cpu().numpy()
	else:
		X_np = X

	knn = NearestNeighbors(n_neighbors=k+1)  # +1 to later exclude THE point
	knn.fit(X_np)
	distances, indices = knn.kneighbors(X_np)

	# Excluding THE point
	indices = indices[:, 1:self.k+1]
	distances = distances[:, 1:self.k+1]

	# Saving as torch tensor
	indices_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
	distances_tensor = torch.tensor(distances, device=self.device, dtype=torch.float32)

	return {'indices': indices_tensor, 'distances': distances_tensor}

def compute_lfvu(X: torch.tensor | np.NDArray,
				 Y: torch.tensor | np.NDArray,
				 N: int,
				 k: int) -> tuple(torch.tensor):
	"""
	Computes LFVU and gradients
	
	Returns:
	--------
	loss : torch tensor
	    Mean LFVU
	LFVU_i : torch tensor
	    LFVU for each point
	""	
	# Computing original distances
	neighbor_indices, d_original = compute_neigbors(X, k)
	
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
	
	# Computing LFVU
	numerator = torch.sum((d_norm - r_norm) ** 2, dim=1)
	denominator = torch.sum((d_norm - d_norm.mean(dim=1, keepdim=True)) ** 2, dim=1) + 1e-8
	LFVU_i = numerator / denominator
	loss = torch.mean(LFVU_i)
	
	return loss, LFVU_i
