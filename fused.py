import torch
from sklearn.cluster import SpectralClustering
import sys
import base64

def center_matrix(n, device=None):
    I = torch.eye(n, device=device)
    return I - torch.ones((n, n), device=device) / n

def compute_trace_score(S, Y):
    n = S.shape[0]
    H = center_matrix(n, device=S.device)
    return torch.trace(S @ H @ Y @ Y.T @ H)

def labels_to_onehot(labels, cluster_number, device='cuda'):
    n = len(labels)
    Y = torch.zeros((n, cluster_number), device=device)
    Y[torch.arange(n), torch.tensor(labels, dtype=torch.long, device=device)] = 1
    return Y

def spectral_clustering(S, cluster_number):
    if isinstance(S, torch.Tensor):
        S_numpy = S.detach().cpu().numpy()
    else:
        S_numpy = S
    labels = SpectralClustering(n_clusters=cluster_number, affinity='precomputed').fit_predict(S_numpy)
    return labels

def fuse_traced_softmax_iterative(S_view_list, cluster_number, lambda_=1.0, max_iter=10, tol=1e-4, device='cuda'):
    V = len(S_view_list)
    n = S_view_list[0].shape[0]
    
    S_view_list = [torch.tensor(S, dtype=torch.float32, device=device) if not isinstance(S, torch.Tensor) else S.to(device) for S in S_view_list]

    S_mean = sum(S_view_list) / V

    init_labels = spectral_clustering(S_mean, cluster_number)
    Y_softmax_onehot = labels_to_onehot(init_labels, cluster_number, device=device)

    mu_softmax_weights = torch.ones(V, device=device) / V
    Sfinal = S_mean.clone()

    for it in range(max_iter):
        trace_scores = torch.tensor([compute_trace_score(S_view_list[v], Y_softmax_onehot) for v in range(V)], device=device)
        mu_new = torch.softmax(trace_scores / lambda_, dim=0)

        S_new = sum(mu_new[v] * S_view_list[v] for v in range(V))

        new_labels = spectral_clustering(S_new, cluster_number)
        Y_new = labels_to_onehot(new_labels, cluster_number, device=device)

        if torch.norm(S_new - Sfinal, p='fro') < tol and torch.norm(mu_new - mu_softmax_weights) < tol:
            break

        Sfinal = S_new
        mu_softmax_weights = mu_new
        Y_softmax_onehot = Y_new

    return Sfinal, mu_softmax_weights, Y_softmax_onehot

