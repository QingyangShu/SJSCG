import numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    precision_score,
    recall_score,
    f1_score
)
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def purity_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    contingency_matrix = np.zeros((np.max(y_pred) + 1, np.max(y_true) + 1))
    for i in range(len(y_true)):
        contingency_matrix[y_pred[i], y_true[i]] += 1
    return np.sum(np.max(contingency_matrix, axis=1)) / np.sum(contingency_matrix)

def evaluate_clustering(S, y_true, n_clusters, device='cpu'):
    if torch.is_tensor(S):
        S = S.detach().to(device).cpu().numpy()

    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                   assign_labels='kmeans', random_state=42)
    y_pred = spectral.fit_predict(S)

    acc = clustering_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    purity = purity_score(y_true, y_pred)

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    new_y_pred = np.zeros_like(y_pred)
    for i in range(len(row_ind)):
        new_y_pred[y_pred == row_ind[i]] = col_ind[i]

    precision = precision_score(y_true, new_y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, new_y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, new_y_pred, average='macro', zero_division=0)

    return acc, nmi, ari, f1, recall, precision, purity
