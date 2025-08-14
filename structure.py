import numpy as np
from sklearn.metrics import pairwise_distances

def reciprocal_knn_graph(X, k=15):
    n = X.shape[0]
    dist = pairwise_distances(X)
    knn_indices = np.argsort(dist, axis=1)[:, 1:k+1]

    A = np.zeros((n, n))
    for i in range(n):
        for j in knn_indices[i]:
            if i in knn_indices[j]:
                A[i, j] = A[j, i] = 1

    A2 = A @ A
    A2[A2 > 0] = 1
    np.fill_diagonal(A2, 0)

    S = A + 0.5 * A2
    S = np.clip(S, 0, 1)
    return S

def structure_admm_multiview(X_views, c, lambda_, mu=1.0, max_iter=20, tol=1e-4):
    def soft_threshold(X, tau):
        return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

    S_structure_list, Z_structure_list = [], []
    for X in X_views:
        n = X.shape[0]
        S = reciprocal_knn_graph(X, k=15)
        H = S.copy()
        Y = np.zeros_like(S)

        for it in range(max_iter):
            D_vec = S.sum(axis=1)
            D = np.diag(D_vec)
            L = D - S
            eigvals, eigvecs = np.linalg.eigh(L)
            Z1 = eigvecs[:, :c]

            W = np.sum(Z1**2, axis=1, keepdims=True) + np.sum(Z1**2, axis=1) - 2 * (Z1 @ Z1.T)
            D_vec = np.sum(S, axis=1)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(D_vec + 1e-8))
            tilde_X = D_inv_sqrt @ X
            XTX = tilde_X @ tilde_X.T
            Q = H - (1/mu) * Y
            A = 2 * XTX + mu * np.eye(n)
            B = 2 * XTX - W + mu * Q
            S_new = np.linalg.solve(A, B)
            S_new = np.maximum((S_new + S_new.T) / 2, 0)

            H = soft_threshold(S_new + Y / mu, lambda_ / mu)
            Y = Y + mu * (S_new - H)

            if np.linalg.norm(S - S_new, ord='fro') < tol:
                break
            S = S_new

        S_structure_list.append(S)
        Z_structure_list.append(Z1)

    return S_structure_list, Z_structure_list
