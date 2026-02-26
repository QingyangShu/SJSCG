import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

def initialize_hatX_mean(X, M):
    X_hat = X.copy()
    n, d = X.shape
    for j in range(d):
        idx = M[:, j] == 1
        mean_val = X[idx, j].mean() if np.any(idx) else 0.0
        X_hat[:, j] = np.where(M[:, j] == 1, X[:, j], mean_val)
    return X_hat

def initialize_Z2_W(X_hat, r):
    U, S, Vt = np.linalg.svd(X_hat, full_matrices=False)
    S_r = np.sqrt(S[:r])
    Z2 = U[:, :r] * S_r[np.newaxis, :]
    W = Vt[:r, :].T * S_r[np.newaxis, :]
    return Z2, W

def simple_knn_graph(X, k=30):
    n = X.shape[0]
    dist = pairwise_distances(X)
    knn_indices = np.argsort(dist, axis=1)[:, 1:k+1]

    A = np.zeros((n, n))
    for i in range(n):
        A[i, knn_indices[i]] = 1

    A2 = A @ A
    A2[A2 > 0] = 1
    np.fill_diagonal(A2, 0)
    S = A + 0.5 * A2
    return np.clip(S, 0, 1)

def complete_admm_multiview(X_views, M_mask, c, lambda_, rho=1.0,
                             max_iter=50, tol=1e-5, knn_k=30):
    X_hat_list, Z_list, W_list = [], [], []
    for v, X in enumerate(X_views):
        X = X.astype(np.float64)
        M = M_mask[:, v].reshape(-1, 1).repeat(X.shape[1], axis=1).astype(np.float64)
        X_hat = initialize_hatX_mean(X, M)
        Z2, W = initialize_Z2_W(X_hat, c)
        X_hat_list.append(X_hat)
        Z_list.append(Z2)
        W_list.append(W)

    Lambda_list = [np.zeros_like(X_hat_list[v]) for v in range(len(X_views))]
    S_complete_list = []

    for v, X in enumerate(X_views):
        M = M_mask[:, v].reshape(-1, 1).repeat(X.shape[1], axis=1).astype(np.float64)
        X_hat = X_hat_list[v]
        Z2 = Z_list[v]
        W = W_list[v]
        Lambda = Lambda_list[v]
        n = X.shape[0]

        for it in range(max_iter):
            S_graph = simple_knn_graph(X_hat, k=knn_k)
            D = np.diag(S_graph.sum(axis=1))
            L = D - S_graph

            A = rho * np.eye(n) + 2 * lambda_ * L
            rhs = rho * (Z2 @ W.T - Lambda) + 2 * (M * X)
            X_hat_new = np.linalg.solve(A, rhs)

            WWT_inv = np.linalg.inv(W.T @ W + 1e-8 * np.eye(c))
            Z2 = (X_hat_new + Lambda) @ W @ WWT_inv

            ZZT_inv = np.linalg.inv(Z2.T @ Z2 + 1e-8 * np.eye(c))
            W = (X_hat_new + Lambda).T @ Z2 @ ZZT_inv

            Lambda += X_hat_new - Z2 @ W.T
            diff = np.linalg.norm(X_hat_new - X_hat, ord='fro') / np.linalg.norm(X_hat, ord='fro')
            X_hat = X_hat_new
            if diff < tol:
                break

        X_hat_list[v] = X_hat
        Z_list[v] = Z2
        W_list[v] = W
        Lambda_list[v] = Lambda
        S_complete = simple_knn_graph(X_hat, k=knn_k)
        S_complete_list.append(S_complete)

    return X_hat_list, Z_list, W_list, S_complete_list




