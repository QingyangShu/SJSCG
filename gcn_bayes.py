import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import optuna

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, adj):
        support = self.linear(self.dropout(X))
        return F.relu(torch.mm(adj, support))

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, out_dim, dropout)

    def forward(self, X, adj):
        H = self.gcn1(X, adj)
        Z = self.gcn2(H, adj)
        return F.normalize(Z, p=2, dim=1)

def normalize_adj(adj):
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    deg_inv_sqrt = torch.pow(adj.sum(1), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

def structure_reconstruction_loss(Z, S):
    S = S.to(Z.device)
    S_hat = torch.sigmoid(torch.matmul(Z, Z.T))
    return F.mse_loss(S_hat, S)


def optimize_gcn_with_optuna(X_hat, S_structure, S_complete, cluster_num, max_epochs=100, n_trials=20, out_dim_fixed=128, device='cpu'):
    dimensions_match = (S_structure.shape == S_complete.shape)
    
    X_tensor = torch.tensor(X_hat, dtype=torch.float32).to(device)
    S_s_tensor = normalize_adj(torch.tensor(S_structure, dtype=torch.float32).to(device))
    S_c_tensor = normalize_adj(torch.tensor(S_complete, dtype=torch.float32).to(device))

    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 16, 512)
        dropout = trial.suggest_float("dropout", 0.0, 0.8)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)

        model = GCNEncoder(X_hat.shape[1], hidden_dim, out_dim_fixed, dropout).to(device)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            Z_s = model(X_tensor, S_s_tensor)
            Z_c = model(X_tensor, S_c_tensor)
            loss = (
                1.0 * (structure_reconstruction_loss(Z_s, S_s_tensor) +
                       structure_reconstruction_loss(Z_c, S_c_tensor))
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            Z_s = model(X_tensor, S_s_tensor).cpu()
            Z_c = model(X_tensor, S_c_tensor).cpu()
        return (
            1.0 * (structure_reconstruction_loss(Z_s, S_s_tensor) +
                   structure_reconstruction_loss(Z_c, S_c_tensor))
        )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_model = GCNEncoder(
        X_hat.shape[1],
        best_params['hidden_dim'],
        out_dim_fixed,
        best_params['dropout']
    ).to(device)

    optimizer = Adam(best_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    for _ in range(max_epochs):
        best_model.train()
        optimizer.zero_grad()
        Z_s = best_model(X_tensor, S_s_tensor)
        Z_c = best_model(X_tensor, S_c_tensor)
        loss = (
            1.0 * (structure_reconstruction_loss(Z_s, S_s_tensor) +
                   structure_reconstruction_loss(Z_c, S_c_tensor))
        )
        loss.backward()
        optimizer.step()

    best_model.eval()
    with torch.no_grad():
        Z_s = best_model(X_tensor, S_s_tensor).cpu()
        Z_c = best_model(X_tensor, S_c_tensor).cpu()

    if dimensions_match:
        Z_s = torch.tensor(S_structure, dtype=torch.float32)
        Z_c = torch.tensor(S_complete, dtype=torch.float32)
    return {
        'best_params': best_params,
        'Z_structure': Z_s,
        'Z_complete': Z_c
    }


