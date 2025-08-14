from readdata import load_multiview_data
import numpy as np
from structure import structure_admm_multiview
from complete import complete_admm_multiview
from gcn_bayes import optimize_gcn_with_optuna
import torch
import warnings
warnings.filterwarnings('ignore')
from adaptive import AlphaFusionEM
from fused import fuse_traced_softmax_iterative
from evaluate_cluster import evaluate_clustering

#----------------------------Parameter setting-------------------------#
mat_file = 'Yale.mat'
missing_rate = 0.9

#----------------------------Data loading-------------------------#
X_views, y, cluster_number, M_mask = load_multiview_data(mat_file, missing_rate=missing_rate, seed=42)

# ----------------------------Structure graph learning-------------------------#
lambda_ = 1e-1
S_structure_views, Z_structure_views = structure_admm_multiview(X_views, cluster_number, lambda_)

# ----------------------------Complete graph learning-------------------------#
lambda_complete = 1e-3
X_hat_views, Z_complete_views, W_complete_views, S_complete_views = complete_admm_multiview(
    X_views, M_mask, c=cluster_number, lambda_=lambda_complete,
    rho=1.0,
    max_iter=50, tol=1e-5, knn_k=30
)

# ----------------------------GCN-------------------------#
Z_s_all, Z_c_all = [], []
out_dim_fixed = 128 
for v in range(len(X_hat_views)):
    result = optimize_gcn_with_optuna(
        X_hat=X_hat_views[v],
        S_structure=S_structure_views[v],
        S_complete=S_complete_views[v],
        cluster_num=cluster_number,
        max_epochs=50,
        n_trials=10,
        out_dim_fixed=out_dim_fixed,
        device='cuda'
    )
    Z_s_all.append(result['Z_structure'].numpy())
    Z_c_all.append(result['Z_complete'].numpy())

# ----------------------------Adaptive fusion-------------------------#
em_solver = AlphaFusionEM(Z_s_all, Z_c_all, r_list=(1 - M_mask).mean(axis=0).tolist(),
                          n_clusters=cluster_number, lambda_val=0.1, max_iter=30, device='cuda')

S_fused_list, alpha_list, Y_list = em_solver.optimize()

# ----------------------------Consensus graph fusion-------------------------#
Sfinal, mu_softmax_weights, Y_softmax_onehot = fuse_traced_softmax_iterative(
    S_view_list=S_structure_views,
    cluster_number=cluster_number,
    lambda_=100.0,
    max_iter=50,
    device='cuda'
)

# ----------------------------Evaluation-------------------------#
acc, nmi, ari, f1, recall, precision, purity = evaluate_clustering(Sfinal, y, cluster_number, device='cuda')
print(f"ACC:      {acc:.4f}")
print(f"NMI:      {nmi:.4f}")
print(f"ARI:      {ari:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Recall:   {recall:.4f}")
print(f"Precision:{precision:.4f}")
print(f"Purity:   {purity:.4f}")
