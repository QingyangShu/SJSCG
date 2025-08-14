import torch
import torch.nn.functional as F

class AlphaFusionEM:
    def __init__(self, Z_s_list, Z_c_list, r_list, n_clusters, lambda_val=0.1, max_iter=10, device='cpu'):
        self.Z_s_list = [torch.tensor(z, dtype=torch.float32).to(device) for z in Z_s_list]
        self.Z_c_list = [torch.tensor(z, dtype=torch.float32).to(device) for z in Z_c_list]
        self.r_list = r_list
        self.n_clusters = n_clusters
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        self.device = device
        self.n_views = len(Z_s_list)
        self.alpha_list = [torch.nn.Parameter(torch.tensor(0.5, device=device, requires_grad=True)) for _ in range(self.n_views)]

    def spectral_embedding(self, S, k):
        S = S.detach().cpu()
        D = torch.diag(S.sum(dim=1))
        L = D - S
        eigvals, eigvecs = torch.linalg.eigh(L)
        idx = torch.argsort(eigvals)[:k]
        return F.normalize(eigvecs[:, idx], p=2, dim=1).to(self.device)

    def optimize(self):
        optimizer = torch.optim.Adam(self.alpha_list, lr=1e-2)

        Y_list = []
        for v in range(self.n_views):
            S_init = 0.5 * self.Z_s_list[v] + 0.5 * self.Z_c_list[v]
            S_sim = torch.matmul(S_init, S_init.T)
            Y_init = self.spectral_embedding(S_sim, self.n_clusters)
            Y_list.append(Y_init)

        for it in range(self.max_iter):
            optimizer.zero_grad()
            total_loss = 0
            S_fused_list = []

            for v in range(self.n_views):
                alpha_v = torch.clamp(self.alpha_list[v], 0.0, 1.0)
                S_v = alpha_v * self.Z_s_list[v] + (1 - alpha_v) * self.Z_c_list[v]
                S_fused_list.append(S_v)

                S_sim = torch.matmul(S_v, S_v.T)
                f_Sv = self.spectral_embedding(S_sim, self.n_clusters)

                loss_fit = F.mse_loss(f_Sv, Y_list[v])
                loss_sparse = self.lambda_val * self.r_list[v] * torch.abs(alpha_v)
                total_loss += loss_fit + loss_sparse

            total_loss.backward()
            optimizer.step()

            Y_list = []
            for v in range(self.n_views):
                alpha_v = torch.clamp(self.alpha_list[v], 0.0, 1.0)
                S_v = alpha_v * self.Z_s_list[v] + (1 - alpha_v) * self.Z_c_list[v]
                S_sim = torch.matmul(S_v, S_v.T)
                Y_v = self.spectral_embedding(S_sim, self.n_clusters)
                Y_list.append(Y_v)

        return S_fused_list, [a.item() for a in self.alpha_list], Y_list
