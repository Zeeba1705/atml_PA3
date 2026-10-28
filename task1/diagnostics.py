import torch


def rho_k(delta, U_k):
    #||U_k U_k^T dW||_F = ||U_k^T dW||_F because U_k has orthonormal columns
    delta= delta.float()

    return torch.linalg.norm(U_k.T @ delta) ** 2 / torch.linalg.norm(delta) ** 2


def eta(delta, W0):
    return torch.linalg.norm(delta.float()) / torch.linalg.norm(W0.float())


def activation_ratio(delta, W0, z):
    #z: (n, d_in) cached down_proj inputs, one ratio per row
    z= z.float()

    num= torch.linalg.norm(z @ delta.float().T, dim=1)
    den= torch.linalg.norm(z @ W0.float().T, dim=1)

    return num / (den + 1e-8)
