import torch

from task1.methods.lora import LoRALinear, add_adapters


def svd_bases(W0, k):
    #W0 = U S V^T in fp32, keep the leading k left/right singular vectors
    U, S, Vh= torch.linalg.svd(W0.float(), full_matrices=False)

    return U[:, :k].contiguous(), Vh[:k].T.contiguous()


def project_factors(A, B, U_k, V_k):
    #B~ = B - U_k (U_k^T B),  A~ = A - (A V_k) V_k^T, never building the dense (I - U_k U_k^T)
    B_t= B - U_k @ (U_k.T @ B)
    A_t= A - (A @ V_k) @ V_k.T

    return B_t, A_t


class OPLoRALinear(LoRALinear):
    def __init__(self, base, rank, alpha, dropout, U_k, V_k):
        super().__init__(base, rank, alpha, dropout)

        #frozen bases, saved separately in oplora_bases.pt
        self.register_buffer("U_k", U_k.to(base.weight.device), persistent=False)
        self.register_buffer("V_k", V_k.to(base.weight.device), persistent=False)

    def factors(self):
        B_t, A_t= project_factors(self.lora_A, self.lora_B, self.U_k, self.V_k)

        return A_t, B_t


def compute_bases(model, targets, k):
    bases= {}

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and name.split(".")[-1] in targets:
            bases[name]= svd_bases(m.weight.detach(), k)

    return bases


def add_oplora(model, targets, rank, alpha, dropout, seed, k, bases=None):
    #svd uses no randomness, so computing bases first keeps the adapter init identical to lora
    if bases is None:
        bases= compute_bases(model, targets, k)

    return add_adapters(
        model,
        targets,
        lambda name, base: OPLoRALinear(base, rank, alpha, dropout, bases[name][0], bases[name][1]),
        seed
    )
