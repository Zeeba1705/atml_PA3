import os
import time

import torch

from task1.methods.lora import LoRALinear, add_adapters, lora_train


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
        #projection in fp32 even inside fp16 autocast, otherwise U_k^T B~ is only ~0 to fp16 precision
        with torch.autocast(device_type=self.lora_A.device.type, enabled=False):
            B_t, A_t= project_factors(self.lora_A, self.lora_B, self.U_k, self.V_k)

        return A_t, B_t


def compute_bases(model, targets, k):
    bases= {}

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and name.split(".")[-1] in targets:
            bases[name]= svd_bases(m.weight.detach(), k)

    return bases


def load_or_compute_bases(model, targets, k, path, model_id):
    #svd once, saved and reused on resume and by diagnostics. model_id guards against a smoke file being reused
    if os.path.exists(path):
        saved= torch.load(path, map_location="cpu")

        assert saved["model_id"] == model_id and saved["k"] == k, f"{path} was made for another model or k"

        return saved["bases"], saved["svd_seconds"]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0= time.time()

    bases= compute_bases(model, targets, k)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    svd_seconds= time.time() - t0

    cpu_bases= {name: (U.cpu(), V.cpu()) for name, (U, V) in bases.items()}

    tmp= path + ".tmp"
    torch.save({"model_id": model_id, "k": k, "bases": cpu_bases, "svd_seconds": svd_seconds}, tmp)
    os.replace(tmp, path)

    return cpu_bases, svd_seconds


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


def oplora_constraint_check(model):
    #max over layers of ||U_k^T B~|| and ||A~ V_k|| in fp32, should be ~0
    worst_B= 0.0
    worst_A= 0.0

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, OPLoRALinear):
                A_t, B_t= m.factors()

                worst_B= max(worst_B, torch.linalg.norm(m.U_k.T @ B_t).item())
                worst_A= max(worst_A, torch.linalg.norm(A_t @ m.V_k).item())

    return worst_B, worst_A


def oplora_train(model, params, microbatches, optimizer, scaler, device, use_fp16, max_grad_norm):
    #the update itself is the lora one: the projection lives inside OPLoRALinear.factors()
    return lora_train(
        model=model,
        params=params,
        microbatches=microbatches,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        use_fp16=use_fp16,
        max_grad_norm=max_grad_norm
    )
