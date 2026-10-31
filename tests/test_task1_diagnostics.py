import torch

from task1.diagnostics import fp16_rounding_fro


def test_fp16_rounding_fro_matches_measured_rounding_error():
    #a "trained" fp32 matrix stored as fp16: the estimate should match the actual storage error
    gen= torch.Generator().manual_seed(0)
    W= 0.02 * torch.randn(256, 512, generator=gen)

    measured= torch.linalg.norm(W.half().float() - W)
    estimate= fp16_rounding_fro(W.half())

    assert abs(estimate / measured - 1) < 0.05


def test_fp16_rounding_fro_handles_zero_and_subnormal_values():
    #fp16 subnormals (< 2^-14) all share ulp 2^-24; zeros must not get a large ulp from frexp's exponent 0
    W= torch.tensor([0.0, 2.0 ** -20, 1.0], dtype=torch.float16)

    expected= torch.sqrt(torch.tensor((2.0 ** -24) ** 2 * 2 + (2.0 ** -10) ** 2) / 12)

    assert torch.allclose(fp16_rounding_fro(W), expected)
