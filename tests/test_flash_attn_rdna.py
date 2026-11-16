"""Correctness of the RDNA Triton flash-attention kernel and sageattn() dispatch against SDPA.

Run: uv run --with pytest pytest tests/test_flash_attn_rdna.py
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# (B, H, H_KV, Nq, Nkv, D, dtype, layout, causal)
CASES = [
    (2, 8, 8, 1000, 1000, 64, torch.float16, "HND", False),   # lengths not a multiple of any block
    (2, 8, 8, 1000, 1000, 128, torch.bfloat16, "NHD", True),
    (1, 8, 2, 513, 513, 128, torch.float16, "HND", True),     # GQA
    (2, 10, 10, 4096, 77, 64, torch.float16, "NHD", False),  # cross-attention
    (1, 4, 4, 300, 900, 128, torch.bfloat16, "HND", True),   # causal, kv longer than q
    (1, 4, 4, 900, 300, 64, torch.float16, "HND", True),     # causal, q longer than kv
]


@pytest.mark.parametrize("entry", ["flash_attn_triton", "sageattn"])
@pytest.mark.parametrize("B,H,H_KV,Nq,Nkv,D,dtype,layout,causal", CASES)
def test_matches_sdpa(entry, B, H, H_KV, Nq, Nkv, D, dtype, layout, causal):
    import sageattention
    from sageattention.triton.flash_attn_rdna4 import flash_attn_triton

    fn = flash_attn_triton if entry == "flash_attn_triton" else sageattention.sageattn
    torch.manual_seed(0)
    q = torch.randn(B, H, Nq, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H_KV, Nkv, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H_KV, Nkv, D, device="cuda", dtype=dtype)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=H_KV != H)

    to_layout = (lambda x: x.transpose(1, 2).contiguous()) if layout == "NHD" else (lambda x: x)
    o = fn(to_layout(q), to_layout(k), to_layout(v), tensor_layout=layout, is_causal=causal)
    if layout == "NHD":
        o = o.transpose(1, 2)

    assert o.shape == ref.shape and o.dtype == dtype
    cos = F.cosine_similarity(o.float().flatten(), ref.float().flatten(), dim=0).item()
    max_err = (o.float() - ref.float()).abs().max().item()
    assert cos > 0.9999 and max_err < 2e-2, f"cos={cos:.6f} max_err={max_err:.4f}"
