"""
Flash Attention Triton kernel optimized for RDNA4 (gfx1201).

Optimization history:
  v1: Basic Flash Attention with online softmax, exp2 trick
      → 1.15-1.37x faster than SDPA
  v2: FP16 P@V dot product (out_dtype=fp32) for D=128
      → Additional 10-17% for D=128 non-causal
      → Final: 1.3-2.2x faster than SDPA, up to 3.7x for D=64 causal

Key design:
  - Online softmax (Flash Attention style) with exp2 trick
  - FP16/BF16 P@V GEMM with fp32 accumulation (out_dtype=tl.float32)
    for D=128 to improve WMMA utilization
  - Triton autotune: searches BLOCK_M, BLOCK_N, num_warps, num_stages
  - Plain pointer arithmetic (no tl.make_block_ptr, removed in Triton 3.9)
  - kv_len may differ from qo_len; num_qo_heads must be a multiple of num_kv_heads

Performance (gfx1201, AMD Radeon AI PRO R9700):
  D=128 non-causal: 44-59 TFLOPS  vs SDPA 33-44 TFLOPS  (1.2-1.4x faster)
  D=128 causal:     22-35 TFLOPS  vs SDPA 12-14 TFLOPS  (1.7-2.2x faster)
  D=64  non-causal: 48-60 TFLOPS  vs SDPA 17-34 TFLOPS  (1.5-1.8x faster)
  D=64  causal:     40-54 TFLOPS  vs SDPA 14-15 TFLOPS  (2.8-3.7x faster)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    k_ptrs, v_ptrs, stride_kn, stride_vn,
    qk_scale, offs_m, offs_n, lo, hi, N_KV,
    BLOCK_N: tl.constexpr,
    MASKED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Streams KV tiles [lo, hi) through the online softmax; MASKED applies causal/tail masks."""
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_idx = start_n + offs_n
        k_valid = k_idx < N_KV

        # Load K tile (transposed: [BLOCK_DMODEL, BLOCK_N])
        if MASKED:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=k_valid[None, :], other=0.0)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn)

        # QK^T scaled attention scores: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * qk_scale

        # Apply causal mask + out-of-bounds padding mask
        if MASKED:
            if IS_CAUSAL:
                mask = (offs_m[:, None] >= k_idx[None, :]) & k_valid[None, :]
            else:
                mask = k_valid[None, :]
            qk = tl.where(mask, qk, float("-inf"))

        # Online softmax update (log2 space for exp2 efficiency)
        m_ij  = tl.max(qk, axis=1)          # row max of this tile
        m_new = tl.maximum(m_i, m_ij)       # updated running max

        # p = exp2(qk - m_new):  softmax weights for this tile
        p = tl.math.exp2(qk - m_new[:, None])   # [BLOCK_M, BLOCK_N]
        # alpha = exp2(m_i - m_new): correction factor for previous output
        alpha = tl.math.exp2(m_i - m_new)        # [BLOCK_M]

        # Load V tile [BLOCK_N, BLOCK_DMODEL]
        if MASKED:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=k_valid[:, None], other=0.0)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn)

        # Accumulate: O = O * alpha + P @ V
        # Using 16-bit P (cast from FP32 softmax) with FP32 accumulation via out_dtype
        # lets Triton emit v_wmma_f32_16x16x16_{f16,bf16} instructions.
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc=acc, out_dtype=tl.float32)

        # Update running normalizer
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new
    return acc, l_i, m_i


@triton.autotune(
    configs=[
        # D=128: FP16 P@V benefits from BN=32, nw=8
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'num_warps': 8, 'num_stages': 1}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'num_warps': 4, 'num_stages': 1}),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 32, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 1}),
        # D=64: larger BN, fewer warps
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':128, 'num_warps': 8, 'num_stages': 1}),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':128, 'num_warps': 4, 'num_stages': 1}),
    ],
    key=['N_Q', 'N_KV', 'BLOCK_DMODEL', 'IS_CAUSAL'],
)
@triton.jit
def flash_attn_fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, H_KV, N_Q, N_KV,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    Flash Attention forward kernel for RDNA (wave32 WMMA).
    Grid: (ceil(N_Q/BLOCK_M), H*Z)

    Uses FP16/BF16 P@V dot product with FP32 out_dtype for improved WMMA utilization.
    Causal masking is top-left aligned (query i sees keys 0..i), matching SDPA.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // (H // H_KV)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Tile pointers for this (batch, head): Q/O [BLOCK_M, D], K^T [D, BLOCK_N], V [BLOCK_N, D]
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_z * stride_kz + off_h_kv * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    v_ptrs = V + off_z * stride_vz + off_h_kv * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)                # running sum
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)   # output accumulator

    # Load Q tile (stays in registers for entire KV loop — Flash Attention key insight)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_Q, other=0.0)

    # Scale factor: use log2 trick so softmax uses exp2 instead of exp
    # exp(x * sm_scale) = exp2(x * sm_scale * log2(e))
    qk_scale = sm_scale * 1.44269504088896340736  # = sm_scale * log2(e)

    # Tiles entirely below the diagonal and inside N_KV need no masking; only the
    # diagonal (causal) and the ragged tail tiles pay for masked loads and tl.where.
    if IS_CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_KV)
        full_hi = tl.minimum(start_m * BLOCK_M, N_KV) // BLOCK_N * BLOCK_N
    else:
        hi = N_KV
        full_hi = N_KV // BLOCK_N * BLOCK_N

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, stride_kn, stride_vn,
                                    qk_scale, offs_m, offs_n, 0, full_hi, N_KV,
                                    BLOCK_N, False, IS_CAUSAL)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, stride_kn, stride_vn,
                                    qk_scale, offs_m, offs_n, full_hi, hi, N_KV,
                                    BLOCK_N, True, IS_CAUSAL)

    # Normalize output
    acc = acc / l_i[:, None]

    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_Q)


def flash_attn_triton(q, k, v, tensor_layout="HND", sm_scale=None, is_causal=False):
    """
    Flash Attention for AMD RDNA via Triton.

    Args:
        q: [batch, qo_heads, qo_len, head_dim] (HND) or [batch, qo_len, qo_heads, head_dim] (NHD)
        k, v: same layout with kv_heads / kv_len; qo_heads must be a multiple of kv_heads
        tensor_layout: "HND" or "NHD"
        sm_scale: attention scale. Default: 1/sqrt(head_dim)
        is_causal: causal masking (top-left aligned, as in SDPA)

    Returns:
        out: same shape, layout and dtype as q (fp16 or bf16)
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16) and k.dtype == q.dtype and v.dtype == q.dtype
    assert q.dim() == 4 and tensor_layout in ("HND", "NHD")

    if tensor_layout == "NHD":
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    B, H, N_Q, D = q.shape
    H_KV, N_KV = k.shape[1], k.shape[2]
    assert D in (64, 128), f"head_dim must be 64 or 128, got {D}"
    assert H % H_KV == 0, "num_qo_heads must be divisible by num_kv_heads"

    if sm_scale is None:
        sm_scale = D ** -0.5

    o = torch.empty_like(q)  # keeps q's strides, so NHD input gives NHD-contiguous output

    # Grid: (num_Q_tiles, H*B) — autotune picks BLOCK_M
    grid = lambda meta: (triton.cdiv(N_Q, meta['BLOCK_M']), H * B)

    flash_attn_fwd_kernel[grid](
        q, k, v, o,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        B, H, H_KV, N_Q, N_KV,
        BLOCK_DMODEL=D,
        IS_CAUSAL=is_causal,
    )
    return o.transpose(1, 2) if tensor_layout == "NHD" else o
