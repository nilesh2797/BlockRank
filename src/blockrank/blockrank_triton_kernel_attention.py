"""
Kernelized BlockRank Attention with Full Mask Support

This implementation properly loads and applies attention masks in Triton kernels,
supporting left-padded sequences within blocks.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional
from typing import Optional, Callable
from typing_extensions import Unpack
try:
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.models.llama.modeling_llama import TransformersKwargs, repeat_kv
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TransformersKwargs = None

# ============================================================================
# Triton Kernel: Block 0 with Mask Support
# ============================================================================

@triton.jit
def _blockrank_block0_fwd_kernel_masked(
    Q, K, V, Mask, Out, M_log,  # pointers
    stride_qb, stride_qh, stride_qm, stride_qd,  # Q strides
    stride_kb, stride_kh, stride_km, stride_kd,  # K strides
    stride_vb, stride_vh, stride_vm, stride_vd,  # V strides
    stride_maskb, stride_maskh, stride_maskm1, stride_maskm2,  # Mask strides (B, 1, H, H) per block
    stride_ob, stride_oh, stride_om, stride_od,  # Out strides
    stride_mb, stride_mh, stride_mm,  # M_log strides
    sm_scale,
    N_HEADS: tl.constexpr,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Block 0 kernel with mask support."""
    # Program IDs
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // N_HEADS
    off_h = off_hz % N_HEADS

    # Offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Pointers to Q, K, V for this block
    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh

    Q_block = Q + q_offset
    K_block = K + k_offset
    V_block = V + v_offset

    # Mask offset for this batch, block 0
    # Mask shape: (B, 1, M, H, H) -> for block 0: (B, 1, 0, H, H)
    # We want: Mask[off_b, 0, 0, :, :]
    mask_offset = off_b * stride_maskb  # batch offset, head=0 (broadcast), block_idx=0

    # Load Q
    q_ptrs = Q_block + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    valid_q_mask = offs_m[:, None] < H
    q = tl.load(q_ptrs, mask=valid_q_mask, other=0.0)

    # Initialize accumulator and statistics
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    # Loop over K, V tiles
    for start_n in range(0, H, BLOCK_N):
        # Load K, V tiles
        k_ptrs = K_block + ((start_n + offs_n[:, None]) * stride_km + offs_d[None, :] * stride_kd)
        v_ptrs = V_block + ((start_n + offs_n[:, None]) * stride_vm + offs_d[None, :] * stride_vd)

        valid_kv_mask = (start_n + offs_n[:, None]) < H
        k = tl.load(k_ptrs, mask=valid_kv_mask, other=0.0)
        v = tl.load(v_ptrs, mask=valid_kv_mask, other=0.0)

        # Load mask tile
        # Mask[off_b, 0, 0, offs_m, start_n + offs_n]
        mask_ptrs = Mask + mask_offset + (offs_m[:, None] * stride_maskm1 + (start_n + offs_n[None, :]) * stride_maskm2)
        mask_tile = tl.load(mask_ptrs, mask=(offs_m[:, None] < H) & ((start_n + offs_n[None, :]) < H), other=float("-inf"))
        mask_tile = mask_tile.to(tl.float32)  # Cast to fp32 for consistency

        # Compute QK^T
        qk = tl.dot(q, tl.trans(k))

        # Apply mask: mask is additive (0 for valid, -inf for masked)
        # Scale qk first, then add mask
        qk = qk * qk_scale + mask_tile

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)

        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # Final normalization
    acc = acc / l_i[:, None]
    m_i = m_i + tl.math.log2(l_i)

    # Store output
    out_offset = off_b * stride_ob + off_h * stride_oh
    out_ptrs = Out + out_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < H)

    # Store log-sum-exp
    m_offset = off_b * stride_mb + off_h * stride_mh
    m_ptrs = M_log + m_offset + offs_m * stride_mm
    tl.store(m_ptrs, m_i, mask=offs_m < H)


# ============================================================================
# Triton Kernel: Middle Blocks with Mask Support
# ============================================================================

@triton.jit
def _blockrank_middle_fwd_kernel_masked(
    Q, K0, K_self, V0, V_self, Mask, Out, M_log,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_k0b, stride_k0h, stride_k0m, stride_k0d,
    stride_ksb, stride_ksh, stride_ksm, stride_ksd,
    stride_v0b, stride_v0h, stride_v0m, stride_v0d,
    stride_vsb, stride_vsh, stride_vsm, stride_vsd,
    stride_maskb, stride_maskh, stride_maskblk, stride_maskm1, stride_maskm2,  # Mask strides
    stride_ob, stride_oh, stride_om, stride_od,
    stride_mb, stride_mh, stride_mm,
    sm_scale,
    N_HEADS: tl.constexpr,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Middle blocks kernel with mask support. Processes all middle blocks in parallel."""
    # Program IDs
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    block_idx = tl.program_id(2)  # Which middle block (0 = block 1, 1 = block 2, etc.)
    off_b = off_hz // N_HEADS
    off_h = off_hz % N_HEADS

    # Actual block index is block_idx + 1 (since middle blocks are 1 to M-2)
    actual_block_idx = block_idx + 1

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Pointers to Q
    q_offset = off_b * stride_qb + off_h * stride_qh + actual_block_idx * H * stride_qm
    Q_block = Q + q_offset

    # Load Q
    q_ptrs = Q_block + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < H, other=0.0)

    # Initialize
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    # ========================================
    # Part 1: Attend to block 0
    # ========================================
    k0_offset = off_b * stride_k0b + off_h * stride_k0h
    v0_offset = off_b * stride_v0b + off_h * stride_v0h
    K0_block = K0 + k0_offset
    V0_block = V0 + v0_offset

    # Cross-block mask construction:
    # mask_cross[i, j] = min(block0_last_row[j], current_block_last_row[i])
    # - block0_last_row[j]: is key j in block 0 valid?
    # - current_block_last_row[i]: is query i in current block valid?

    # Load block 0's last row (valid keys in block 0)
    mask0_last_row_offset = off_b * stride_maskb + (H - 1) * stride_maskm1
    mask0_last_row_ptrs = Mask + mask0_last_row_offset + offs_n * stride_maskm2
    mask0_last_row = tl.load(mask0_last_row_ptrs, mask=offs_n < H, other=float("-inf"))  # (BLOCK_N,)
    mask0_last_row = mask0_last_row.to(tl.float32)  # Cast to fp32

    # Load current block's last row (valid queries in current block)
    mask_curr_last_row_offset = off_b * stride_maskb + actual_block_idx * stride_maskblk + (H - 1) * stride_maskm1
    mask_curr_last_row_ptrs = Mask + mask_curr_last_row_offset + offs_m * stride_maskm2
    mask_curr_last_row = tl.load(mask_curr_last_row_ptrs, mask=offs_m < H, other=float("-inf"))  # (BLOCK_M,)
    mask_curr_last_row = mask_curr_last_row.to(tl.float32)  # Cast to fp32

    for start_n in range(0, H, BLOCK_N):
        # Load K0, V0
        k_ptrs = K0_block + ((start_n + offs_n[:, None]) * stride_k0m + offs_d[None, :] * stride_k0d)
        v_ptrs = V0_block + ((start_n + offs_n[:, None]) * stride_v0m + offs_d[None, :] * stride_v0d)

        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < H, other=0.0)
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < H, other=0.0)

        # Construct cross-block mask tile
        # For keys at positions start_n:start_n+BLOCK_N
        mask0_last_row_tile_ptrs = Mask + mask0_last_row_offset + (start_n + offs_n) * stride_maskm2
        mask0_last_row_tile = tl.load(mask0_last_row_tile_ptrs, mask=(start_n + offs_n) < H, other=float("-inf"))  # (BLOCK_N,)
        mask0_last_row_tile = mask0_last_row_tile.to(tl.float32)  # Cast to fp32

        # Combine: mask_cross[i, j] = min(block0_last_row[j], curr_last_row[i])
        # Broadcast to (BLOCK_M, BLOCK_N)
        mask_cross_tile = tl.minimum(
            mask0_last_row_tile[None, :],     # (1, BLOCK_N) - block 0 keys
            mask_curr_last_row[:, None]       # (BLOCK_M, 1) - current block queries
        )  # Now returns fp32

        # Compute QK^T and apply mask
        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale + mask_cross_tile

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)

        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # ========================================
    # Part 2: Causal self-attention
    # ========================================
    ks_offset = off_b * stride_ksb + off_h * stride_ksh + actual_block_idx * H * stride_ksm
    vs_offset = off_b * stride_vsb + off_h * stride_vsh + actual_block_idx * H * stride_vsm
    K_self_block = K_self + ks_offset
    V_self_block = V_self + vs_offset

    # Mask for self-attention: Mask[off_b, 0, actual_block_idx, :, :]
    mask_self_offset = off_b * stride_maskb + actual_block_idx * stride_maskblk

    for start_n in range(0, H, BLOCK_N):
        # Load K_self, V_self
        k_ptrs = K_self_block + ((start_n + offs_n[:, None]) * stride_ksm + offs_d[None, :] * stride_ksd)
        v_ptrs = V_self_block + ((start_n + offs_n[:, None]) * stride_vsm + offs_d[None, :] * stride_vsd)

        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < H, other=0.0)
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < H, other=0.0)

        # Load mask tile for self-attention
        mask_self_ptrs = Mask + mask_self_offset + (offs_m[:, None] * stride_maskm1 + (start_n + offs_n[None, :]) * stride_maskm2)
        mask_self_tile = tl.load(mask_self_ptrs, mask=(offs_m[:, None] < H) & ((start_n + offs_n[None, :]) < H), other=float("-inf"))
        mask_self_tile = mask_self_tile.to(tl.float32)  # Cast to fp32 for consistency

        # Compute QK^T and apply mask
        qk = tl.dot(q, tl.trans(k))
        qk = qk * qk_scale + mask_self_tile

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)

        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    # Final normalization
    acc = acc / l_i[:, None]
    m_i = m_i + tl.math.log2(l_i)

    # Store output
    out_offset = off_b * stride_ob + off_h * stride_oh + actual_block_idx * H * stride_om
    out_ptrs = Out + out_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < H)

    # Store log-sum-exp
    m_offset = off_b * stride_mb + off_h * stride_mh + actual_block_idx * H * stride_mm
    m_ptrs = M_log + m_offset + offs_m * stride_mm
    tl.store(m_ptrs, m_i, mask=offs_m < H)


# ============================================================================
# Triton Kernel: Last Block with Mask Support
# ============================================================================

@triton.jit
def _blockrank_last_fwd_kernel_masked(
    Q, K, V, Mask, Out, M_log,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_km, stride_kd,
    stride_vb, stride_vh, stride_vm, stride_vd,
    stride_maskb, stride_maskh, stride_maskblk, stride_maskm1, stride_maskm2,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_mb, stride_mh, stride_mm,
    sm_scale,
    N_HEADS: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Last block kernel with mask support."""
    # Program IDs
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // N_HEADS
    off_h = off_hz % N_HEADS

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Pointers to Q (from last block)
    q_offset = off_b * stride_qb + off_h * stride_qh + (M - 1) * H * stride_qm
    Q_block = Q + q_offset

    # Load Q
    q_ptrs = Q_block + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < H, other=0.0)

    # Initialize
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504

    # K/V base pointers
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    K_all = K + k_offset
    V_all = V + v_offset

    # Mask base offset
    mask_base = off_b * stride_maskb

    # Load last block's last row once (valid queries in last block)
    mask_last_block_last_row_offset = mask_base + (M - 1) * stride_maskblk + (H - 1) * stride_maskm1
    mask_last_block_last_row_ptrs = Mask + mask_last_block_last_row_offset + offs_m * stride_maskm2
    mask_last_block_last_row = tl.load(mask_last_block_last_row_ptrs, mask=offs_m < H, other=float("-inf"))  # (BLOCK_M,)
    mask_last_block_last_row = mask_last_block_last_row.to(tl.float32)  # Cast to fp32

    # Loop over all blocks
    for block_m in range(M):
        # Starting position in the flattened sequence
        block_start = block_m * H

        # Mask offset for this block: Mask[off_b, 0, block_m, :, :]
        mask_block_offset = mask_base + block_m * stride_maskblk

        is_self_attn = (block_m == M - 1)

        # Loop over tiles within this block
        for start_n in range(0, H, BLOCK_N):
            abs_start_n = block_start + start_n

            # Load K, V tiles
            k_ptrs = K_all + ((abs_start_n + offs_n[:, None]) * stride_km + offs_d[None, :] * stride_kd)
            v_ptrs = V_all + ((abs_start_n + offs_n[:, None]) * stride_vm + offs_d[None, :] * stride_vd)

            k = tl.load(k_ptrs, mask=(abs_start_n + offs_n[:, None]) < (M * H), other=0.0)
            v = tl.load(v_ptrs, mask=(abs_start_n + offs_n[:, None]) < (M * H), other=0.0)

            # Load mask tile
            if is_self_attn:
                # Self-attention: use full 2D causal mask
                mask_ptrs = Mask + mask_block_offset + (offs_m[:, None] * stride_maskm1 + (start_n + offs_n[None, :]) * stride_maskm2)
                mask_tile = tl.load(mask_ptrs, mask=(offs_m[:, None] < H) & ((start_n + offs_n[None, :]) < H), other=float("-inf"))
                mask_tile = mask_tile.to(tl.float32)  # Cast to fp32 for consistency
            else:
                # Cross-block: combine block_m's last row with last block's last row
                # Load block_m's last row (valid keys in block_m)
                mask_blockm_last_row_offset = mask_block_offset + (H - 1) * stride_maskm1
                mask_blockm_last_row_ptrs = Mask + mask_blockm_last_row_offset + (start_n + offs_n) * stride_maskm2
                mask_blockm_last_row_tile = tl.load(mask_blockm_last_row_ptrs, mask=(start_n + offs_n) < H, other=float("-inf"))  # (BLOCK_N,)
                mask_blockm_last_row_tile = mask_blockm_last_row_tile.to(tl.float32)  # Cast to fp32

                # Combine: mask_cross[i, j] = min(blockm_last_row[j], last_block_last_row[i])
                mask_tile = tl.minimum(
                    mask_blockm_last_row_tile[None, :],     # (1, BLOCK_N) - block_m keys
                    mask_last_block_last_row[:, None]       # (BLOCK_M, 1) - last block queries
                )  # Now returns fp32

            # Compute QK^T and apply mask
            qk = tl.dot(q, tl.trans(k))
            qk = qk * qk_scale + mask_tile

            # Online softmax update
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_ij[:, None])
            alpha = tl.math.exp2(m_i - m_ij)

            l_ij = tl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_ij

    # Final normalization
    acc = acc / l_i[:, None]
    m_i = m_i + tl.math.log2(l_i)

    # Store output
    out_offset = off_b * stride_ob + off_h * stride_oh + (M - 1) * H * stride_om
    out_ptrs = Out + out_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < H)

    # Store log-sum-exp
    m_offset = off_b * stride_mb + off_h * stride_mh + (M - 1) * H * stride_mm
    m_ptrs = M_log + m_offset + offs_m * stride_mm
    tl.store(m_ptrs, m_i, mask=offs_m < H)


# ============================================================================
# Triton Kernel: Compute Last Block Attention Scores
# ============================================================================

@triton.jit
def _compute_last_block_scores_kernel(
    Q, K, Mask, S_out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_km, stride_kd,
    stride_maskb, stride_maskh, stride_maskblk, stride_maskm1, stride_maskm2,
    stride_sb, stride_sh, stride_sm, stride_sn,
    sm_scale,
    M: tl.constexpr,
    H: tl.constexpr,
    MH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERIES: tl.constexpr,
    BLOCK_M_PADDED: tl.constexpr,  # Padded version for tl.dot (multiple of 16)
    BLOCK_N: tl.constexpr,
):
    """
    Compute unnormalized attention scores (QK^T + mask) for last NUM_QUERIES positions of last block.

    Grid: (num_heads, batch_size)
    Each program computes scores for one head in one batch element.

    Note: BLOCK_M_PADDED is NUM_QUERIES rounded up to nearest multiple of 16 for tl.dot compatibility.
    """
    # Program IDs
    off_h = tl.program_id(0)
    off_b = tl.program_id(1)

    # Query offset: last NUM_QUERIES positions of sequence
    query_start = MH - NUM_QUERIES
    offs_m = tl.arange(0, BLOCK_M_PADDED)  # Use padded size for tl.dot compatibility
    offs_d = tl.arange(0, HEAD_DIM)

    # Load queries for last NUM_QUERIES positions (pad with zeros for extra rows)
    q_offset = off_b * stride_qb + off_h * stride_qh + query_start * stride_qm
    q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < NUM_QUERIES, other=0.0)  # (BLOCK_M_PADDED, HEAD_DIM)

    # Note: We use sm_scale directly (not multiplied by 1.44269504) to match eager implementation
    # The main kernels use log2-based softmax, but here we return unnormalized scores like eager

    # Key/value offset
    k_offset = off_b * stride_kb + off_h * stride_kh

    # Mask base
    mask_base = off_b * stride_maskb

    # Output offset
    s_offset = off_b * stride_sb + off_h * stride_sh

    # Loop over all key positions (across all blocks)
    for block_m in range(M):
        block_start = block_m * H
        mask_block_offset = mask_base + block_m * stride_maskblk

        is_last_block = (block_m == M - 1)

        # Process keys in tiles
        for start_n in range(0, H, BLOCK_N):
            abs_start_n = block_start + start_n
            offs_n = tl.arange(0, BLOCK_N)

            # Load keys
            k_ptrs = K + k_offset + ((abs_start_n + offs_n[:, None]) * stride_km + offs_d[None, :] * stride_kd)
            k = tl.load(k_ptrs, mask=(abs_start_n + offs_n[:, None]) < MH, other=0.0)  # (BLOCK_N, HEAD_DIM)

            # Compute QK^T (using sm_scale directly, like eager implementation)
            qk = tl.dot(q, tl.trans(k)) * sm_scale  # (BLOCK_M_PADDED, BLOCK_N)

            # Load and apply mask
            if is_last_block:
                # Self-attention: use full 2D mask for last NUM_QUERIES rows
                # Note: We load mask for actual rows (H - NUM_QUERIES to H-1), pad with -inf for extra rows
                mask_ptrs = Mask + mask_block_offset + ((H - NUM_QUERIES + offs_m[:, None]) * stride_maskm1 + (start_n + offs_n[None, :]) * stride_maskm2)
                mask_tile = tl.load(mask_ptrs, mask=(offs_m[:, None] < NUM_QUERIES) & ((start_n + offs_n[None, :]) < H), other=float("-inf"))
                mask_tile = mask_tile.to(tl.float32)  # Cast to fp32 for consistency
            else:
                # Cross-block: use broadcasted last rows/diagonal
                # Keys from block_m: load last row (indicates valid key positions)
                mask_blockm_last_row_ptrs = Mask + mask_block_offset + (H - 1) * stride_maskm1 + (start_n + offs_n) * stride_maskm2
                mask_blockm_last_row = tl.load(mask_blockm_last_row_ptrs, mask=(start_n + offs_n) < H, other=float("-inf"))
                mask_blockm_last_row = mask_blockm_last_row.to(tl.float32)  # Cast to fp32

                # Queries from last block: load diagonal elements (indicates valid query positions)
                # For each query position (H - NUM_QUERIES + offs_m), check if it's valid by loading mask[query_pos, query_pos]
                # The diagonal is 0 for valid (non-padded) positions and -inf for padded positions
                query_pos = H - NUM_QUERIES + offs_m
                mask_last_block_diag_offset = mask_base + (M - 1) * stride_maskblk
                mask_last_block_diag_ptrs = Mask + mask_last_block_diag_offset + query_pos * stride_maskm1 + query_pos * stride_maskm2
                mask_last_block_diag = tl.load(mask_last_block_diag_ptrs, mask=offs_m < NUM_QUERIES, other=float("-inf"))
                mask_last_block_diag = mask_last_block_diag.to(tl.float32)  # Cast to fp32

                # Combine: mask[i,j] = min(key_valid[j], query_valid[i])
                mask_tile = tl.minimum(
                    mask_blockm_last_row[None, :],           # KEY validity: (1, BLOCK_N)
                    mask_last_block_diag[:, None]            # QUERY validity: (BLOCK_M_PADDED, 1)
                )  # Now returns fp32

            # Apply mask
            qk = qk + mask_tile  # (BLOCK_M_PADDED, BLOCK_N)

            # Store scores (only for actual NUM_QUERIES rows, not padded rows)
            s_ptrs = S_out + s_offset + (offs_m[:, None] * stride_sm + (abs_start_n + offs_n[None, :]) * stride_sn)
            tl.store(s_ptrs, qk, mask=(offs_m[:, None] < NUM_QUERIES) & ((abs_start_n + offs_n[None, :]) < MH))


# ============================================================================
# Python Wrapper
# ============================================================================

def compute_last_block_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor,
    num_blocks: int,
    block_size: int,
    sm_scale: float,
    num_queries: int = 16,
) -> torch.Tensor:
    """
    Compute unnormalized attention scores for the last block using Triton kernel.

    This computes QK^T + mask for the last num_queries positions.
    Only used when return_last_block_attn_scores=True.

    Args:
        query: (B, N, M*H, D)
        key: (B, N, M*H, D)
        attention_mask: (B, 1, M, H, H)
        num_blocks: M
        block_size: H
        sm_scale: scaling factor
        num_queries: number of query positions to return (default 16, like eager)

    Returns:
        s_last: (B, N, num_queries, M*H) - unnormalized attention scores for last num_queries positions
    """
    B, N, MH, D = query.shape
    M, H = num_blocks, block_size

    # Allocate output tensor
    s_last = torch.empty(B, N, num_queries, MH, device=query.device, dtype=torch.float32)

    # Choose BLOCK_N based on sequence length
    BLOCK_N = 64 if MH <= 512 else 128

    # Pad num_queries to nearest multiple of 16 for tl.dot compatibility
    # Triton's matrix multiply requires aligned dimensions
    BLOCK_M_PADDED = ((num_queries + 15) // 16) * 16

    # Grid: (num_heads, batch_size)
    grid = (N, B)

    # Launch kernel
    _compute_last_block_scores_kernel[grid](
        query, key, attention_mask, s_last,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        attention_mask.stride(0), attention_mask.stride(1),
        attention_mask.stride(2),  # block
        attention_mask.stride(3), attention_mask.stride(4),  # query_pos, key_pos
        s_last.stride(0), s_last.stride(1), s_last.stride(2), s_last.stride(3),
        sm_scale,
        M=M,
        H=H,
        MH=MH,
        HEAD_DIM=D,
        NUM_QUERIES=num_queries,
        BLOCK_M_PADDED=BLOCK_M_PADDED,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    return s_last


def kernelized_blockrank_attention_forward_with_full_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    num_blocks: int,
    block_size: int,
    sm_scale: float,
    return_last_block_attn_scores: bool = False,
    num_last_queries: int = 16,
):
    """
    Kernelized BlockRank attention with full mask support.

    Args:
        query: (B, N, M*H, D)
        key: (B, N, M*H, D)
        value: (B, N, M*H, D)
        attention_mask: (B, 1, M, H, H) - Additive mask (0 for attend, -inf for mask)
        num_blocks: M
        block_size: H
        sm_scale: scaling factor
        return_last_block_attn_scores: if True, return unnormalized attention scores for last block
        num_last_queries: number of query positions to return scores for (default 16)

    Returns:
        output: (B, N, M*H, D)
        s_last: (B, N, num_last_queries, M*H) if return_last_block_attn_scores else None
    """
    B, N, MH, D = query.shape
    M, H = num_blocks, block_size

    assert MH == M * H
    assert attention_mask.shape == (B, 1, M, H, H), f"Expected mask shape (B, 1, M, H, H), got {attention_mask.shape}"

    output = torch.empty_like(query)
    m_log = torch.empty((B, N, MH), device=query.device, dtype=torch.float32)

    BLOCK_M = min(64, H)
    BLOCK_N = min(64, H)

    grid_m = triton.cdiv(H, BLOCK_M)
    grid = (grid_m, B * N)

    # ========================================
    # Launch Block 0 kernel
    # ========================================
    # Mask strides: (B, 1, M, H, H)
    # We need: batch_stride, head_stride(unused), query_stride, key_stride
    _blockrank_block0_fwd_kernel_masked[grid](
        query, key, value, attention_mask, output, m_log,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        attention_mask.stride(0), attention_mask.stride(1),  # batch, head
        attention_mask.stride(3), attention_mask.stride(4),  # query_pos, key_pos (within block)
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        m_log.stride(0), m_log.stride(1), m_log.stride(2),
        sm_scale,
        N_HEADS=N,
        H=H,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    if M == 1:
        return output

    # ========================================
    # Launch middle blocks kernel (FUSED - all blocks in parallel!)
    # ========================================
    if M > 2:
        K0 = key[:, :, :H, :]
        V0 = value[:, :, :H, :]

        # Launch all middle blocks in parallel with 3D grid
        # Grid: (grid_m, B * N, num_middle_blocks)
        num_middle_blocks = M - 2
        grid_middle = (grid_m, B * N, num_middle_blocks)

        # Mask strides: (B, 1, M, H, H)
        _blockrank_middle_fwd_kernel_masked[grid_middle](
            query, K0, key, V0, value, attention_mask, output, m_log,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            K0.stride(0), K0.stride(1), K0.stride(2), K0.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            V0.stride(0), V0.stride(1), V0.stride(2), V0.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            attention_mask.stride(0), attention_mask.stride(1),  # batch, head
            attention_mask.stride(2),  # block
            attention_mask.stride(3), attention_mask.stride(4),  # query_pos, key_pos
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            m_log.stride(0), m_log.stride(1), m_log.stride(2),
            sm_scale,
            N_HEADS=N,
            H=H,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )

    # ========================================
    # Launch last block kernel
    # ========================================
    _blockrank_last_fwd_kernel_masked[grid](
        query, key, value, attention_mask, output, m_log,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        attention_mask.stride(0), attention_mask.stride(1),  # batch, head
        attention_mask.stride(2),  # block
        attention_mask.stride(3), attention_mask.stride(4),  # query_pos, key_pos
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        m_log.stride(0), m_log.stride(1), m_log.stride(2),
        sm_scale,
        N_HEADS=N,
        M=M,
        H=H,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    # Optionally compute attention scores for last block
    if return_last_block_attn_scores:
        s_last = compute_last_block_attention_scores(
            query, key, attention_mask, M, H, sm_scale, num_last_queries
        )
        return output, s_last
    else:
        return output, None


class KernelizedBlockRankAttentionWithMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, attention_mask, num_blocks, block_size, sm_scale,
                return_last_block_attn_scores, num_last_queries):
        output, s_last = kernelized_blockrank_attention_forward_with_full_mask(
            query, key, value, attention_mask, num_blocks, block_size, sm_scale,
            return_last_block_attn_scores, num_last_queries
        )
        return output, s_last

    @staticmethod
    def backward(ctx, grad_output, grad_s_last):
        raise NotImplementedError("Backward pass not implemented yet")


def blockrank_attention_with_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    num_blocks: int,
    block_size: int,
    sm_scale: float,
    return_last_block_attn_scores: bool = False,
    num_last_queries: int = 16,
):
    """
    User-facing API for kernelized BlockRank attention with full mask support.

    Args:
        query: (B, N, M*H, D)
        key: (B, N, M*H, D)
        value: (B, N, M*H, D)
        attention_mask: (B, 1, M, H, H)
        num_blocks: M
        block_size: H
        sm_scale: scaling factor
        return_last_block_attn_scores: if True, return unnormalized attention scores
        num_last_queries: number of last query positions to return scores for (default 16)

    Returns:
        output: (B, N, M*H, D)
        s_last: (B, N, num_last_queries, M*H) if return_last_block_attn_scores else None
    """
    return KernelizedBlockRankAttentionWithMask.apply(
        query, key, value, attention_mask, num_blocks, block_size, sm_scale,
        return_last_block_attn_scores, num_last_queries
    )

def triton_blockrank_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """
    Triton-kernelized BlockRank attention implementation for Transformers.

    Implements the BlockRank attention pattern using Triton kernels where:
    - Block 0 attends causally to itself
    - Blocks 1..M-2 attend causally to self and fully to block 0
    - Block M-1 attends fully to all blocks and causally to itself

    Args:
        module: The attention module
        query: (B, N, M*H, D) Query tensor
        key: (B, Nk, M*H, D) Key tensor
        value: (B, Nk, M*H, D) Value tensor
        attention_mask: (B, 1, M, H, H) Additive mask (0 for allowed, -inf for masked)
                        Note: The mask is used to infer M and H, but masking is
                        handled by the kernel based on BlockRank pattern
        scaling: Attention scaling factor
        dropout: Dropout probability (not supported in forward-only kernel)

    Returns:
        attn_output: (B, M*H, N, D) Attention output
        attn_weights: None (kernelized version doesn't return weights)
    """
    B, N, MH, D = query.shape
    _, Nk, _, _ = key.shape

    assert attention_mask is not None, "BlockRank attention requires an attention mask"
    assert len(attention_mask.shape) == 5, "Attention mask must be 5D for BlockRank attention"
    _, _, M, H, _ = attention_mask.shape
    assert H == MH // M, f"Block size H={H} does not match MH // M = {MH // M}"

    # Repeat K/V heads for GQA/MQA so that key/value heads match query heads
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    # Note: dropout not supported in kernelized forward-only version
    if dropout > 0.0 and module.training:
        import warnings
        warnings.warn("Dropout is not supported in kernelized BlockRank attention, ignoring dropout parameter")

    # Check if we need to return attention scores
    # Support layer-specific configuration
    layers_to_return_scores = kwargs.get('layers_to_return_scores', None)

    if layers_to_return_scores is not None:
        # If specific layers are specified, only return scores for those layers
        layer_idx = getattr(module, 'layer_idx', None)
        if layer_idx is not None and layer_idx in layers_to_return_scores:
            return_last_block_attn_scores = True
        else:
            return_last_block_attn_scores = False
    else:
        # Default behavior: use the parameter directly
        return_last_block_attn_scores = kwargs.get('return_last_block_attn_scores', False)

    num_last_queries = kwargs.get('num_last_queries', 16)

    # Run kernelized attention
    # query, key, value are in (B, N, M*H, D) format
    attn_output, s_last = blockrank_attention_with_mask(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        num_blocks=M,
        block_size=H,
        sm_scale=scaling,
        return_last_block_attn_scores=return_last_block_attn_scores,
        num_last_queries=num_last_queries,
    )

    # attn_output shape: (B, N, M*H, D)
    # Transformers expects output as (B, M*H, N, D), so we transpose
    attn_output = attn_output.transpose(1, 2).contiguous()

    # Return attention scores if requested (same format as eager)
    return attn_output, s_last


def triton_blockrank_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Callable = None,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create BlockRank attention mask for Triton-kernelized attention.

    Converts a binary block attention mask (B, M, H) to a 5D causal block mask
    (B, 1, M, H, H) where each block has causal masking and padding is properly handled.

    Note: The kernelized implementation uses this mask primarily to infer M and H.
    The actual masking logic is handled by the Triton kernels based on BlockRank pattern.

    Args:
        batch_size: Batch size
        cache_position: Cache position tensor (not used, for interface compatibility)
        kv_length: Key-value length (not used, for interface compatibility)
        kv_offset: KV offset (not used, for interface compatibility)
        mask_function: Mask function (not used, for interface compatibility)
        attention_mask: (B, M, H) Binary mask where 1=valid, 0=padding
        dtype: Output dtype
        **kwargs: Additional arguments (may include model config)

    Returns:
        mask: (B, 1, M, H, H) Additive mask (0 for attend, -inf for mask)
    """
    assert attention_mask is not None, "attention_mask must be provided for BlockRank attention"
    B, M, H = attention_mask.shape

    # Convert to boolean: True=valid, False=padding
    mask = attention_mask.bool().view(B, 1, M, 1, H)  # (B, 1, M, 1, H)

    # Create causal mask for each block
    causal_mask = torch.tril(torch.ones(H, H, device=mask.device, dtype=torch.bool))  # (H, H)

    # Combine: valid tokens + causal constraint
    mask = mask & (causal_mask[None, None, None, :, :])  # (B, 1, M, H, H)

    # Convert to additive mask: 0 for attend, -inf for mask
    min_dtype = 0.7 * torch.finfo(dtype).min  # Use 0.7 to avoid overflow
    mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)

    return mask


def register_triton_blockrank_attention():
    """
    Register the Triton-kernelized BlockRank attention implementation with Transformers.

    This registers the kernelized version under the following names:
    - "triton_blockrank": Direct kernelized version
    - "triton_blockrank_compiled": torch.compiled version with default mode

    Usage:
        >>> from triton_blockrank_attention import register_triton_blockrank_attention
        >>> register_triton_blockrank_attention()
        >>>
        >>> # Then use in model config:
        >>> config.attn_implementation = "triton_blockrank"
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "Transformers is not available. Please install it with: "
            "pip install transformers"
        )

    # Register base kernelized version
    AttentionInterface.register("triton_blockrank", triton_blockrank_attention_forward)
    AttentionMaskInterface.register("triton_blockrank", triton_blockrank_attention_mask)

    # Register torch.compiled versions for additional optimization
    AttentionInterface.register("triton_blockrank_compiled", torch.compile(triton_blockrank_attention_forward, mode="default"))
    AttentionMaskInterface.register("triton_blockrank_compiled", triton_blockrank_attention_mask)