"""
BlockRank Attention Implementation for Transformers

This module implements the BlockRank attention mechanism from the BlockRank paper,
enables efficient attention over block-structured inputs for in-context ranking.

The attention pattern:
- Block 0 (instruction): Causal self-attention only
- Blocks 1..M-2 (documents): Attend to block 0 + causal self-attention
- Block M-1 (query): Attend to all previous blocks + causal self-attention
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable
from typing_extensions import Unpack
from torch import nn

from transformers import AttentionInterface, AttentionMaskInterface
from transformers.models.llama.modeling_llama import TransformersKwargs, repeat_kv
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Set to True only for debugging - adds validation overhead
_DEBUG = False

def check_left_padded_mask(attention_mask: torch.Tensor, verbose: bool = False):
    """
    Check if a (B, 1, M, H, H) attention mask is properly left-padded for each block.

    For each block on M axis:
    - All padding should be on the left
    - After the first valid token, there should be no padding tokens

    Args:
        attention_mask: torch.Tensor of shape (B, 1, M, H, H)
                       where -inf indicates masked (padding) and 0.0 indicates valid
        verbose: If True, return detailed violation information

    Returns:
        is_valid: bool or dict with details about violations
    """
    B, _, M, H, _ = attention_mask.shape

    # For causal masks, check the last row of each block (it sees all tokens in that block)
    # Shape: (B, 1, M, H) - the last row of each H×H block
    last_rows = attention_mask[:, :, :, -1, :]  # (B, 1, M, H)
    last_rows = last_rows.squeeze(1)  # (B, M, H)

    # Determine which positions are valid (0.0) vs masked (-inf)
    # valid_mask: True where token is valid, False where padded
    is_valid_token = (last_rows == 0.0)  # (B, M, H)
    is_padding = ~is_valid_token  # (B, M, H)

    # For proper left-padding:
    # After the first valid token (True), all subsequent tokens should be valid (True)
    # Equivalently: once we see False (padding) after True (valid), it's a violation

    # Compute cumulative OR from left to right
    # If a position has seen any valid token before (including itself), cumsum > 0
    cumsum_valid = torch.cumsum(is_valid_token.float(), dim=-1)  # (B, M, H)

    # A padding token is invalid if it appears after a valid token
    # i.e., is_padding=True AND cumsum_valid > 0 (but we need cumsum_valid from previous positions)
    cumsum_valid_shifted = torch.cat([
        torch.zeros(B, M, 1, device=attention_mask.device),
        cumsum_valid[:, :, :-1]
    ], dim=-1)  # (B, M, H)

    # Violation: padding appears after we've seen a valid token
    violations = is_padding & (cumsum_valid_shifted > 0)  # (B, M, H)

    # Check if each block has any violations
    has_violation = violations.any(dim=-1)  # (B, M)
    is_properly_left_padded = ~has_violation  # (B, M)

    if not verbose:
        return torch.all(~has_violation).item()

    # Find first violation position in each block (for debugging)
    violation_positions = torch.where(violations,
                                     torch.arange(H, device=attention_mask.device).view(1, 1, H),
                                     torch.tensor(H, device=attention_mask.device))  # (B, M, H)
    first_violation_pos = violation_positions.min(dim=-1)  # (B, M)

    # Count valid tokens in each block (from last row)
    num_valid_tokens = is_valid_token.sum(dim=-1)  # (B, M)

    return {
        'is_properly_left_padded': is_properly_left_padded,  # (B, M)
        'has_violation': has_violation,  # (B, M)
        'first_violation_pos': first_violation_pos.values,  # (B, M)
        'num_valid_tokens': num_valid_tokens,  # (B, M)
        'violations_per_block': violations.sum(dim=-1),  # (B, M) - count of violations
    }

def eager_blockrank_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    '''
    Eager BlockRank attention implementation.

    Implements the BlockRank attention pattern where:
    - Block 0 attends causally to itself
    - Blocks 1..M-2 attend causally to self and fully to block 0
    - Block M-1 attends fully to all blocks and causally to itself

    Args:
        module: The attention module
        query: (B, N, M*H, D) Query tensor
        key: (B, Nk, M*H, D) Key tensor
        value: (B, Nk, M*H, D) Value tensor
        attention_mask: (B, 1, M, H, H) Additive mask (0 for allowed, -inf for masked)
        scaling: Attention scaling factor
        dropout: Dropout probability

    Returns:
        attn_output: (B, M*H, N, D) Attention output
        attn_weights: Attention weights (for compatibility, returns None)
    '''
    B, N, MH, D = query.shape
    _, Nk, _, _ = key.shape
    assert attention_mask is not None, "BlockRank attention requires an attention mask"
    assert len(attention_mask.shape) == 5, "Attention mask must be 5D for BlockRank attention"
    _, _, M, H, _ = attention_mask.shape
    assert H == MH // M, f"Block size H={H} does not match MH // M = {MH // M}"

    # Repeat K/V heads for GQA/MQA so that key/value heads match query heads
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    # Reshape to block structure
    query = query.view(B, N, M, H, D)
    key = key.view(B, N, M, H, D)
    value = value.view(B, N, M, H, D)
    attention_mask = attention_mask.view(B, 1, M, H, H)  # redundant but explicit
    device, dtype = query.device, query.dtype

    # Validate mask is properly left-padded (only in debug mode)
    if _DEBUG:
        assert check_left_padded_mask(attention_mask, verbose=False), \
            "Attention mask is not properly left-padded per block"

    # Output tensor
    out = torch.empty((B, N, M, H, D), device=device, dtype=dtype)

    # Convenience views
    Q = query * scaling  # (B, N, M, H, D)
    K = key             # (B, N, M, H, D)
    V = value           # (B, N, M, H, D)

    # -----------------------------
    # Block 0: causal self-attention
    # -----------------------------
    Q0 = Q[:, :, 0]                 # (B, N, H, D)
    K0 = K[:, :, 0]                 # (B, N, H, D)
    V0 = V[:, :, 0]                 # (B, N, H, D)
    m0 = attention_mask[:, :, 0]    # (B, 1, H, H)

    s0 = torch.matmul(Q0, K0.transpose(-2, -1)) + m0
    p0 = F.softmax(s0, dim=-1, dtype=torch.float32).to(dtype)
    if dropout:
        p0 = F.dropout(p0, p=dropout, training=module.training)
    out[:, :, 0] = torch.matmul(p0, V0)

    # Early return if only one block
    if M == 1:
        out = out.view(B, N, MH, D).transpose(1, 2).contiguous()
        return out, None

    # ------------------------------------------------------------
    # Middle blocks (1..M-2): full to block 0 + causal to self
    # Compute in parallel by concatenating [K0 | Kself], [V0 | Vself]
    # ------------------------------------------------------------
    if M > 2:
        Q_mid = Q[:, :, 1:M-1]                # (B, N, M-2, H, D)
        K_self = K[:, :, 1:M-1]               # (B, N, M-2, H, D)
        V_self = V[:, :, 1:M-1]               # (B, N, M-2, H, D)

        # Repeat block 0 K/V for each middle block
        K0_rep = K0.unsqueeze(2).expand(B, N, M-2, H, D)  # (B, N, M-2, H, D)
        V0_rep = V0.unsqueeze(2).expand(B, N, M-2, H, D)  # (B, N, M-2, H, D)

        # Concatenate: [K0 | Kself]
        K_mid = torch.cat([K0_rep, K_self], dim=-2)       # (B, N, M-2, 2H, D)
        V_mid = torch.cat([V0_rep, V_self], dim=-2)       # (B, N, M-2, 2H, D)

        # Compute attention scores
        s_mid = torch.matmul(Q_mid, K_mid.transpose(-2, -1))  # (B, N, M-2, H, 2H)

        # Build concatenated mask:
        # - first H columns: broadcast "last valid" row from block 0
        # - next H columns: per-block causal self mask
        mask_first_cols = attention_mask[:, :, 0, -1, :]                     # (B, 1, H)
        mask_first = mask_first_cols.unsqueeze(2).unsqueeze(2)               # (B, 1, 1, 1, H)
        mask_first = mask_first.expand(B, 1, M-2, H, H)                      # (B, 1, M-2, H, H)
        mask_self = attention_mask[:, :, 1:M-1]                              # (B, 1, M-2, H, H)

        # Combine: take minimum (more restrictive) of block 0 mask and self mask for first H cols
        mask_first = torch.minimum(mask_first, mask_self[:, :, :, -1, :, None])  # (B, 1, M-2, H, H)
        m_mid = torch.cat([mask_first, mask_self], dim=-1)                   # (B, 1, M-2, H, 2H)

        s_mid = s_mid + m_mid
        p_mid = F.softmax(s_mid, dim=-1, dtype=torch.float32).to(dtype)
        if dropout:
            p_mid = F.dropout(p_mid, p=dropout, training=module.training)
        out[:, :, 1:M-1] = torch.matmul(p_mid, V_mid)

    # ------------------------------------------------------------
    # Last block (M-1): full to all blocks, causal to self
    # Concatenate K/V across all blocks
    # ------------------------------------------------------------
    Q_last = Q[:, :, M-1]                                        # (B, N, H, D)
    K_all = K.reshape(B, N, M * H, D)                            # (B, N, M*H, D)
    V_all = V.reshape(B, N, M * H, D)                            # (B, N, M*H, D)

    # Mask for other blocks (0..M-2): take last row and broadcast over query rows
    mask_others = attention_mask[:, :, :M-1, -1, :]              # (B, 1, M-1, H)
    mask_others = mask_others.reshape(B, 1, (M - 1) * H)         # (B, 1, (M-1)*H)
    mask_others = mask_others.unsqueeze(-2).expand(B, 1, H, (M - 1) * H)  # (B, 1, H, (M-1)*H)
    mask_self_last = attention_mask[:, :, M-1]                   # (B, 1, H, H)

    # Combine: minimum of other blocks mask and self mask
    mask_others = torch.minimum(mask_others, mask_self_last[:, :, -1, :, None])
    m_last = torch.cat([mask_others, mask_self_last], dim=-1)    # (B, 1, H, M*H)

    s_last = torch.matmul(Q_last, K_all.transpose(-2, -1)) + m_last  # (B, N, H, M*H)
    p_last = F.softmax(s_last, dim=-1, dtype=torch.float32).to(dtype)
    if dropout:
        p_last = F.dropout(p_last, p=dropout, training=module.training)
    out[:, :, M-1] = torch.matmul(p_last, V_all)

    # Reshape output to expected format
    out = out.view(B, N, MH, D).transpose(1, 2).contiguous()  # (B, M*H, N, D)

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

    if return_last_block_attn_scores:
        s_last = s_last[:, :, -num_last_queries:] if s_last.size(-2) >= num_last_queries else s_last  # (B, N, num_last_queries, M*H)
    else:
        s_last = None
    return out, s_last

def eager_blockrank_attention_mask(
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
    Create BlockRank attention mask for eager attention.

    Converts a binary block attention mask (B, M, H) to a 5D causal block mask
    (B, 1, M, H, H) where each block has causal masking and padding is properly handled.

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
    assert attention_mask is not None, "attention_mask must be provided for BlockRank eager attention"
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

def flex_blockrank_attention_mask(
    batch_size: int,
    cache_position: torch.Tensor,
    kv_length: int,
    kv_offset: int = 0,
    mask_function: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Mask interface for flex_blockrank attention.
    
    Creates a BlockMask for efficient sparse attention computation.
    This is called once to set up the mask, avoiding overhead in the forward pass.
    
    Args:
        attention_mask: (B, M, H) binary mask where 1=valid, 0=padding
    Returns:
        BlockMask object that will be used by flex_attention
    """
    assert attention_mask is not None, "attention_mask must be provided for BlockRank flex attention"
    
    # Extract dimensions from attention_mask
    # attention_mask shape: (B, M, H) where M=num_blocks, H=block_size
    B, M, H = attention_mask.shape
    MH = M * H  # Total sequence length
    
    # Store as bool tensor for efficient lookup
    block_valid_mask = attention_mask.bool()
    
    # Define BlockRank mask function
    # This captures block_valid_mask, M, H in the closure
    def blockrank_mask_fn(b, h, q_idx, kv_idx):
        """
        BlockRank attention mask logic.
        
        Returns True if attention from q_idx to kv_idx is allowed.
        
        BlockRank pattern:
        - Block 0: causal self-attention only
        - Blocks 1..M-2 (middle): attend to block 0 + causal self-attention
        - Block M-1 (last): attend to all previous blocks + causal self-attention
        
        Note: Must avoid control flow on tensor values for flex_attention tracing.
        """
        # Determine which block each index belongs to
        q_block = q_idx // H
        kv_block = kv_idx // H
        
        # Position within blocks
        q_pos = q_idx % H
        kv_pos = kv_idx % H
        
        # Check if tokens are valid (not padding)
        q_valid = block_valid_mask[b, q_block, q_pos]
        kv_valid = block_valid_mask[b, kv_block, kv_pos]
        both_valid = q_valid & kv_valid
        
        # Causal constraint
        causal = q_pos >= kv_pos
        
        # Block classifications
        same_block = q_block == kv_block
        is_block_0 = q_block == 0
        is_last_block = q_block == (M - 1)
        is_middle_block = ~is_block_0 & ~is_last_block
        kv_is_block_0 = kv_block == 0
        
        # BlockRank attention patterns (using boolean logic, no control flow):
        # - Block 0: only attends to itself causally
        #   -> same_block & causal
        block_0_pattern = is_block_0 & same_block & causal
        
        # - Middle blocks (1..M-2): attend to block 0 OR causal self-attention
        #   -> (kv_is_block_0 | same_block) & causal_when_same_block
        #   -> (kv_is_block_0 & ~same_block) | (same_block & causal)
        middle_pattern = is_middle_block & ((kv_is_block_0 & ~same_block) | (same_block & causal))
        
        # - Last block: attends to all previous blocks causally
        #   -> If same block: causal
        #   -> If different block (and kv_idx < M*H): True
        #   -> (same_block & causal) | (~same_block & kv_block < q_block)
        last_pattern = is_last_block & ((same_block & causal) | (~same_block & (kv_block < q_block)))
        
        # Combine all patterns with validity check
        result = both_valid & (block_0_pattern | middle_pattern | last_pattern)
        
        return result
    
    # Create BlockMask once here (expensive operation, should be cached)
    # B, H are batch and head dimensions - pattern varies per batch but not per head
    block_mask = create_block_mask(
        blockrank_mask_fn,
        B=B,  # Need batch-specific patterns due to padding
        H=None,  # Pattern is same across all heads (will broadcast)
        Q_LEN=MH,
        KV_LEN=MH,
        device=attention_mask.device,
    )
    
    return block_mask

def flex_blockrank_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],  # This will be the BlockMask now
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    '''
    Flex Attention BlockRank implementation
    
    Functionally identical to eager_blockrank_attention_forward but uses PyTorch's
    flex_attention API with pre-computed BlockMask for efficient sparse computation.

    Shapes:
      query: (B, N, M*H, D) - already in correct format for flex_attention
      key:   (B, Nk, M*H, D)
      value: (B, Nk, M*H, D)
      attention_mask: BlockMask object created by flex_blockrank_attention_mask
    
    Note: flex_attention expects (B, H, S, D) format which matches our (B, N, M*H, D)
    
    Semantics:
      - Block 0 attends causally to itself.
      - Blocks 1..M-2 attend causally to self and fully to block 0.
      - Block M-1 attends fully to all blocks and causally to itself.
    '''
    B, N, MH, D = query.shape
    
    # Repeat K/V heads for GQA/MQA so that key/value heads match query heads
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    # attention_mask is now a BlockMask object created in the mask interface
    block_mask = attention_mask
    
    # query, key, value are already in (B, N, M*H, D) format
    # This is exactly what flex_attention expects: (B, H, S, D)
    # No transpose needed!
    
    # Apply flex attention with pre-computed BlockMask
    attn_output = flex_attention(
        query,
        key, 
        value,
        block_mask=block_mask,
        scale=scaling,
        enable_gqa=False,  # We already handled GQA via repeat_kv
    )
    
    # attn_output shape: (B, N, M*H, D)
    # Transformers expects output as (B, M*H, N, D), so we transpose
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    return attn_output, None  # flex_attention doesn't return attention weights

def sdpa_blockrank_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    '''
    Eager BlockRank attention implementation.

    Implements the BlockRank attention pattern where:
    - Block 0 attends causally to itself
    - Blocks 1..M-2 attend causally to self and fully to block 0
    - Block M-1 attends fully to all blocks and causally to itself

    Args:
        module: The attention module
        query: (B, N, M*H, D) Query tensor
        key: (B, Nk, M*H, D) Key tensor
        value: (B, Nk, M*H, D) Value tensor
        attention_mask: (B, 1, M, H, H) Additive mask (0 for allowed, -inf for masked)
        scaling: Attention scaling factor
        dropout: Dropout probability

    Returns:
        attn_output: (B, M*H, N, D) Attention output
        attn_weights: Attention weights (for compatibility, returns None)
    '''
    B, N, MH, D = query.shape
    _, Nk, _, _ = key.shape
    assert attention_mask is not None, "BlockRank attention requires an attention mask"
    assert len(attention_mask.shape) == 5, "Attention mask must be 5D for BlockRank attention"
    _, _, M, H, _ = attention_mask.shape
    assert H == MH // M, f"Block size H={H} does not match MH // M = {MH // M}"
    
    # # Convert attention mask to boolean if it has additive format (0.0 for attend, -inf for mask)
    # if attention_mask.dtype != torch.bool and attention_mask.max() < 1e-6:
    #     # Mask is in additive format: 0.0 = attend, -inf = mask
    #     # Convert to boolean: True = attend, False = mask
    #     attention_mask = (attention_mask > -1.0)
    

    # Repeat K/V heads for GQA/MQA so that key/value heads match query heads
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    # Reshape to block structure
    query = query.view(B, N, M, H, D)
    key = key.view(B, N, M, H, D)
    value = value.view(B, N, M, H, D)
    attention_mask = attention_mask.view(B, 1, M, H, H)  # redundant but explicit
    device, dtype = query.device, query.dtype

    # Validate mask is properly left-padded (only in debug mode)
    if _DEBUG:
        assert check_left_padded_mask(attention_mask, verbose=False), \
            "Attention mask is not properly left-padded per block"

    # Output tensor
    out = torch.empty((B, N, M, H, D), device=device, dtype=dtype)

    # Convenience views
    Q = query  # (B, N, M, H, D)
    K = key             # (B, N, M, H, D)
    V = value           # (B, N, M, H, D)

    # -----------------------------
    # Block 0: causal self-attention
    # -----------------------------
    Q0 = Q[:, :, 0]                 # (B, N, H, D)
    K0 = K[:, :, 0]                 # (B, N, H, D)
    V0 = V[:, :, 0]                 # (B, N, H, D)
    m0 = attention_mask[:, :, 0]    # (B, 1, H, H)

    out[:, :, 0] = F.scaled_dot_product_attention(
        Q0, K0, V0,
        attn_mask=m0,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
    )

    # Early return if only one block
    if M == 1:
        out = out.view(B, N, MH, D).transpose(1, 2).contiguous()
        return out, None

    # ------------------------------------------------------------
    # Middle blocks (1..M-2): full to block 0 + causal to self
    # Compute in parallel by concatenating [K0 | Kself], [V0 | Vself]
    # ------------------------------------------------------------
    if M > 2:
        Q_mid = Q[:, :, 1:M-1]                # (B, N, M-2, H, D)
        K_self = K[:, :, 1:M-1]               # (B, N, M-2, H, D)
        V_self = V[:, :, 1:M-1]               # (B, N, M-2, H, D)

        # Repeat block 0 K/V for each middle block
        K0_rep = K0.unsqueeze(2).expand(B, N, M-2, H, D)  # (B, N, M-2, H, D)
        V0_rep = V0.unsqueeze(2).expand(B, N, M-2, H, D)  # (B, N, M-2, H, D)

        # Concatenate: [K0 | Kself]
        K_mid = torch.cat([K0_rep, K_self], dim=-2)       # (B, N, M-2, 2H, D)
        V_mid = torch.cat([V0_rep, V_self], dim=-2)       # (B, N, M-2, 2H, D)

        # Build concatenated mask:
        # - first H columns: broadcast "last valid" row from block 0
        # - next H columns: per-block causal self mask
        mask_first_cols = attention_mask[:, :, 0, -1, :]                     # (B, 1, H)
        mask_first = mask_first_cols.unsqueeze(2).unsqueeze(2)               # (B, 1, 1, 1, H)
        mask_first = mask_first.expand(B, 1, M-2, H, H)                      # (B, 1, M-2, H, H)
        mask_self = attention_mask[:, :, 1:M-1]                              # (B, 1, M-2, H, H)

        # Combine: take minimum (more restrictive) of block 0 mask and self mask for first H cols
        mask_first = torch.minimum(mask_first, mask_self[:, :, :, -1, :, None])  # (B, 1, M-2, H, H)
        m_mid = torch.cat([mask_first, mask_self], dim=-1)                   # (B, 1, M-2, H, 2H)

        # Compute attention using SDPA
        out[:, :, 1:M-1] = F.scaled_dot_product_attention(
            Q_mid, K_mid, V_mid,
            attn_mask=m_mid,
            dropout_p=dropout if module.training else 0.0,
            scale=scaling,
        )
    
    # ------------------------------------------------------------
    # Last block (M-1): full to all blocks, causal to self
    # Concatenate K/V across all blocks
    # ------------------------------------------------------------
    Q_last = Q[:, :, M-1]                                        # (B, N, H, D)
    K_all = K.reshape(B, N, M * H, D)                            # (B, N, M*H, D)
    V_all = V.reshape(B, N, M * H, D)                            # (B, N, M*H, D)

    # Mask for other blocks (0..M-2): take last row and broadcast over query rows
    mask_others = attention_mask[:, :, :M-1, -1, :]              # (B, 1, M-1, H)
    mask_others = mask_others.reshape(B, 1, (M - 1) * H)         # (B, 1, (M-1)*H)
    mask_others = mask_others.unsqueeze(-2).expand(B, 1, H, (M - 1) * H)  # (B, 1, H, (M-1)*H)
    mask_self_last = attention_mask[:, :, M-1]                   # (B, 1, H, H)

    # Combine: minimum of other blocks mask and self mask
    mask_others = torch.minimum(mask_others, mask_self_last[:, :, -1, :, None])
    m_last = torch.cat([mask_others, mask_self_last], dim=-1)    # (B, 1, H, M*H)

    out[:, :, M-1] = F.scaled_dot_product_attention(
        Q_last, K_all, V_all,
        attn_mask=m_last,
        dropout_p=dropout if module.training else 0.0,
        scale=scaling,
    )
    
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

    s_last = None
    if return_last_block_attn_scores:
        # Compute attention weights for last num_last_queries tokens only (for compatibility)
        Q_last = Q_last[:, :, -num_last_queries:] if Q_last.size(-2) >= num_last_queries else Q_last  # (B, N, num_last_queries, D) or less
        s_last = torch.matmul(Q_last, K_all.transpose(-2, -1))  # (B, N, num_last_queries, M*H)
        m_last = m_last[:, :, -num_last_queries:] if m_last.size(-2) >= num_last_queries else m_last  # (B, 1, num_last_queries, M*H)
        s_last = s_last + m_last

    # Reshape output to expected format
    out = out.view(B, N, MH, D).transpose(1, 2).contiguous()  # (B, M*H, N, D)

    return out, s_last  # Return last block's attention weights

def register_blockrank_attention():
    # Register the BlockRank attention implementation with Transformers
    for mode in ["default", "max-autotune", 'eager']:
        AttentionInterface.register(f"{mode}_blockrank", torch.compile(eager_blockrank_attention_forward, mode=mode) if mode != 'eager' else eager_blockrank_attention_forward)
        AttentionMaskInterface.register(f"{mode}_blockrank", eager_blockrank_attention_mask)
    
    AttentionInterface.register(f"flex_blockrank", torch.compile(flex_blockrank_attention_forward))
    AttentionMaskInterface.register(f"flex_blockrank", flex_blockrank_attention_mask)

    AttentionInterface.register(f"sdpa_blockrank", sdpa_blockrank_attention_forward)
    AttentionMaskInterface.register(f"sdpa_blockrank", eager_blockrank_attention_mask)

    AttentionInterface.register(f"sdpa_compiled_blockrank", torch.compile(sdpa_blockrank_attention_forward))
    AttentionMaskInterface.register(f"sdpa_compiled_blockrank", eager_blockrank_attention_mask)