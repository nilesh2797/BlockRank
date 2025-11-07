"""
Auxiliary attention loss for BlockRank training.

This module implements the contrastive loss that optimizes query-document
attention patterns during fine-tuning.
"""

import torch
import torch.nn.functional as F


def compute_auxiliary_attention_loss(
    attention_scores: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_ids: torch.Tensor | None = None,
    temperature: float = 0.05,
    return_logits = False,
) -> torch.Tensor:
    """
    Compute InfoNCE contrastive loss on attention scores at first loss position (should correspond to "[" token).

    This loss encourages the model to attend more strongly to relevant documents
    from the "[" token position (first non-masked token in the completion).

    Args:
        attention_scores: (B, N, H1, M*H) - attention logits from specified layer
            Returns last h1 query positions from the last block
        labels: (B, M*H) - label tensor with -100 for masked positions
        attention_mask: (B, M, H) - block-wise attention mask
        answer_ids: (B, max_num_answers) - positive document indices, padded with -1, or None return only logits
        temperature: float - temperature for InfoNCE loss (default: 0.1)

    Returns:
        loss: scalar tensor - InfoNCE contrastive loss
    """
    B, M, H = attention_mask.shape
    _, N, h1, MH = attention_scores.shape
    assert MH == M * H, "Attention scores last dimension must match M*H"

    # Step 1: Find bracket token position (first non-masked token in last h1 positions)
    # Look at last h1 positions in labels to find the "[" bracket token
    last_h1_labels = labels[:, -h1:]  # (B, h1)
    # Find first position where labels > -100 (first non-masked token = bracket)
    bracket_mask = last_h1_labels > -100  # (B, h1)
    bracket_indices = bracket_mask.int().argmax(dim=-1)[:, None]  # (B,) - index in [0, h1)
    bracket_indices = torch.hstack([bracket_indices-1, bracket_indices])  # (B, 2) - take bracket and previous token
    assert torch.all(bracket_indices >= 0), "Bracket token not found in last h1 positions."

    # Step 2: Extract attention scores at bracket position for document blocks only
    # attention_scores shape: (B, N, h1, M*H)
    # We need to index each batch item with its specific bracket position
    bracket_attn_logits = attention_scores.take_along_dim(bracket_indices[:, None, :, None], dim=2) # (B, N, 2, M*H)

    # Extract only document tokens (skip first block H and last block H)
    bracket_attn_logits = bracket_attn_logits[..., H:-H]  # (B, N, 2, (M-2)*H)

    # Step 2: Apply softmax over all document tokens
    bracket_attn = F.softmax(bracket_attn_logits, dim=-1)  # (B, N, 2, (M-2)*H)

    # Step 3: Reshape to separate documents
    # Documents are in blocks 1 to M-2 (block 0 is instruction, block M-1 is query)
    num_docs = M - 2
    bracket_attn = bracket_attn.reshape(B, N, -1, num_docs, H)  # (B, N, 2, num_docs, H)

    # Step 4: Aggregate to document-level scores
    # Average over attention heads, signal query tokens, sum over tokens within each document
    doc_scores = bracket_attn.mean(dim=(1, 2)).sum(dim=-1)  # (B, num_docs)

    # Step 5: Compute InfoNCE loss with multiple positives
    # answer_ids shape: (B, max_num_answers), values are doc indices or -1 (padding)

    # Apply temperature scaling
    logits = doc_scores / temperature  # (B, num_docs)

    if answer_ids is None:
        assert return_logits, "If answer_ids is None, return_logits must be True."
        return logits

    # Create mask for valid positives (B, num_docs+1) to safely handle -1 padding
    pos_mask = torch.zeros(B, num_docs + 1, dtype=torch.bool, device=logits.device)
    safe_answer_ids = torch.where(answer_ids >= 0, answer_ids, num_docs)  # Map negative -> num_docs
    pos_mask.scatter_(1, safe_answer_ids, True)
    pos_mask = pos_mask[:, :-1]  # Remove last column, back to (B, num_docs)
    
    # InfoNCE: -log(sum(exp(pos)) / sum(exp(all)))
    pos_logsumexp = torch.logsumexp(logits.masked_fill(~pos_mask, float('-inf')), dim=1)
    all_logsumexp = torch.logsumexp(logits, dim=1)
    
    loss = -(pos_logsumexp - all_logsumexp).mean()
    accuracy = pos_mask.gather(1, logits.argmax(dim=1, keepdim=True)).float().mean()
    
    return (loss, accuracy, logits) if return_logits else (loss, accuracy)
