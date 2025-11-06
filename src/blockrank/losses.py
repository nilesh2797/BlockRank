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
    answer_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 0.05,
    return_logits = False,
) -> torch.Tensor:
    """
    Compute InfoNCE contrastive loss on attention scores at first loss position (should correspond to "[" token).

    This loss encourages the model to attend more strongly to relevant documents
    from the "[" token position (first non-masked token in the completion).

    Args:
        attention_scores: (B, N, 16, M*H) - attention logits from specified layer
            Returns last 16 query positions from the last block
        labels: (B, M*H) - label tensor with -100 for masked positions
        answer_ids: (B, max_num_answers) - positive document indices, padded with -1
        attention_mask: (B, M, H) - block-wise attention mask
        temperature: float - temperature for InfoNCE loss (default: 0.1)

    Returns:
        loss: scalar tensor - InfoNCE contrastive loss
    """
    B, M, H = attention_mask.shape

    # Step 1: Find bracket token position (first non-masked token in last 16 positions)
    # Look at last 16 positions in labels to find the "[" bracket token
    last_16_labels = labels[:, -16:]  # (B, 16)
    # Find first position where labels > -100 (first non-masked token = bracket)
    bracket_mask = last_16_labels > -100  # (B, 16)
    bracket_indices = bracket_mask.int().argmax(dim=-1)  # (B,) - index in [0, 15]

    # Step 2: Extract attention scores at bracket position for document blocks only
    # attention_scores shape: (B, N, 16, M*H)
    # We need to index each batch item with its specific bracket position
    batch_indices = torch.arange(B, device=attention_scores.device)
    bracket_attn_logits = attention_scores[batch_indices, :, bracket_indices, :]  # (B, N, M*H)

    # Extract only document tokens (skip first block H and last block H)
    bracket_attn_logits = bracket_attn_logits[:, :, H:-H]  # (B, N, (M-2)*H)

    # Step 2: Apply softmax over all document tokens
    bracket_attn = F.softmax(bracket_attn_logits, dim=-1)  # (B, N, (M-2)*H)

    # Step 3: Reshape to separate documents
    # Documents are in blocks 1 to M-2 (block 0 is instruction, block M-1 is query)
    num_docs = M - 2
    bracket_attn = bracket_attn.reshape(B, -1, num_docs, H)  # (B, N, num_docs, H)

    # Step 4: Aggregate to document-level scores
    # Average over attention heads, sum over tokens within each document
    doc_scores = bracket_attn.mean(dim=1).sum(dim=-1)  # (B, num_docs)

    # Step 5: Compute InfoNCE loss with multiple positives
    # answer_ids shape: (B, max_num_answers), values are doc indices or -1 (padding)

    # Apply temperature scaling
    logits = doc_scores / temperature  # (B, num_docs)

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
