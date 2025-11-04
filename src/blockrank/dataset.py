import os, sys, glob
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset
from .utils import (
    remap_documents,
    create_prompt_completion_format,
    create_conversation_format,
)

import torch
import numpy as np
from datasets import load_dataset, DatasetDict, IterableDatasetDict
from transformers import AutoTokenizer
import datasets

datasets.enable_caching()

def load_icr_dataset_hf(
    data_path: str,
    tokenizer: AutoTokenizer,
    num_documents: int = -1,
    seed: Optional[int] = 42,
    train_test_split: float = 1.0,
    streaming: bool = False,
    eval_mode: bool = False,
    use_blockrank: bool = False,
) -> DatasetDict:
    """
    Returns a DatasetDict with 'train' and 'test' splits, each item containing:
      - messages: list[{'role','content'}]
      - query: str
      - answer_ids: list[int]
      - num_documents: int
    """
    # handle sharded jsonl files
    if "*" in data_path:
        import glob
        data_files = sorted(list(glob.glob(data_path)))
    else:
        data_files = data_path
    cache_dir = os.path.join(os.path.dirname(data_path), "hf_cache")
    raw = load_dataset("json", data_files=data_files, split="train", streaming=streaming, cache_dir=cache_dir)

    if streaming:
        raw = raw.shuffle(seed=seed or 42)
        print('WARNING: Streaming mode enabled; train/test split will not be created.')
        ds_dict = IterableDatasetDict({"train": raw})
    else:
        if train_test_split >= 1.0:
            ds_dict = DatasetDict({"train": raw})
        else:
            ds_dict = raw.train_test_split(test_size=1 - train_test_split, seed=seed or 42)

    PROMPT_SEGMENT_SEP = "<<end_of_block_prompt_segment>>" if use_blockrank else "\n"

    def _sample_and_format(example, idx):
        query = example["query"]
        query_id = example.get("query_id", str(idx))
        documents = example["documents"]
        answer_ids = example["answer_ids"]
        if isinstance(documents, list):
            if isinstance(documents[0], dict):
                documents = {doc.get("doc_id", str(i)): f'{doc.get("title", "")} {doc.get("text", "")}'.strip() for i, doc in enumerate(documents)}
            else:
                documents = {str(i): doc for i, doc in enumerate(documents)}

        remapped_docs, remapped_doc_ids, remapped_ans_ids = remap_documents(
            documents=documents,
            answer_ids=answer_ids,
            num_samples=num_documents,
            seed=(seed or 42) + idx,
            sample=not eval_mode,
            add_padding_docs=not eval_mode,
        )

        pc = create_prompt_completion_format(query, remapped_docs, [] if eval_mode else remapped_ans_ids, sep=PROMPT_SEGMENT_SEP)
        return {
            "query": query,
            "query_id": query_id,
            "answer_ids": remapped_ans_ids,
            "remapped_doc_ids": remapped_doc_ids,
            "num_documents": len(remapped_docs),
            **pc,
        }
    
    def _tokenize_batch(batch):
        full_input_ids = tokenizer.apply_chat_template(
            [x + y  for x, y in zip(batch['prompt'], batch['completion'])],
            continue_final_message=eval_mode,
        )
        prompt_lengths = [len(x) for x in tokenizer.apply_chat_template(
            batch['prompt'],
        )]

        return {
            'input_ids': full_input_ids,
            'prompt_lengths': prompt_lengths,
        }

    def _block_tokenize_batch(batch):
        all_block_input_texts = []
        for itr in range(len(batch['prompt'])):
            input_texts = tokenizer.apply_chat_template(batch['prompt'][itr]+batch['completion'][itr], tokenize=False, continue_final_message=eval_mode)
            block_input_texts = input_texts.split(PROMPT_SEGMENT_SEP)
            n = len(block_input_texts)
            block_input_texts = [f'\n{x}' if i > 0 and i < n-1 else x for i, x in enumerate(block_input_texts)]
            all_block_input_texts.append(block_input_texts)

        indptr = np.cumsum([0] + [len(x) for x in all_block_input_texts])

        all_block_input_ids = tokenizer(
            [x for y in all_block_input_texts for x in y],
            add_special_tokens=False,
            return_attention_mask=False,
        )['input_ids']
        all_block_input_ids = [all_block_input_ids[indptr[i]:indptr[i+1]] for i in range(len(all_block_input_texts))]
        block_lengths = [[len(x) for x in y] for y in all_block_input_ids]
        all_block_input_ids = [[x for y in ex for x in y] for ex in all_block_input_ids]
        return {
            'input_ids': all_block_input_ids,
            'block_lengths': block_lengths,
        }

    ds_dict = ds_dict.map(_sample_and_format, with_indices=True, remove_columns=['documents'], num_proc=os.cpu_count()-2)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    ds_dict = ds_dict.map(
        _block_tokenize_batch if use_blockrank else _tokenize_batch,
        batched=True,
        batch_size=64,
        # remove_columns=['prompt', 'completion'],
        num_proc=os.cpu_count()-2
    )

    ds_dict = ds_dict.with_format("torch")

    return ds_dict
def icr_collate_fn(batch, tok, pad_to_multiple_of=8, max_seq_length=None, always_max_len=False) -> Dict[str, torch.Tensor]:
    pad_token_id = tok.pad_token_id
    padding_side = tok.padding_side
    if always_max_len:
        max_seq_length = max_seq_length or max([item['input_ids'].size(0) for item in batch])
    else:
        max_seq_length = min(max([item['input_ids'].size(0) for item in batch]), max_seq_length or int(1e9))
    
    if pad_to_multiple_of is not None:
        max_seq_length = ((max_seq_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    padding_input_id = torch.full((max_seq_length,), pad_token_id, dtype=torch.long)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'].squeeze(0)[:max_seq_length] for item in batch] + [padding_input_id],
        batch_first=True,
        padding_value=pad_token_id,
        padding_side=padding_side,
    )[:-1] # remove extra padding row
    B, S = input_ids.shape # batch size, seq len
    attention_mask = (input_ids != pad_token_id)
    labels = input_ids.clone()
    labels[input_ids == pad_token_id] = -100 # pad tokens not to be predicted
    
    # Adjust prompt_lengths based on padding side and truncation
    prompt_lengths = torch.tensor([item['prompt_lengths'] for item in batch])
    original_lengths = torch.tensor([item['input_ids'].size(0) for item in batch])
    
    if padding_side == 'left':
        # With left padding, prompt positions shift right by padding amount
        padding_amounts = max_seq_length - torch.min(original_lengths, torch.tensor(max_seq_length))
        adjusted_len_prompt = prompt_lengths + padding_amounts
    else:  # right padding
        # With right padding, if sequence is truncated, adjust prompt length
        adjusted_len_prompt = torch.min(prompt_lengths, torch.tensor(max_seq_length))
    
    labels[torch.arange(S)[None, :] < adjusted_len_prompt[:, None]] = -100 # prompt tokens not to be predicted

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

def block_icr_collate_fn(batch, tok, pad_to_multiple_of=16, max_block_length=None, always_max_len=False) -> Dict[str, torch.Tensor]:
    pad_token_id = tok.pad_token_id
    padding_side = tok.padding_side
    B = len(batch)
    M = len(batch[0]['block_lengths']) - 1  # number of prompt blocks
    # merge completion block into last prompt block
    for item in batch:
        item['last_prompt_block_lengths'] = item['block_lengths'][-2].item()
        item['completion_lengths'] = item['block_lengths'][-1].item()
        item['block_lengths'] = item['block_lengths'][:-1]
        item['block_lengths'][-1] += item['completion_lengths']
        assert sum(item['block_lengths']) == item['input_ids'].size(0), "Block lengths do not sum to input_ids length"
        assert len(item['block_lengths']) == M, "Number of blocks mismatch"

    if always_max_len:
        max_block_length = max_block_length or max([item['block_lengths'].max().item() for item in batch])
    else:
        max_block_length = min(max([item['block_lengths'].max().item() for item in batch]), max_block_length or int(1e9))

    if pad_to_multiple_of is not None:
        max_block_length = ((max_block_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    all_block_input_ids = []
    for item in batch:
        indptr = torch.cumsum(torch.cat([torch.tensor([0]), item['block_lengths']]), dim=0)
        item['block_input_ids'] = [item['input_ids'][indptr[i]:indptr[i+1]][:max_block_length] for i in range(len(item['block_lengths']))]
        all_block_input_ids.extend(item['block_input_ids'])

    padding_input_id = torch.full((max_block_length,), pad_token_id, dtype=torch.long)
    input_ids = torch.nn.utils.rnn.pad_sequence(
        all_block_input_ids + [padding_input_id],
        batch_first=True,
        padding_value=pad_token_id,
        padding_side=padding_side,
    )[:-1] # remove extra padding row
    BM, H = input_ids.shape # batch size, seq len
    input_ids = input_ids.view(B, M, H)
    attention_mask = (input_ids != pad_token_id)
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100 # pad tokens not to be predicted
    labels[:, :-1, :] = -100  # only last block trainable


    # Adjust last_prompt_block_lengths based on padding side and truncation
    last_prompt_block_lengths = torch.tensor([item['last_prompt_block_lengths'] for item in batch])
    original_lengths = torch.tensor([item['block_lengths'][-1] for item in batch])

    if padding_side == 'left':
        # With left padding, prompt positions shift right by padding amount
        padding_amounts = max_block_length - torch.min(original_lengths, torch.tensor(max_block_length))
        adjusted_len_prompt = last_prompt_block_lengths + padding_amounts
    else:  # right padding
        # With right padding, if sequence is truncated, adjust prompt length
        adjusted_len_prompt = torch.min(last_prompt_block_lengths, torch.tensor(max_block_length))

    labels[:, -1, :] = torch.where(
        torch.arange(max_block_length)[None, :] < adjusted_len_prompt[:, None], 
        -100, 
        labels[:, -1, :]) # prompt tokens not to be predicted

    # permutation invariant position ids respecting block boundaries
    position_ids = attention_mask.cumsum(-1)
    position_ids[:, 1:-1] += position_ids[:, 0].max(dim=-1).values[:, None, None] # offset by previous block max
    position_ids[:, -1] += 16384  # a large position offset for last block
    position_ids = torch.clamp_min(position_ids-1, 0)
    position_ids[~attention_mask] = 0 # pad positions

    # Extract and pad answer_ids from batch items (for auxiliary loss)
    answer_ids_padded = torch.nn.utils.rnn.pad_sequence(
        [item['answer_ids'] for item in batch],
        batch_first=True,
        padding_value=-1
    )  # Shape: (B, max_num_answers), padded with -1

    return {
        'input_ids': input_ids.view(B, M*H),
        'position_ids': position_ids.view(B, M*H),
        'attention_mask': attention_mask,
        'labels': labels.view(B, M*H),
        'num_blocks': torch.tensor(M, dtype=torch.long, device=input_ids.device),
        'answer_ids': answer_ids_padded,  # Padded 2D tensor (B, max_num_answers)
    }