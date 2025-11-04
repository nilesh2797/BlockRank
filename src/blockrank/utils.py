"""
Utility functions for document ranking with torchtune.
"""

import random
from typing import Dict, List, Tuple, Optional, Any
import json
import pytrec_eval


def remap_documents(
    documents: Dict[Any, str],
    answer_ids: List[str] | None,
    num_samples: int,
    seed: Optional[int] = None,
    sample: bool = True,
    add_padding_docs: bool = True,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Sample a subset of documents from the full set, ensuring at least one answer is included.

    Args:
        documents: Dictionary mapping document IDs to content
        answer_ids: List of correct answer document IDs (can be None or empty)
        num_samples: Number of documents to sample
        seed: Random seed for reproducibility
        add_padding_docs: Whether to add dummy documents if not enough documents are available

    Returns:
        Tuple of (sampled_documents_dict, remapped_answer_id)
        - sampled_documents_dict: Dict with keys "0" to "num_samples-1"
        - remapped_answer_id: List of String ID of the answer in the new 0-indexed range (empty if no answer_ids)
    """
    if seed is not None:
        random.seed(seed)

    if answer_ids is None:
        answer_ids = []

    if num_samples <= 0 or (num_samples > len(documents) and not add_padding_docs):
        num_samples = len(documents)

    # Get all document IDs
    all_doc_ids = list(documents.keys())

    if sample:
        answer_ids = answer_ids[:num_samples]

        # Sample negatives (excluding the answer)
        negative_ids = [doc_id for doc_id in all_doc_ids if doc_id not in answer_ids]

        # Sample num_samples-len(answer_ids) negatives
        num_negatives = min(num_samples - len(answer_ids), len(negative_ids))
        sampled_negatives = random.sample(negative_ids, num_negatives)

        # Combine answer + negatives
        sampled_ids = answer_ids + sampled_negatives
    else:
        # No sampling, take the first num_samples documents
        sampled_ids = all_doc_ids[:num_samples]

    # If not enough documents, add dummy documents
    if add_padding_docs:
        while len(sampled_ids) < num_samples:
            dummy_id = f"dummy_{len(sampled_ids)}"
            documents[dummy_id] = "This is a dummy document."
            sampled_ids.append(dummy_id)

    # Shuffle to randomize answer position
    random.shuffle(sampled_ids)

    # Create remapped dictionary with new IDs 0 to num_samples-1
    remapped_docs = []
    remapped_ids = []
    remapped_answer_ids = []

    for new_id, old_id in enumerate(sampled_ids):
        remapped_docs.append(documents[old_id])
        remapped_ids.append(old_id)
        if old_id in answer_ids:
            remapped_answer_ids.append(new_id)

    return remapped_docs, remapped_ids, remapped_answer_ids

def format_ranking_prompt(
    query: str,
    documents: List[str],
    sep: str,
) -> str:
    """
    Format query and documents into a prompt for document ranking.

    Args:
        query: The query text
        documents: Dictionary mapping document IDs to content
        answer_ids: Correct answer document ID (optional, for training)

    Returns:
        Formatted prompt string (default). If include_answer=True and answer_ids provided, returns tuple (prompt/segments, answer_ids/completion)
    """
    # Build the prompt
    instruction = '''You will be given a query and a list of documents. Each document will be formatted as ID: <id> | CONTENT: <content> | END ID: <id>. You need to read carefully and understand all of them and your goal is to find all document(s) from the list that can help answer the query.'''

    prompt_parts = [f"{instruction}\nQuery: {query}\n\nDocuments:"]

    format_doc = lambda id, text: f"ID: {id} | CONTENT: {text} | END ID: {id}"

    # Add documents in order
    doc_ids = range(len(documents))
    for doc_id in doc_ids:
        formatted_doc = format_doc(doc_id, documents[doc_id].strip())
        prompt_parts.append(formatted_doc)

    # Final query section
    final_section = (
        '\n====== Now let\'s start! ======\n'
        "Which document is most relevant to answer the query? Print out the ID of the document.\n"
        f"Query: {query}\n"
        "The following document(s) can help answer the query:"
    )
    prompt_parts.append(final_section)

    prompt = sep.join(prompt_parts)
    return prompt

def create_conversation_format(
    query: str,
    documents: Dict[str, str],
    answer_ids: List[str],
    sep="\n",
) -> List[Dict[str, str]]:
    return [
        {"role": "user", "content": format_ranking_prompt(query, documents, sep)},
        {"role": "assistant", "content": f"Final Answer: {sep}[" + (', '.join([str(x) for x in answer_ids]) + f"]" if answer_ids else "")}
    ]

def create_prompt_completion_format(
    query: str,
    documents: Dict[str, str],
    answer_ids: List[str],
    sep="\n",
) -> Dict[str, str]:
    m = create_conversation_format(query, documents, answer_ids, sep)
    return {"prompt": [m[0]], "completion": [m[1]]}

def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """
    Save data to JSONL file.

    Args:
        data: List of dictionaries
        file_path: Path to output JSONL file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def compute_token_stats(texts: List[str], tokenizer) -> Dict[str, float]:
    """
    Compute token statistics for a list of texts.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer object

    Returns:
        Dictionary with statistics (mean, max, min token counts)
    """
    token_counts = []
    for text in texts:
        tokens = tokenizer.encode(text)
        token_counts.append(len(tokens))

    return {
        "mean": sum(token_counts) / len(token_counts),
        "max": max(token_counts),
        "min": min(token_counts),
        "total": sum(token_counts),
    }

def parse_predicted_id(generated_text: str, valid_ids: Optional[List[str]] = None) -> Optional[str]:
    """
    Parse document ID from generated text.

    Args:
        generated_text: Text generated by the model
        valid_ids: Optional list of valid document IDs to validate against

    Returns:
        Parsed document ID or None if parsing fails
    """
    # Clean the text
    text = generated_text.strip()

    # Try to extract just the number
    # Handle cases like "5", "The answer is 5", "[5]", etc.
    import re

    # Look for standalone numbers
    numbers = re.findall(r'\b\d+\b', text)

    if not numbers:
        return None

    # Take the first number found
    predicted_id = numbers[0]

    # Validate against valid_ids if provided
    if valid_ids is not None:
        if predicted_id not in valid_ids:
            return None

    return predicted_id

def load_qrels(qrels_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load qrels file (either TREC format or BEIR TSV format).

    Args:
        qrels_path: Path to qrels file

    Returns:
        Dictionary mapping query_id -> {doc_id: relevance_score}
    """
    qrels = {}
    with open(qrels_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or (i == 0 and line.startswith('query-id')):  # Skip header in BEIR format
                continue

            parts = line.split('\t') if '\t' in line else line.split()

            if len(parts) == 4:  # TREC format: query_id iteration doc_id relevance
                query_id, _, doc_id, relevance = parts
            elif len(parts) == 3:  # BEIR format: query_id doc_id relevance
                query_id, doc_id, relevance = parts
            else:
                continue

            relevance = int(relevance)
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][str(doc_id)] = relevance

    return qrels

def calculate_accuracy(
    predictions: List[int | List[int]],
    eval_ds,
    qrels: Optional[Dict[str, Dict[str, int]]] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Calculate accuracy and ranking metrics.

    Args:
        predictions: List of predicted document indices (can be single int or list of ints for top-k)
        eval_ds: Evaluation dataset containing answer_ids, query_id, and remapped_doc_ids
        qrels: Optional qrels dictionary mapping query_id -> {doc_id: relevance}
        verbose: Print detailed predictions

    Returns:
        Dictionary with metrics including accuracy, nDCG@k, and MRR@k
    """
    # Extract ground truth from eval_ds
    ground_truth = list(eval_ds['answer_ids'])[:len(predictions)]

    # Normalize predictions and ground_truth
    # For predictions: if it's a list of ints (top-k), keep it; if single value, wrap it
    normalized_preds = []
    for p in predictions:
        if isinstance(p, list):
            normalized_preds.append([int(x) for x in p])
        else:
            normalized_preds.append([int(p)])

    ground_truth = [int(g) if isinstance(g, int | str | float) else [int(x) for x in g] for g in ground_truth]

    correct = 0
    total = len(predictions)
    invalid = 0
    assert total == len(ground_truth), "Predictions and ground truth length mismatch."

    metrics = {}

    # Calculate basic accuracy using top-1 prediction (first element)
    for pred, gt in zip(normalized_preds, ground_truth):
        try:
            # For accuracy, check if top-1 prediction (first element) matches ground truth
            top1_pred = [pred[0]] if pred else []
            gt_list = gt if isinstance(gt, list) else [gt]
            correct += (set(top1_pred).issubset(set(gt_list)))
        except:
            invalid += 1
        if verbose:
            print(f"Pred (top-1): {pred[0] if pred else None} | GT: {gt}")

    metrics.update({
        "accuracy": 100 * correct / total if total > 0 else 0.0,
        "exact_match": correct,
        "total": total,
        "invalid_predictions": invalid,
        "invalid_rate": 100 * invalid / total if total > 0 else 0.0,
    })

    # Calculate ranking metrics if qrels is provided
    if qrels is not None:
        # Build run dictionary for pytrec_eval: query_id -> {doc_id: score}
        run = {}

        for i in range(len(normalized_preds)):
            query_id = str(eval_ds['query_id'][i])
            remapped_doc_ids = eval_ds['remapped_doc_ids'][i]

            # Skip if query not in qrels
            if query_id not in qrels:
                continue

            # Get top-k predictions for this query
            pred_ranking = normalized_preds[i]  # List of indices in ranked order

            # Create ranking: assign scores based on position in the ranking
            # Higher rank (earlier position) = higher score
            run[query_id] = {}

            # Assign scores to ranked documents (descending scores for ascending ranks)
            for rank, doc_idx in enumerate(pred_ranking):
                if doc_idx < len(remapped_doc_ids):
                    doc_id = remapped_doc_ids[doc_idx]
                    # Score decreases with rank: rank 0 (top) gets highest score
                    run[query_id][str(doc_id)] = float(len(pred_ranking) - rank)

            # Assign score of 0 to all unranked documents
            for doc_idx, doc_id in enumerate(remapped_doc_ids):
                if doc_idx not in pred_ranking:
                    run[query_id][str(doc_id)] = 0.0

        # Define metrics to calculate
        measures = {
            'ndcg_cut_1',
            'ndcg_cut_3',
            'ndcg_cut_5',
            'ndcg_cut_10',
            'recip_rank',
        }

        # Create evaluator
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)

        # Evaluate
        results = evaluator.evaluate(run)

        # Aggregate results across queries
        ndcg_scores = {k: [] for k in [1, 3, 5, 10]}
        mrr_scores = []

        for query_id, query_results in results.items():
            for k in [1, 3, 5, 10]:
                ndcg_scores[k].append(query_results.get(f'ndcg_cut_{k}', 0.0))
            mrr_scores.append(query_results.get('recip_rank', 0.0))

        # Calculate averages
        for k in [1, 3, 5, 10]:
            if ndcg_scores[k]:
                metrics[f'ndcg@{k}'] = 100*sum(ndcg_scores[k]) / len(ndcg_scores[k])
            # Also add MRR@k (which is just MRR evaluated up to rank k)
            # For MRR, we can compute it for all k values, though it's typically just one value
            if mrr_scores:
                metrics[f'mrr@{k}'] = 100*sum(mrr_scores) / len(mrr_scores)

    return metrics