import os
import sys
import argparse
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial

# Add scripts directory to path for train module imports
sys.path.insert(0, os.path.dirname(__file__))
from train import setup_model_and_tokenizer, load_config, ModelArgs, DataArgs, TrainArgs, logger

from transformers import HfArgumentParser, set_seed
from blockrank.dataset import load_icr_dataset_hf, icr_collate_fn, block_icr_collate_fn
from blockrank.utils import calculate_accuracy, load_qrels
from blockrank.losses import compute_auxiliary_attention_loss
from accelerate import Accelerator, DataLoaderConfiguration
import wandb

class IncrementalMetrics:
    """Incremental metric tracker with proper per-query aggregation."""
    def __init__(self, eval_data, qrels):
        self.answer_ids, self.query_ids, self.doc_ids = eval_data
        self.qrels = qrels
        self.last_idx = 0
        self.agg = {'correct': 0, 'total': 0, 'invalid': 0}
        self.run_dict = {}  # Store run dictionary incrementally, evaluate all at once

    def update(self, all_preds):
        """Incrementally update all metrics (accuracy + ranking)."""
        new_preds = all_preds[self.last_idx:]
        if not new_preds:
            return self._build_results()

        n = len(all_preds)
        ground_truth_raw = self.answer_ids[self.last_idx:n]

        # Normalize ground truth (match calculate_accuracy logic)
        ground_truth = [int(g) if isinstance(g, (int, str, float)) else [int(x) for x in g] for g in ground_truth_raw]

        # Normalize predictions
        normalized_preds = []
        for p in new_preds:
            if isinstance(p, list):
                normalized_preds.append([int(x) for x in p])
            else:
                normalized_preds.append([int(p)])

        # 1. Fast accuracy update (cheap)
        for pred, gt in zip(normalized_preds, ground_truth):
            try:
                top1_pred = [pred[0]] if pred else []
                gt_list = gt if isinstance(gt, list) else [gt]
                self.agg['correct'] += (set(top1_pred).issubset(set(gt_list)))
                self.agg['total'] += 1
            except:
                self.agg['invalid'] += 1
                self.agg['total'] += 1

        # 2. Incremental ranking metrics (only if qrels available)
        if self.qrels and self.query_ids:
            # Build run dict for NEW queries and add to accumulated run_dict
            for i, pred_ranking in enumerate(normalized_preds):
                global_idx = self.last_idx + i
                query_id = str(self.query_ids[global_idx])
                remapped_doc_ids = self.doc_ids[global_idx]

                if query_id not in self.qrels:
                    continue

                # Create ranking scores for this query
                self.run_dict[query_id] = {}
                for rank, doc_idx in enumerate(pred_ranking):
                    if doc_idx < len(remapped_doc_ids):
                        doc_id = str(remapped_doc_ids[doc_idx])
                        self.run_dict[query_id][doc_id] = float(len(pred_ranking) - rank)

                # Assign 0 to unranked docs
                for doc_idx, doc_id in enumerate(remapped_doc_ids):
                    if doc_idx not in pred_ranking:
                        self.run_dict[query_id][str(doc_id)] = 0.0

        self.last_idx = n
        return self._build_results()

    def _build_results(self):
        """Build current metrics from aggregated values."""
        results = {
            'accuracy': 100 * self.agg['correct'] / self.agg['total'] if self.agg['total'] > 0 else 0.0,
            'exact_match': self.agg['correct'],
            'total': self.agg['total'],
            'invalid_predictions': self.agg['invalid'],
            'invalid_rate': 100 * self.agg['invalid'] / self.agg['total'] if self.agg['total'] > 0 else 0.0,
        }

        # Evaluate all queries at once (avoid pytrec_eval evaluator reuse issues)
        if self.run_dict and self.qrels:
            import pytrec_eval
            measures = {'ndcg_cut_1', 'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10', 'recip_rank'}
            evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, measures)
            query_results = evaluator.evaluate(self.run_dict)

            # Aggregate across all queries (match calculate_accuracy logic exactly)
            num_queries = len(query_results)
            for k in [1, 3, 5, 10]:
                ndcg_sum = sum(qm.get(f'ndcg_cut_{k}', 0.0) for qm in query_results.values())
                mrr_sum = sum(qm.get('recip_rank', 0.0) for qm in query_results.values())
                results[f'ndcg@{k}'] = 100 * ndcg_sum / num_queries
                results[f'mrr@{k}'] = 100 * mrr_sum / num_queries

        return results

def main():
    # Reuse train.py argument parsing
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None)
    cfg_args, remaining = ap.parse_known_args()
    cfg = load_config(cfg_args.config)

    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    merged = {**cfg.get("model", {}), **cfg.get("data", {}), **cfg.get("eval", {})}
    margs, dargs, targs = parser.parse_dict(merged)
    # Override model path if checkpoint provided
    if cfg_args.checkpoint:
        margs.model_name_or_path = cfg_args.checkpoint
    if 'blockrank' in margs.attn_implementation:
        margs.use_blockrank = True
        logger.info("BlockRank attention enabled based on attn_implementation=" + margs.attn_implementation)

    dataloader_config = DataLoaderConfiguration(
        split_batches=False,
        even_batches=True,
        use_seedable_sampler=True,
    )
    accelerator = Accelerator(dataloader_config=dataloader_config)
    set_seed(targs.seed)

    # Initialize W&B on main process only
    if accelerator.is_local_main_process:
        wandb.init(
            project=getattr(targs, "wandb_project", "blockrank-attn-eval"),
            name=os.path.basename(targs.output_dir) + f"_{os.path.basename(margs.model_name_or_path)}_attn",
            config={
                "model": margs.__dict__,
                "data": dargs.__dict__,
                "eval": targs.__dict__,
                "checkpoint": cfg_args.checkpoint,
                "attn_layer": targs.aux_layer_idx,
            },
            job_type="attn_eval",
        )

    # Load model and tokenizer (reuse from train.py)
    model, tok = setup_model_and_tokenizer(margs, device_map='cuda:0')
    model.eval()

    # Load eval dataset (reuse from train.py)
    with accelerator.main_process_first():
        ds = load_icr_dataset_hf(
            data_path=dargs.data_path,
            tokenizer=tok,
            num_documents=-1,
            seed=dargs.dataset_seed,
            train_test_split=dargs.train_test_split,
            streaming=dargs.streaming,
            eval_mode=True,
            use_blockrank=margs.use_blockrank,
        )
        eval_ds = ds["test"] if ds.get("test", None) is not None else ds["train"]
        qrels = load_qrels(dargs.qrels_path) if hasattr(dargs, 'qrels_path') and dargs.qrels_path and os.path.exists(dargs.qrels_path) else None

    accelerator.wait_for_everyone()
    logger.info(f"Loaded {len(eval_ds)} examples")

    # Pre-extract data for fast incremental metrics
    metric_tracker = None
    if accelerator.is_local_main_process:
        eval_data = (
            list(eval_ds['answer_ids']),
            list(eval_ds['query_id']) if qrels else None,
            list(eval_ds['remapped_doc_ids']) if qrels else None
        )
        metric_tracker = IncrementalMetrics(eval_data, qrels)
        logger.info("Initialized incremental metric tracker")

    # Setup data collator (reuse from train.py)
    # Select appropriate collate function based on use_blockrank
    pad_to_multiple_of = dargs.__dict__.get("pad_to_multiple_of", 16)
    if margs.use_blockrank:
        data_collator = partial(block_icr_collate_fn, tok=tok, max_block_length=dargs.max_block_length, pad_to_multiple_of=pad_to_multiple_of)
        logger.info(f"Using BlockRank collate function with max_block_length={dargs.max_block_length}, pad_to_multiple_of={pad_to_multiple_of}")
    else:
        data_collator = partial(icr_collate_fn, tok=tok, max_seq_length=dargs.max_seq_length, pad_to_multiple_of=pad_to_multiple_of)
        logger.info(f"Using standard collate function with max_seq_length={dargs.max_seq_length}, pad_to_multiple_of={pad_to_multiple_of}")
    batch_size = getattr(targs, "per_device_eval_batch_size", None) or getattr(targs, "eval_batch_size", None) or 1
    dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    # Prepare model and dataloader with Accelerator
    model, dataloader = accelerator.prepare(model, dataloader)
    logger.info(f"Running attention-based evaluation on {accelerator.num_processes} processes...")
    logger.info(f"Using attention layer {targs.aux_layer_idx} for predictions")

    # Optimize by preventing computation in layers after the target layer
    unwrapped_model = accelerator.unwrap_model(model)
    target_layer_idx = targs.aux_layer_idx

    # Find the model's layer list
    if hasattr(unwrapped_model, 'model') and hasattr(unwrapped_model.model, 'layers'):
        layers = unwrapped_model.model.layers
    elif hasattr(unwrapped_model, 'layers'):
        layers = unwrapped_model.layers
    else:
        layers = None
        logger.warning("Could not find model layers, will compute all layers")

    original_forwards = []
    if layers is not None and target_layer_idx + 1 < len(layers):
        def identity_forward(self, hidden_states, *args, **kwargs):
            return hidden_states
        
        for i in range(target_layer_idx + 1, len(layers)):
            original_forwards.append((i, layers[i].forward))
            layers[i].forward = identity_forward.__get__(layers[i], type(layers[i]))
        unwrapped_model.lm_head.forward = identity_forward.__get__(unwrapped_model.lm_head, type(unwrapped_model.lm_head))
        logger.info(f"Monkey-patched {len(original_forwards)} layers & LM head after layer {target_layer_idx} with identity forward")

    all_attn_preds = []
    with torch.no_grad():
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc="Evaluating")
        for batch in pbar:
            # Forward pass with attention output
            labels = batch.pop('labels')
            answer_ids = batch.pop('answer_ids', None)
            out = unwrapped_model(**batch, output_attentions=True, layers_to_return_scores=[target_layer_idx], num_last_queries=32)

            attn_scores = compute_auxiliary_attention_loss(
                attention_scores=out.attentions[0],
                labels=labels,
                attention_mask=batch['attention_mask'],
                answer_ids=None,
                return_logits=True,
            )

            # Get top-k predictions (k=10 for ranking metrics)
            k = min(10, attn_scores.shape[-1])
            attn_preds = torch.topk(attn_scores, k=k, dim=-1).indices
            attn_preds = accelerator.gather_for_metrics(attn_preds)

            if accelerator.is_local_main_process:
                all_attn_preds.extend([pred.cpu().tolist() for pred in attn_preds])

                # Update metrics incrementally every iteration
                results = metric_tracker.update(all_attn_preds)
                pbar.set_postfix({"acc": f"{results['accuracy']:.2f}%", "ndcg@10": f"{results.get('ndcg@10', 0):.2f}"})

                # Log to wandb every 20 batches to avoid flooding
                if pbar.n % 20 == 0 or pbar.n == len(dataloader):
                    wandb.log({f"intermediate_eval/{k}": v for k, v in results.items()}, step=len(all_attn_preds))

        accelerator.wait_for_everyone()

    # Only main process computes metrics and saves
    if accelerator.is_local_main_process:
        results = calculate_accuracy(all_attn_preds, eval_ds, qrels=qrels)

        # Log to W&B
        wandb.log(results)

        # Save results
        os.makedirs(targs.output_dir, exist_ok=True)
        metrics_file = os.path.join(targs.output_dir, f"attn_eval_{os.path.basename(margs.model_name_or_path)}_metrics.json")
        results_with_config = {
            **results,
            "attn_layer": targs.aux_layer_idx,
        }
        with open(metrics_file, "w") as f:
            json.dump(results_with_config, f, indent=2)

        logger.info(f"\n{'='*50}\nAttention-based Evaluation Results:")
        logger.info(f"  Attention Layer: {targs.aux_layer_idx}")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        logger.info(f"{'='*50}\n")
        logger.info(f"Saved to {metrics_file}")

        # Log some example predictions
        examples_table = wandb.Table(
            columns=["Predicted ID", "Ground Truth"],
            data=[
                [str(all_attn_preds[i]), str(eval_ds['answer_ids'][i])]
                for i in range(min(100, len(all_attn_preds)))
            ]
        )
        wandb.log({"predictions_sample": examples_table})

        # Finish W&B run
        wandb.finish()

if __name__ == "__main__":
    main()