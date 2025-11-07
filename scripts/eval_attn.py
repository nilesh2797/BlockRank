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

    # Replace subsequent layers with pass-through module to skip computation
    class PassThroughLayer(torch.nn.Module):
        """A no-op layer that just passes hidden_states through without computation"""
        def forward(self, hidden_states, **kwargs):
            return hidden_states

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
            labels = batch.pop('labels')  # Remove labels from batch
            answer_ids = batch.pop('answer_ids', None)  # Remove answer_ids if present
            out = unwrapped_model(**batch, output_attentions=True, layers_to_return_scores=[target_layer_idx])

            attn_scores = compute_auxiliary_attention_loss(
                attention_scores=out.attentions[0],
                labels=labels,
                attention_mask=batch['attention_mask'],
                answer_ids=None,
                return_logits=True,
            )  # (B, num_docs)

            # Get top-k predictions (k=10 for ranking metrics)
            k = min(10, attn_scores.shape[-1])  # Handle cases where num_docs < 10
            attn_preds = torch.topk(attn_scores, k=k, dim=-1).indices # (B, k)

            # Gather predictions from all processes
            attn_preds = accelerator.gather_for_metrics(attn_preds)

            if accelerator.is_local_main_process:
                # Convert to list of lists for calculate_accuracy
                batch_preds = [pred.cpu().tolist() for pred in attn_preds]
                all_attn_preds.extend(batch_preds)

                # Calculate intermediate accuracy every 20 batches
                if pbar.n % 20 == 0 or pbar.n == len(dataloader):
                    # Create a slice of eval_ds for current predictions
                    results = calculate_accuracy(
                        all_attn_preds,
                        eval_ds.select(range(len(all_attn_preds))),
                        qrels=qrels,
                    )
                    wandb.log({f"intermediate_eval/{k}": v for k, v in results.items()}, step=len(all_attn_preds))
                    logger.info({"acc": f"{results['accuracy']:.2f}%", "ndcg@10": f"{results.get('ndcg@10', 0):.2f}"})

        accelerator.wait_for_everyone()

    # Only main process computes metrics and saves
    if accelerator.is_local_main_process:
        results = calculate_accuracy(all_attn_preds, eval_ds, qrels=qrels)

        # Log to W&B
        wandb.log(results)

        # Save results
        os.makedirs(targs.output_dir, exist_ok=True)
        metrics_file = os.path.join(targs.output_dir, "attn_eval_metrics.json")
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