import os
import sys
import argparse
import json
from tqdm import tqdm
import torch
from functools import partial

# Add scripts directory to path for train module imports
sys.path.insert(0, os.path.dirname(__file__))
from train import setup_model_and_tokenizer, load_config, ModelArgs, DataArgs, TrainArgs, logger

from transformers import HfArgumentParser, set_seed
from blockrank.dataset import load_icr_dataset_hf, icr_collate_fn
from blockrank.utils import parse_predicted_id, calculate_accuracy
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
            project=getattr(targs, "wandb_project", "blockrank-eval"),
            name=os.path.basename(targs.output_dir) + f"_{os.path.basename(margs.model_name_or_path)}",
            config={
                "model": margs.__dict__,
                "data": dargs.__dict__,
                "eval": targs.__dict__,
                "checkpoint": cfg_args.checkpoint,
            },
            job_type="eval",
        )

    # Load model and tokenizer (reuse from train.py)
    model, tok = setup_model_and_tokenizer(margs)
    model.eval()

    # Load eval dataset (reuse from train.py)
    with accelerator.main_process_first():
        ds = load_icr_dataset_hf(
            data_path=dargs.data_path,
            tokenizer=tok,
            num_documents=-1,
            seed=dargs.dataset_seed,
            train_test_split=dargs.train_test_split,
            streaming=False,
            eval_mode=True,
        )
        eval_ds = ds["test"] if ds.get("test", None) is not None else ds["train"]

    accelerator.wait_for_everyone()
    logger.info(f"Loaded {len(eval_ds)} examples")

    # Setup data collator (reuse from train.py)
    data_collator = partial(icr_collate_fn, tok=tok, max_seq_length=dargs.max_seq_length)
    batch_size = getattr(targs, "per_device_eval_batch_size", None) or getattr(targs, "eval_batch_size", None) or 1
    dataloader = torch.utils.data.DataLoader(eval_ds, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    # Prepare model and dataloader with Accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    generation_max_length = targs.generation_max_length if hasattr(targs, 'generation_max_length') else 10
    logger.info(f"Running evaluation on {accelerator.num_processes} processes...")

    all_generated = []
    all_predictions = []
    with torch.no_grad():
        pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in pbar:
            unwrapped_model = accelerator.unwrap_model(model)
            generated_toks = unwrapped_model.generate(**batch, max_new_tokens=generation_max_length, pad_token_id=tok.pad_token_id)
            generated_toks = generated_toks[:, batch['input_ids'].shape[1]:]  # only new tokens

            # Pad to max length before gathering to ensure same shape across GPUs
            max_gen_len = generated_toks.shape[1]
            
            # Get max length across all processes
            max_gen_len_tensor = torch.tensor([max_gen_len], device=generated_toks.device)
            gathered_lens = accelerator.gather(max_gen_len_tensor)
            global_max_len = gathered_lens.max().item()
            
            # Pad to global max length
            if generated_toks.shape[1] < global_max_len:
                padding = torch.full(
                    (generated_toks.shape[0], global_max_len - generated_toks.shape[1]),
                    tok.pad_token_id,
                    dtype=generated_toks.dtype,
                    device=generated_toks.device
                )
                generated_toks = torch.cat([padding, generated_toks], dim=1)
    

            # Gather from all processes
            generated_toks = accelerator.gather_for_metrics(generated_toks)

            if accelerator.is_local_main_process:
                generated_txts = [tok.decode(g, skip_special_tokens=True) for g in generated_toks.cpu()]
                all_generated.extend(generated_txts)
                all_predictions.extend([parse_predicted_id(txt) for txt in generated_txts])
                results = calculate_accuracy(all_predictions, list(eval_ds['answer_ids'])[:len(all_predictions)])
                wandb.log({f"intermediate_eval/{k}": v for k, v in results.items()})
                pbar.set_description("Eval Acc: {:.4f}".format(results['accuracy']))
    
            accelerator.wait_for_everyone()

    # Only main process computes metrics and saves
    if accelerator.is_local_main_process:
        results = calculate_accuracy(all_predictions, list(eval_ds['answer_ids']))

        # Log to W&B
        wandb.log(results)
        
        # Log some example predictions
        examples_table = wandb.Table(
            columns=["Generated Text", "Predicted ID", "Ground Truth"],
            data=[
                [all_generated[i], all_predictions[i], eval_ds['answer_ids'][i]]
                for i in range(min(100, len(all_generated)))
            ]
        )
        wandb.log({"predictions_sample": examples_table})

        # Save results
        os.makedirs(targs.output_dir, exist_ok=True)
        metrics_file = os.path.join(targs.output_dir, "eval_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n{'='*50}\nEvaluation Results:")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        logger.info(f"{'='*50}\n")
        logger.info(f"Saved to {metrics_file}")
        
        # Finish W&B run
        wandb.finish()

if __name__ == "__main__":
    main()
