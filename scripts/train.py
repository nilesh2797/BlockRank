import os
import argparse
import logging
import accelerate
from accelerate.utils import TorchDynamoPlugin
from functools import partial
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    HfArgumentParser,
    TrainerCallback,
)
import datasets
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from blockrank.dataset import load_icr_dataset_hf, icr_collate_fn, block_icr_collate_fn
from blockrank.trainer import BlockRankAuxLossTrainer
from torch import distributed as dist

from blockrank import blockrank_std_attention, blockrank_triton_kernel_attention
blockrank_std_attention.register_blockrank_attention(); # standard SDPA-based and torch compiled BlockRank
blockrank_triton_kernel_attention.register_triton_blockrank_attention(); # Triton-kernel based BlockRank - only supports inference at the moment!

datasets.enable_caching()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
dynamo_plugin = TorchDynamoPlugin(
    backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
    mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
    fullgraph=True,
    dynamic=False,
    use_regional_compilation=True,
)
accelerator = accelerate.Accelerator()
IS_MAIN = accelerator.is_main_process

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO if IS_MAIN else logging.WARN,
)
logger = logging.getLogger("blockrank")
print = logger.debug if not IS_MAIN else print

@dataclass
class ModelArgs:
    model_name_or_path: str = ""
    use_4bit: bool = False
    use_lora: bool = False
    lora_r: int = -1
    lora_alpha: int = -1
    lora_dropout: float = 0.0
    lora_target_modules: Optional[str] = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    trust_remote_code: bool = False
    use_blockrank: bool = False
    attn_implementation: str = ""  # e.g., "default_blockrank", "triton_blockrank"

@dataclass
class DataArgs:
    data_path: str | None = None
    val_data_path: str | None = None
    num_documents: int | None = None
    streaming: bool = False
    train_test_split: float = 0.99
    max_seq_length: int | None = None
    dataset_seed: int = 42
    max_block_length: int = 256
    qrels_path: Optional[str] = None

@dataclass
class TrainArgs(SFTConfig):
    # Inherit all TrainingArguments via SFTConfig, add/override defaults you use
    output_dir: str
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    report_to: Optional[str] = "wandb"
    run_name: Optional[str] = None
    do_eval: bool = True
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    remove_unused_columns: bool = False
    save_strategy: str = "no"
    logging_first_step: bool = True
    # available in recent transformers; safe to default None
    gradient_checkpointing_kwargs: Optional[dict] = None
    # skip_prepare_dataset True by default
    dataset_kwargs: Dict[str, Any] = field(default_factory=lambda: {"skip_prepare_dataset": True})
    # Auxiliary loss parameters
    use_aux_loss: bool = False
    aux_layer_idx: int = 20
    aux_loss_weight: float = 0.1
    aux_temperature: float = 0.1

def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        logger.info("No config file provided or file not found. Using defaults/CLI.")
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    logger.info(f"Loaded configuration from {path}")
    return cfg

def setup_model_and_tokenizer(m: ModelArgs, device_map: str = "auto"):
    logger.info(f"Loading tokenizer and model from {m.model_name_or_path}")
    tok = AutoTokenizer.from_pretrained(m.model_name_or_path, use_fast=True, trust_remote_code=m.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
        tok.pad_token_id = tok.unk_token_id

    q_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if m.use_4bit else None

    # set device_map only when not in distributed mode
    model = AutoModelForCausalLM.from_pretrained(
        m.model_name_or_path,
        quantization_config=q_cfg,
        dtype=torch.bfloat16,
        trust_remote_code=m.trust_remote_code,
        **({} if dist.is_initialized() else {"device_map": device_map}),
    )
    if m.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configure BlockRank attention if requested
    if m.use_blockrank and m.attn_implementation:
        logger.info(f"Setting attention implementation to {m.attn_implementation}")
        model.set_attn_implementation(m.attn_implementation)

    if m.use_lora:
        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=m.lora_r,
            lora_alpha=m.lora_alpha,
            lora_dropout=m.lora_dropout,
            target_modules=m.lora_target_modules.split(","),
            bias="none",
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tok

def load_datasets(d: DataArgs, tokenizer, use_blockrank: bool = False):
    kwargs = {
        'tokenizer': tokenizer,
        'num_documents': d.num_documents,
        'seed': d.dataset_seed,
        'streaming': d.streaming,
        'use_blockrank': use_blockrank,
        'eval_mode': False,
    }
    if d.val_data_path:
        # Build separate train and val from two files
        train_ds = load_icr_dataset_hf(data_path=d.data_path, train_test_split=1.0, **kwargs)["train"]
        val_ds = load_icr_dataset_hf(data_path=d.val_data_path, train_test_split=1.0, **kwargs)["train"]
    else:
        ds = load_icr_dataset_hf(data_path=d.data_path, train_test_split=d.train_test_split, **kwargs)
        train_ds = ds["train"]
        val_ds = ds.get("test", None)

    logger.info(f"Train: {train_ds} | Val examples: {val_ds}")
    return train_ds, val_ds

def main():
    # Parse --config first, then dataclass args (so we support YAML or pure CLI)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", type=str, default=None)
    cfg_args, remaining = ap.parse_known_args()
    cfg = load_config(cfg_args.config)

    parser = HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    if cfg:
        merged = {**cfg.get("model", {}), **cfg.get("data", {}), **cfg.get("training", {})}
        margs, dargs, targs = parser.parse_dict(merged)
    else:
        margs, dargs, targs = parser.parse_args_into_dataclasses(remaining)
    if 'blockrank' in margs.attn_implementation:
        margs.use_blockrank = True
        logger.info("BlockRank attention enabled based on attn_implementation=" + margs.attn_implementation)
    set_seed(targs.seed)

    if IS_MAIN:
        os.makedirs(targs.output_dir, exist_ok=True)

        with open(os.path.join(targs.output_dir, "train_config.yaml"), "w") as f:
            yaml.dump(
                {
                    "model": asdict(margs),
                    "data": asdict(dargs),
                    "training": targs.to_dict(),  # ensures enums and types are serializable
                },
                f,
                default_flow_style=False,
            )

    model, tok = setup_model_and_tokenizer(margs)
    with accelerator.main_process_first():
        train_ds, val_ds = load_datasets(dargs, tok, use_blockrank=margs.use_blockrank)
    if val_ds is None:
        targs.eval_strategy = "no"
        targs.do_eval = False

    # Minor DDP tweak for LoRA
    if margs.use_lora and getattr(targs, "ddp_find_unused_parameters", None) is None:
        targs.ddp_find_unused_parameters = False

    # Select appropriate collate function based on use_blockrank
    if margs.use_blockrank:
        data_collator = partial(block_icr_collate_fn, tok=tok, max_block_length=dargs.max_block_length)
        logger.info(f"Using BlockRank collate function with max_block_length={dargs.max_block_length}")
    else:
        data_collator = partial(icr_collate_fn, tok=tok, max_seq_length=dargs.max_seq_length)
        logger.info(f"Using standard collate function with max_seq_length={dargs.max_seq_length}")
    logger.info("Initializing Trainer")

    # Use custom trainer if auxiliary loss is enabled
    TrainerClass = BlockRankAuxLossTrainer if targs.use_aux_loss else SFTTrainer
    if targs.use_aux_loss:
        logger.info(f"Using BlockRankAuxLossTrainer with aux_layer_idx={targs.aux_layer_idx}, "
                   f"aux_loss_weight={targs.aux_loss_weight}, aux_temperature={targs.aux_temperature}")

    trainer = TrainerClass(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds if targs.do_eval else None,
        data_collator=data_collator,
    )

    # Preview one training batch (enable with: PRINT_TRAIN_BATCH=1)
    if os.environ.get("PRINT_TRAIN_BATCH", "0") == "1":
        dl = trainer.get_train_dataloader()
        batch = next(iter(dl))

        def _show_sample(i: int = 0):
            ids = batch["input_ids"][i]
            labels = batch["labels"][i]
            am = batch.get("attention_mask")
            print("Batch shapes:", {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")})
            print("Seq len:", ids.shape[-1], "| Attn sum:",
                  int(am[i].sum().item()) if am is not None else "N/A")

            print("\nDecoded input:")
            print(tok.decode(ids, skip_special_tokens=False))

            pad_id = tok.pad_token_id or tok.eos_token_id
            loss_ids = torch.where(labels != -100, labels, torch.full_like(labels, pad_id))
            print("\nAssistant-loss tokens only:")
            print(tok.decode(loss_ids, skip_special_tokens=False))

            mask_line = "".join("█" if int(x) != -100 else "·" for x in labels.tolist())
            print("\nLabel mask (█=loss, ·=ignore):")
            print(mask_line)

        _show_sample(0)

    logger.info("Starting training")
    try:
        train_result = trainer.train()
        trainer.save_model(targs.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

        if targs.do_eval:
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        logger.info(f"Training completed. Model saved to: {targs.output_dir}")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Saving checkpoint...")
        trainer.save_model(os.path.join(targs.output_dir, "interrupted"))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # trainer.save_model(os.path.join(targs.output_dir, "exception"))
        raise

if __name__ == "__main__":
    main()