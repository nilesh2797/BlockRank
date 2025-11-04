# BlockRank Training Guide

This guide provides detailed instructions for training BlockRank models on your own data.

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training Modes](#training-modes)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Multi-GPU Training](#multi-gpu-training)

## Quick Start

### 1. Prepare Your Data

Format your data as JSONL (see [DATA_FORMAT.md](DATA_FORMAT.md)):

```json
{
  "query": "what is machine learning",
  "query_id": "q1",
  "documents": [
    {"doc_id": "0", "title": "ML Overview", "text": "Machine learning is..."},
    {"doc_id": "1", "title": "AI Basics", "text": "Artificial intelligence..."}
  ],
  "answer_ids": ["0"]
}
```

### 2. Create Configuration File

Copy and modify an existing config:

```bash
cp src/configs/train_full_blockrank_10p_msmsarco.yaml my_config.yaml
```

Edit `my_config.yaml`:

```yaml
model:
  model_name_or_path: "mistralai/Mistral-7B-Instruct-v0.3"
  use_blockrank: true
  attn_implementation: "default_blockrank" # or "sdpa_compiled_blockrank"

data:
  data_path: "data/your_data.jsonl"
  num_documents: 30
  max_block_length: 160

training:
  output_dir: "outputs/my-model"
  num_train_epochs: 1
  per_device_train_batch_size: 1 # increase if memory allows
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-6
  use_aux_loss: true
  aux_layer_idx: 20 # for mistral-7b
  aux_loss_weight: 0.1
```

### 3. Run Training

```bash
python scripts/train.py --config my_config.yaml
```

## Configuration

### Model Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name_or_path` | str | Required | Base model to fine-tune |
| `use_blockrank` | bool | `false` | Enable BlockRank attention |
| `attn_implementation` | str | None | Attention backend (`eager_blockrank`, `sdpa_blockrank`, `triton_blockrank`) |
| `use_lora` | bool | `false` | Enable LoRA fine-tuning |

**Attention Implementations:**

- `eager_blockrank`: Pure PyTorch (best for debugging)
- `default_blockrank`: compiled with torch.compile (balanced)
- `sdpa_compiled_blockrank`: Uses F.scaled_dot_product_attention (balanced)
- `flex_blockrank`: Uses flex_attention (experimental, PyTorch 2.5+)
- `triton_blockrank`: Custom Triton kernels (fastest, inference-only)

**Recommendation**: Use `default_blockrank` for training, `triton_blockrank` for inference.

### Data Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | Required | Path to JSONL training data |
| `qrels_path` | str | `None` | Path to qrels file (optional for ranking evals) |
| `num_documents` | int | `30` | Number of documents per query |
| `max_block_length` | int | `160` | Max tokens per block (BlockRank mode) |
| `max_seq_length` | int | `4096` | Max total sequence length (standard mode, ignored for blockrank) |
| `pad_to_multiple_of` | int | `16` | Pad sequences to multiple of N |
| `train_test_split` | float | `1.0` | Train/test split ratio |
| `dataset_seed` | int | `42` | Random seed for data sampling |

**Key Parameter: `num_documents`**
- Controls how many documents are sampled per query during training
- Larger values = more challenging training but longer sequences
- Recommended: 20-50 for training, -1 (all) for evaluation

**Key Parameter: `max_block_length`**
- Maximum tokens per document block (BlockRank mode only)
- Affects memory usage and context window
- Recommended: 128-256 depending on document length

### Training Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | Required | Where to save checkpoints |
| `num_train_epochs` | int | `1` | Number of training epochs |
| `per_device_train_batch_size` | int | `1` | Batch size per GPU |
| `gradient_accumulation_steps` | int | `16` | Gradient accumulation steps |
| `learning_rate` | float | `5e-6` | Learning rate |
| `lr_scheduler_type` | str | `"cosine"` | LR scheduler type |
| `warmup_ratio` | float | `0.03` | Warmup ratio |
| `weight_decay` | float | `0.0` | Weight decay |
| `max_grad_norm` | float | `1.0` | Gradient clipping |
| `save_strategy` | str | `"epoch"` | When to save checkpoints |
| `logging_steps` | int | `10` | Log every N steps |
| `use_aux_loss` | bool | `false` | Enable auxiliary contrastive loss |
| `aux_layer_idx` | int | `20` | Layer to extract attention from |
| `aux_loss_weight` | float | `0.1` | Weight for auxiliary loss |
| `aux_temperature` | float | `0.1` | Temperature for InfoNCE loss |

**Effective Batch Size** = `per_device_train_batch_size` × `gradient_accumulation_steps` × `num_gpus`

**Recommended**: Effective batch size of 16-32 for most setups.

## Training Modes

### 1. Standard Fine-tuning (No Auxiliary Loss)

Train with standard language modeling objective only:

```yaml
training:
  use_aux_loss: false
```

### 2. BlockRank with Auxiliary Loss (Recommended)

Train with both LM loss and auxiliary attention loss:

```yaml
training:
  use_aux_loss: true
  aux_layer_idx: 20  # Middle layer (for 32-layer models)
  aux_loss_weight: 0.1
  aux_temperature: 0.1
```

**Total Loss** = `lm_loss + λ * aux_loss`

### 3. LoRA Fine-tuning (Memory Efficient)

Use LoRA for parameter-efficient fine-tuning:

```yaml
model:
  use_lora: true
  lora_r: 64  # Rank (16-128)
  lora_alpha: 128  # Typically 2×rank
  lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

- Train only ~0.5% of parameters
- Faster training and less memory

## Hyperparameter Tuning

### Learning Rate

**Recommended ranges:**
- Full fine-tuning: `5e-7` to `5e-6`
- LoRA: `5e-6` to `1e-4`


### Auxiliary Layer Selection

| Layer Position (mistral) | Trade-off |
|---------------|-----------|
| Early (0-14) | More low-level features, less semantic |
| **Middle (15-25)** | **Recommended** - balanced semantic representation |
| Late (25-32) | closer to output - weaker attention based signals |

**For Mistral-7B (32 layers)**: we use layer 20 in experiments

### Number of Documents

| `num_documents` | Training Time | Difficulty | Generalization |
|----------------|---------------|------------|----------------|
| 10 | Fast | Easy | Moderate |
| 20-30 | **Recommended** | Moderate | Good |
| 50-100 | Slow | Hard | Best |

## Multi-GPU Training

### Using Accelerate

1. **Create Accelerate config:**

```bash
accelerate config
```

Or use provided configs:
- [src/configs/accelerate_config.yaml](../src/configs/accelerate_config.yaml) - Multi-GPU
- [src/configs/accelerate_config_deepspeed.yaml](../src/configs/accelerate_config_deepspeed.yaml) - DeepSpeed ZeRO

2. **Launch training:**

```bash
accelerate launch --config_file src/configs/accelerate_config.yaml \
    scripts/train.py --config my_config.yaml
```

### Training Scripts

You can also use the provided bash training script:

```bash
# Single GPU
./run_training.sh single my_config.yaml

# Multi GPU
./run_training.sh multi my_config.yaml
```

## Monitoring

### Weights & Biases

Training automatically logs to W&B. Key metrics:

- `train/loss`: Total loss (LM + auxiliary)
- `train/lm_loss`: Language modeling loss
- `train/aux_loss`: Auxiliary attention loss
- `train/aux_accuracy`: Attention-based ranking accuracy
- `train/learning_rate`: Current learning rate

---

For more help, open an issue on [GitHub](https://github.com/nilesh2797/BlockRank/issues).
