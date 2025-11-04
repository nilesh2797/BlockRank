# BlockRank: Scalable In-context Ranking with Generative Models

[![Paper](https://img.shields.io/badge/Paper-arXiv:2510.05396-b31b1b.svg)](https://arxiv.org/abs/2510.05396)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow)](https://huggingface.co/quicktensor/blockrank-msmarco-mistral-7b)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nilesh2797/BlockRank/blob/main/quickstart.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

<p align="center">
  <img src="assets/blockrank_diagram.png" alt="BlockRank Architecture" width="650"/>
</p>

**BlockRank** makes LLMs efficient for document ranking by using structured sparse attention and attention-based inference, achieving **2-4× faster inference** with competitive accuracy on BEIR benchmarks.

## Key Features

- 🚀 **Linear Complexity**: O(n) attention instead of O(n²) through block-sparse patterns
- ⚡ **Fast Inference**: Allows skipping autoregressive decoding using attention scores directly
- 🎯 **Strong Performance**: Matches or outperforms state-of-the-art listwise rankers
- 🔧 **Easy Integration**: Existing LLMs (Qwen, Llama, etc) can be easily made a BlockRank model

## Installation

```bash
pip install git+https://github.com/nilesh2797/BlockRank.git
```

Or clone for development:

```bash
git clone https://github.com/nilesh2797/BlockRank.git
cd BlockRank
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Interactive Demo

Try the Colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nilesh2797/BlockRank/blob/main/quickstart.ipynb)

### As a Library

```python
import blockrank

# Import dataset utilities
from blockrank.dataset import load_icr_dataset_hf, calculate_accuracy

# Import attention modules
from blockrank import blockrank_std_attention
from blockrank import blockrank_triton_kernel_attention
# standard SDPA-based and torch compiled BlockRank
blockrank_std_attention.register_blockrank_attention(); 
# Triton-kernel based BlockRank - only supports inference at the moment!
blockrank_triton_kernel_attention.register_triton_blockrank_attention();

# Import training components
from blockrank.losses import compute_auxiliary_attention_loss
from blockrank.trainer import BlockRankAuxLossTrainer
```

### Training

```bash
# Prepare your data (JSONL format - see docs/DATA_FORMAT.md)
# Configure training (see src/configs/ for examples)

# Single GPU
python scripts/train.py --config your_config.yaml

# Multi-GPU
accelerate launch --config_file src/configs/accelerate_config.yaml \
    scripts/train.py --config your_config.yaml
```

Details in [docs/TRAINING.md](docs/TRAINING.md).

### Evaluation

```bash
# Fast attention-based inference (recommended)
python scripts/eval_attn.py \
    --config src/configs/eval_beir.yaml \
    --checkpoint your-model \
    --attn_layer 20

# Standard decode-based inference
python scripts/eval_decode.py \
    --config src/configs/eval_beir.yaml \
    --checkpoint your-model
```

## How It Works

BlockRank introduces three changes to standard transformer LLMs:

**1. Structured Sparse Attention**
Documents attend only to instructions and themselves (causal), while the query attends to all. Reduces complexity from O(n²) → O(n).

**2. Auxiliary Contrastive Loss**
Mid-layer InfoNCE loss on attention patterns strengthens query-document relevance signals:
```
L = L_lm + λ * L_aux
```

**3. Attention-Based Inference**
Extract relevance scores directly from attention maps during prefill stage:
```python
score_i = Σ attention[layer, head, query_token, doc_i_tokens]
```

## Model Zoo

| Model | Base | Training Data | Download |
|-------|------|---------------|----------|
| **BlockRank-Mistral-7B** | Mistral-7B-Instruct-v0.3 | 10% MS MARCO (50K) | [🤗 HuggingFace](https://huggingface.co/quicktensor/blockrank-msmarco-mistral-7b) |
| *More models coming soon...* | | | |

## Documentation

- **[Training Guide](docs/TRAINING.md)** - Detailed training instructions, hyperparameters, and best practices
- **[Data Format](docs/DATA_FORMAT.md)** - Data preparation and format specifications
- **[Paper](https://arxiv.org/abs/2510.05396)** - Full technical details and benchmarks

## Project Structure

```
BlockRank/
├── src/blockrank/          # Python package
│   ├── blockrank_std_attention.py      # PyTorch attention implementations
│   ├── blockrank_triton_kernel_attention.py  # Triton kernels (fastest, but inference-only)
│   ├── dataset.py          # Data loading and collation
│   ├── losses.py           # Auxiliary contrastive loss
│   ├── trainer.py          # Custom trainer with aux loss
│   └── utils.py            # Utilities (metrics, formatting)
├── scripts/                # CLI scripts
│   ├── train.py            # Training script
│   ├── eval_attn.py        # Attention-based evaluation
│   └── eval_decode.py      # Decode-based evaluation
├── configs/                # Training & eval configs
├── docs/                   # Documentation
├── data/                   # Downloaded datasets
└── quickstart.ipynb        # Quickstart notebook
```

## Citation

```bibtex
@article{gupta2025blockrank,
  title={Scalable In-context Ranking with Generative Models},
  author={Gupta, Nilesh and You, Chong and Bhojanapalli, Srinadh and Kumar, Sanjiv and Dhillon, Inderjit and Yu, Felix},
  journal={arXiv preprint arXiv:2510.05396},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- **Paper**: [arXiv:2510.05396](https://arxiv.org/abs/2510.05396)
- **Issues**: [GitHub Issues](https://github.com/nilesh2797/BlockRank/issues)
- **Author**: [Nilesh Gupta](https://nilesh2797.github.io/)

---

<p align="center">
  <b>⭐ Star us on GitHub if BlockRank is useful for your research!</b>
</p>
