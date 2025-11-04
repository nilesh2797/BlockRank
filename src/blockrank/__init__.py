"""
BlockRank: Scalable In-context Ranking with Generative Models

A library for efficient document ranking with LLMs using structured sparse attention
and auxiliary contrastive loss.
"""

__version__ = "0.1.0"

# Import main modules
from . import blockrank_std_attention
from . import blockrank_triton_kernel_attention
from . import dataset
from . import losses
from . import utils
from . import trainer

# Import commonly used functions and classes
from .dataset import (
    load_icr_dataset_hf,
    icr_collate_fn,
    block_icr_collate_fn,
)

from .utils import (
    format_ranking_prompt,
    remap_documents,
    calculate_accuracy,
    load_qrels,
    parse_predicted_id,
)

from .losses import (
    compute_auxiliary_attention_loss,
)

from .trainer import (
    BlockRankAuxLossTrainer,
)

# Export all for "from blockrank import *"
__all__ = [
    # Modules
    "blockrank_std_attention",
    "blockrank_triton_kernel_attention",
    "dataset",
    "losses",
    "utils",
    "trainer",
    # Functions
    "load_icr_dataset_hf",
    "icr_collate_fn",
    "block_icr_collate_fn",
    "format_ranking_prompt",
    "remap_documents",
    "calculate_accuracy",
    "load_qrels",
    "parse_predicted_id",
    "compute_auxiliary_attention_loss",
    "BlockRankAuxLossTrainer",
]
