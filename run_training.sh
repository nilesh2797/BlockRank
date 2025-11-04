#!/bin/bash

# Training script launcher for Mistral-7B ICR fine-tuning
# Usage:
#   ./run_training.sh single configs/default.yaml          # Single GPU
#   ./run_training.sh multi configs/default.yaml           # Multi-GPU
#   ./run_training.sh multi configs/large_scale.yaml       # Multi-GPU with large dataset

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODE=${1:-single}
CONFIG=${2:-configs/train_full_blockrank_10p_msmsarco.yaml}
ACCELERATE_CONFIG=${3:-configs/accelerate_config.yaml}

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    print_error "Config file not found: $CONFIG"
    exit 1
fi

print_info "Starting training with configuration: $CONFIG"
print_info "Mode: $MODE"

# Run training based on mode
if [ "$MODE" == "single" ]; then
    print_info "Running single-GPU training..."
    python scripts/train.py --config "$CONFIG"

elif [ "$MODE" == "multi" ]; then
    print_info "Running multi-GPU training with accelerate..."

    # Check if accelerate is installed
    if ! command -v accelerate &> /dev/null; then
        print_error "accelerate not found. Please install it: pip install accelerate"
        exit 1
    fi

    # Check if accelerate config exists
    if [ ! -f "$ACCELERATE_CONFIG" ]; then
        print_warning "Accelerate config not found: $ACCELERATE_CONFIG"
        print_info "Creating default accelerate config..."
        accelerate config default
        ACCELERATE_CONFIG="default_config.yaml"
    fi

    print_info "Using accelerate config: $ACCELERATE_CONFIG"
    accelerate launch --config_file "$ACCELERATE_CONFIG" scripts/train.py --config "$CONFIG"

else
    print_error "Invalid mode: $MODE"
    echo "Usage: $0 {single|multi} [config_file] [accelerate_config]"
    echo "Examples:"
    echo "  $0 single configs/default.yaml"
    echo "  $0 multi configs/default.yaml accelerate_config_simple.yaml"
    exit 1
fi

print_info "Training completed successfully!"
