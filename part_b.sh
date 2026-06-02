#!/usr/bin/env bash
# =============================================================================
# run_distributed.sh — Multi-GPU launcher for train.py using PyTorch torchrun
# Part B: Option 2 — PyTorch Torchrun
#
# Usage:
#   ./run_distributed.sh [OPTIONS]
#
# Pass-through options are forwarded directly to train.py:
#   --epochs     INT    Number of training epochs          (default: 5)
#   --batch-size INT    Per-GPU batch size                 (default: 64)
#   --lr         FLOAT  Learning rate                      (default: 1e-3)
#   --data       PATH   Path to dataset root               (default: ./data)
#   --output-dir PATH   Where to save checkpoints/logs     (default: ./outputs)
#   --tracker    STR    Experiment tracker: wandb|mlflow|none (default: none)
#
# Examples:
#   ./run_distributed.sh
#   ./run_distributed.sh --epochs 10 --batch-size 128 --lr 5e-4 --tracker wandb
#   ./run_distributed.sh --epochs 3 --output-dir ./my_run
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Colour helpers (gracefully degraded if terminal doesn't support colours)
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'  # No Colour

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
header()  { echo -e "${BOLD}$*${NC}"; }

# -----------------------------------------------------------------------------
# Default train.py arguments (overridden by CLI flags below)
# -----------------------------------------------------------------------------
EPOCHS=5
BATCH_SIZE=64
LR=1e-3
DATA="./data"
OUTPUT_DIR="./outputs"
TRACKER="none"

# -----------------------------------------------------------------------------
# Argument parsing — collect known flags; anything unrecognised is passed through
# -----------------------------------------------------------------------------
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            EPOCHS="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --lr)
            LR="$2"; shift 2 ;;
        --data)
            DATA="$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --tracker)
            TRACKER="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,20p' "$0"  # Print the usage block at the top of this file
            exit 0 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# -----------------------------------------------------------------------------
# Validate arguments
# -----------------------------------------------------------------------------
if ! [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    error "--epochs must be a positive integer (got: $EPOCHS)"
    exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    error "--batch-size must be a positive integer (got: $BATCH_SIZE)"
    exit 1
fi

if ! python3 -c "import sys; v=float('$LR'); sys.exit(0 if v > 0 else 1)" 2>/dev/null; then
    error "--lr must be a positive float (got: $LR)"
    exit 1
fi

if [[ "$TRACKER" != "wandb" && "$TRACKER" != "mlflow" && "$TRACKER" != "none" ]]; then
    error "--tracker must be one of: wandb, mlflow, none (got: $TRACKER)"
    exit 1
fi

# -----------------------------------------------------------------------------
# Dependency checks
# -----------------------------------------------------------------------------
check_dep() {
    if ! command -v "$1" &>/dev/null; then
        error "Required dependency '$1' not found."
        echo "  Install PyTorch >= 1.10 which ships torchrun:"
        echo "    pip install torch torchvision"
        exit 1
    fi
}

check_dep python3
check_dep torchrun

if [[ ! -f "part_A.py" && ! -f "train.py" ]]; then
    error "Training script not found. Expected 'train.py' (or 'part_A.py') in the current directory."
    exit 1
fi

# Use part_A.py if train.py doesn't exist (assessment context)
TRAIN_SCRIPT="train.py"
[[ ! -f "train.py" ]] && TRAIN_SCRIPT="part_A.py"

# -----------------------------------------------------------------------------
# GPU / device detection
# -----------------------------------------------------------------------------
GPU_COUNT=$(python3 - <<'EOF'
import torch
print(torch.cuda.device_count())
EOF
)

if ! [[ "$GPU_COUNT" =~ ^[0-9]+$ ]]; then
    warn "Could not query GPU count via PyTorch — defaulting to CPU single-process mode."
    GPU_COUNT=0
fi

# Determine world size (number of processes to launch)
if [[ "$GPU_COUNT" -ge 2 ]]; then
    WORLD_SIZE=$GPU_COUNT
    DEVICE_LABEL="${GPU_COUNT}x GPU"
    MODE="distributed"
elif [[ "$GPU_COUNT" -eq 1 ]]; then
    WORLD_SIZE=1
    DEVICE_LABEL="1x GPU (single process)"
    MODE="single"
    warn "Only 1 GPU detected — launching single-process training (no DDP)."
else
    WORLD_SIZE=1
    DEVICE_LABEL="CPU (no CUDA available)"
    MODE="cpu"
    warn "No CUDA GPUs detected — falling back to single-process CPU training."
fi

# -----------------------------------------------------------------------------
# Logging setup — write to runs/ directory with timestamps
# -----------------------------------------------------------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="runs/${TIMESTAMP}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/launcher.log"

# Tee all subsequent output to the log file as well as stdout
exec > >(tee -a "$LOG_FILE") 2>&1

# -----------------------------------------------------------------------------
# Run header
# -----------------------------------------------------------------------------
echo ""
header "============================================================"
header "   PyTorch torchrun Distributed Launcher"
header "============================================================"
echo ""
info "Timestamp    : $(date '+%Y-%m-%d %H:%M:%S')"
info "Device       : ${DEVICE_LABEL}"
info "World size   : ${WORLD_SIZE}"
info "Mode         : ${MODE}"
echo ""
info "Training script : ${TRAIN_SCRIPT}"
info "Epochs          : ${EPOCHS}"
info "Batch size      : ${BATCH_SIZE} (per GPU)"
info "Learning rate   : ${LR}"
info "Data path       : ${DATA}"
info "Output dir      : ${OUTPUT_DIR}"
info "Tracker         : ${TRACKER}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && info "Extra args      : ${EXTRA_ARGS[*]}"
echo ""
info "Log file        : ${LOG_FILE}"
header "------------------------------------------------------------"
echo ""

# -----------------------------------------------------------------------------
# Launch training via torchrun
# -----------------------------------------------------------------------------
# torchrun is the recommended replacement for torch.distributed.launch (PyTorch >= 1.10).
# --nproc_per_node sets the number of worker processes per machine.
# For single-GPU / CPU fallback, nproc_per_node=1 gives identical behaviour to
# running python directly but goes through the same code path for consistency.

TORCHRUN_CMD=(
    torchrun
    --nproc_per_node="${WORLD_SIZE}"
    --standalone                # single-node run; no rendezvous server needed
    --nnodes=1
    "${TRAIN_SCRIPT}"
    --epochs       "${EPOCHS}"
    --batch-size   "${BATCH_SIZE}"
    --lr           "${LR}"
    --data         "${DATA}"
    --output-dir   "${OUTPUT_DIR}"
    --tracker      "${TRACKER}"
)

# Append any unrecognised pass-through arguments
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    TORCHRUN_CMD+=("${EXTRA_ARGS[@]}")
fi

info "Launching: ${TORCHRUN_CMD[*]}"
echo ""

# -----------------------------------------------------------------------------
# Execute with error handling
# -----------------------------------------------------------------------------
set +e  # Temporarily disable 'exit on error' so we can handle it ourselves
"${TORCHRUN_CMD[@]}"
EXIT_CODE=$?
set -e

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    success "Training completed successfully."
    success "Checkpoints and metrics saved to: ${OUTPUT_DIR}"
    success "Launcher log saved to            : ${LOG_FILE}"
else
    error "Training exited with code ${EXIT_CODE}."
    echo ""
    header "Troubleshooting guidance:"
    echo "  • CUDA out of memory  → reduce --batch-size (try half the current value)"
    echo "  • NCCL timeout        → check GPU interconnect, reduce --nproc_per_node"
    echo "  • Missing dependency  → run: pip install -r requirements.txt"
    echo "  • torchrun not found  → upgrade PyTorch: pip install --upgrade torch"
    echo "  • Permission denied   → run: chmod +x run_distributed.sh"
    echo ""
    echo "  Full log available at: ${LOG_FILE}"
    exit $EXIT_CODE
fi

echo ""
header "============================================================"
header "   Run complete — $(date '+%Y-%m-%d %H:%M:%S')"
header "============================================================"
echo ""