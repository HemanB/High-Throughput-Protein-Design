#!/bin/bash
###############################################################################
# Setup script for High-Throughput Protein Design Pipeline
#
# Usage:
#   ./setup.sh                          # Full setup (conda env + ProteinMPNN)
#   ./setup.sh --conda-prefix /cwork/netid  # Install conda env to a specific dir
#   ./setup.sh --skip-conda             # Skip conda env creation
#   ./setup.sh --mpnn-path /path/to/ProteinMPNN  # Use existing install
###############################################################################
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
SKIP_CONDA=false
FORCE=false
MPNN_PATH=""
CONDA_PREFIX_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-conda)
            SKIP_CONDA=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        --mpnn-path)
            MPNN_PATH="$2"
            shift 2
            ;;
        --conda-prefix)
            CONDA_PREFIX_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --conda-prefix DIR        Install conda env and cache to DIR"
            echo "                            (recommended: your /cwork/<netid> directory)"
            echo "  --skip-conda              Skip conda environment creation"
            echo "  --force, -f               Recreate conda env if it already exists (no prompt)"
            echo "  --mpnn-path PATH          Path to existing ProteinMPNN installation"
            echo "  -h, --help                Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./setup.sh --conda-prefix /cwork/\$USER"
            echo "  ./setup.sh --skip-conda --mpnn-path /path/to/ProteinMPNN"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "High-Throughput Protein Design Pipeline - Setup"
echo "============================================================"
echo ""

# ── Step 1: Conda environment ───────────────────────────────────────────────
if [[ "$SKIP_CONDA" == "false" ]]; then
    echo "Step 1: Creating conda environment..."

    # Find conda — try the function first, then common module, then PATH
    if ! command -v conda &>/dev/null; then
        # Try loading the Anaconda module (Duke HPC)
        if command -v module &>/dev/null; then
            module load Anaconda3 2>/dev/null || true
        fi
    fi
    if ! command -v conda &>/dev/null; then
        echo "Error: conda not found. Load your Anaconda module or install Miniconda first."
        echo "  On Duke HPC: module load Anaconda3/2024.02"
        exit 1
    fi

    eval "$(conda shell.bash hook)"

    # ── Redirect conda dirs if --conda-prefix is set ─────────────────────
    if [[ -n "$CONDA_PREFIX_DIR" ]]; then
        mkdir -p "$CONDA_PREFIX_DIR/.conda/pkgs" "$CONDA_PREFIX_DIR/.conda/envs"
        conda config --add pkgs_dirs "$CONDA_PREFIX_DIR/.conda/pkgs"
        conda config --add envs_dirs "$CONDA_PREFIX_DIR/.conda/envs"
        echo "  Conda packages: $CONDA_PREFIX_DIR/.conda/pkgs"
        echo "  Conda envs:     $CONDA_PREFIX_DIR/.conda/envs"
    fi

    # ── Create the environment ───────────────────────────────────────────
    if conda env list | grep -q "protein_design"; then
        if [[ "$FORCE" == "true" ]]; then
            echo "  Environment 'protein_design' already exists. Recreating (--force)..."
            conda env remove -n protein_design -y
            conda env create -f "$REPO_ROOT/environment.yml"
        elif [[ -t 0 ]]; then
            echo "  Environment 'protein_design' already exists."
            read -p "  Recreate it? (y/N): " choice
            if [[ "$choice" =~ ^[Yy]$ ]]; then
                conda env remove -n protein_design -y
                conda env create -f "$REPO_ROOT/environment.yml"
            else
                echo "  Skipping environment creation."
            fi
        else
            echo "  Environment 'protein_design' already exists. Skipping."
            echo "  (Use --force to recreate in non-interactive mode.)"
        fi
    else
        echo "  Creating environment (this may take several minutes)..."
        conda env create -f "$REPO_ROOT/environment.yml"
    fi
    echo "  Conda environment ready."
else
    echo "Step 1: Skipping conda environment creation (--skip-conda)."
fi
echo ""

# ── Step 2: ProteinMPNN ─────────────────────────────────────────────────────
echo "Step 2: Setting up ProteinMPNN..."

if [[ -n "$MPNN_PATH" ]]; then
    if [[ ! -d "$MPNN_PATH" ]]; then
        echo "Error: ProteinMPNN path does not exist: $MPNN_PATH"
        exit 1
    fi
    if [[ ! -f "$MPNN_PATH/protein_mpnn_run.py" ]]; then
        echo "Error: protein_mpnn_run.py not found in $MPNN_PATH"
        exit 1
    fi
    echo "  Using existing ProteinMPNN at: $MPNN_PATH"
    echo "  Set 'proteinmpnn.install_path' in your config to: $MPNN_PATH"
else
    MPNN_TARGET="$REPO_ROOT/ProteinMPNN"
    if [[ -d "$MPNN_TARGET" ]]; then
        echo "  ProteinMPNN already exists at $MPNN_TARGET"
    else
        echo "  Cloning ProteinMPNN..."
        git clone https://github.com/dauparas/ProteinMPNN.git "$MPNN_TARGET"
    fi
    echo "  ProteinMPNN ready at: $MPNN_TARGET"
    echo "  Config default './ProteinMPNN' will resolve correctly."
fi
echo ""

# ── Step 3: Create user config ──────────────────────────────────────────────
echo "Step 3: Config file..."
if [[ ! -f "$REPO_ROOT/config.json" ]]; then
    cp "$REPO_ROOT/config_template.json" "$REPO_ROOT/config.json"
    echo "  Created config.json from template."
    echo "  IMPORTANT: Edit config.json with your paths before running the pipeline."
else
    echo "  config.json already exists. Skipping."
fi
echo ""

# ── Step 4: Validate environment ────────────────────────────────────────────
echo "Step 4: Validating environment..."

# Check container runtime
if command -v singularity &>/dev/null; then
    echo "  Singularity: $(singularity --version)"
elif command -v apptainer &>/dev/null; then
    echo "  Apptainer: $(apptainer --version)"
elif command -v docker &>/dev/null; then
    echo "  Docker: $(docker --version)"
else
    echo "  WARNING: No container runtime found (singularity/apptainer/docker)."
    echo "  You need one installed to run RFDiffusion and AlphaFold 3."
fi

# Check jq
if command -v jq &>/dev/null; then
    echo "  jq: $(jq --version)"
else
    echo "  WARNING: jq not found. It will be available after: conda activate protein_design"
fi

# Check GPU
if command -v nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    if [[ -n "$gpu_name" ]]; then
        echo "  GPU: $gpu_name"
    else
        echo "  GPU: nvidia-smi found but no GPU detected (expected on login nodes)."
    fi
else
    echo "  GPU: nvidia-smi not found (expected on login nodes without GPU)."
fi

echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Setup Complete"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Edit config.json with your institution-specific paths"
echo "  2. Activate the environment:  conda activate protein_design"
echo "  3. Run the pipeline:          ./pipeline.sh config.json"
echo ""
echo "See README.md for detailed configuration instructions."
echo "See examples/example_config.json for a reference configuration."
