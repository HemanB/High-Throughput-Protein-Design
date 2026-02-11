#!/bin/bash
###############################################################################
# High-Throughput Protein Design Pipeline
#
# Stages:
#   1. RFDiffusion   - Generate backbone structures
#   2. ProteinMPNN   - Design sequences for generated backbones
#   3. AF3 Input Gen - Convert MPNN output to AlphaFold 3 JSON
#   4. AlphaFold 3   - Predict structures of designed sequences
#
# Usage:
#   sbatch pipeline.sh /path/to/config.json
#   bash pipeline.sh /path/to/config.json   # for testing (non-SLURM)
###############################################################################
set -euo pipefail

# ── Parse config ─────────────────────────────────────────────────────────────
CONFIG_FILE="${1:-}"
if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
    echo "Usage: sbatch pipeline.sh /path/to/config.json"
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Resolve repo root (directory containing this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Read config values ───────────────────────────────────────────────────────
jq_get() { jq -r "$1" "$CONFIG_FILE"; }

# Pipeline
BASE_DIR=$(jq_get '.pipeline.base_dir')
INPUT_PDB=$(jq_get '.pipeline.input_pdb')
CONDA_ENV=$(jq_get '.pipeline.conda_env')
CONTAINER_RUNTIME=$(jq_get '.pipeline.container_runtime')
CUDA_MODULE=$(jq_get '.pipeline.cuda_module')

# RFDiffusion
RFD_SIF=$(jq_get '.rfdiffusion.container_path')
RFD_MODEL_PATH=$(jq_get '.rfdiffusion.model_path')
RFD_PWD=$(jq_get '.rfdiffusion.internal_pwd')
RFD_NUM_DESIGNS=$(jq_get '.rfdiffusion.num_designs')
RFD_CONTIGS=$(jq_get '.rfdiffusion.contigs')
RFD_HOTSPOTS=$(jq_get '.rfdiffusion.hotspots')
RFD_EXTRA_ARGS=$(jq_get '.rfdiffusion.extra_args')
RFD_TMP_DIR=$(jq_get '.rfdiffusion.tmp_dir')
RFD_CACHE_DIR=$(jq_get '.rfdiffusion.cache_dir')

# ProteinMPNN
MPNN_INSTALL_PATH=$(jq_get '.proteinmpnn.install_path')
MPNN_NUM_SEQUENCES=$(jq_get '.proteinmpnn.num_sequences')
MPNN_BACKBONE_NOISE=$(jq_get '.proteinmpnn.backbone_noise')
MPNN_SAMPLING_TEMP=$(jq_get '.proteinmpnn.sampling_temp')
MPNN_SEED=$(jq_get '.proteinmpnn.seed')
MPNN_BATCH_SIZE=$(jq_get '.proteinmpnn.batch_size')
MPNN_DESIGNED_CHAIN=$(jq_get '.proteinmpnn.designed_chain')
MPNN_TARGET_CHAIN=$(jq_get '.proteinmpnn.target_chain')
MPNN_CHAINS_TO_DESIGN=$(jq_get '.proteinmpnn.chains_to_design')
MPNN_PROCESS_COUNT=$(jq_get '.proteinmpnn.process_count')

# AlphaFold 3
AF3_SIF=$(jq_get '.alphafold3.container_path')
AF3_DB_PATH=$(jq_get '.alphafold3.database_path')
AF3_MODEL_PATH=$(jq_get '.alphafold3.model_path')
AF3_PROGRAMS_PATH=$(jq_get '.alphafold3.programs_path')
AF3_NUM_SEEDS=$(jq_get '.alphafold3.num_seeds')
AF3_NUM_SAMPLES=$(jq_get '.alphafold3.num_samples')
AF3_USE_TEMPLATES=$(jq_get '.alphafold3.use_templates')
AF3_TEMPLATE_CIF_DIR=$(jq_get '.alphafold3.template_cif_dir')
AF3_TEMPLATE_DATE=$(jq_get '.alphafold3.template_release_date')
AF3_TMP_DIR=$(jq_get '.alphafold3.tmp_dir')

# ── Resolve ProteinMPNN path ────────────────────────────────────────────────
if [[ "$MPNN_INSTALL_PATH" == ./* ]]; then
    MPNN_INSTALL_PATH="${REPO_ROOT}/${MPNN_INSTALL_PATH#./}"
fi
MPNN_SCRIPT="${MPNN_INSTALL_PATH}/protein_mpnn_run.py"

# ── Container runtime helper ────────────────────────────────────────────────
container_run() {
    if [[ "$CONTAINER_RUNTIME" == "docker" ]]; then
        docker run --gpus all "$@"
    else
        singularity run --nv "$@"
    fi
}

container_exec() {
    if [[ "$CONTAINER_RUNTIME" == "docker" ]]; then
        docker run --gpus all "$@"
    else
        singularity exec --nv "$@"
    fi
}

# ── Error handler ────────────────────────────────────────────────────────────
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "ERROR: Pipeline failed at $(date) with exit code $exit_code"
        echo "Check the output above for details."
    fi
}
trap cleanup EXIT

check_error() {
    if [[ $? -ne 0 ]]; then
        echo "Error in step: $1"
        exit 1
    fi
}

# ── Initialize environment ───────────────────────────────────────────────────
echo "============================================================"
echo "High-Throughput Protein Design Pipeline"
echo "============================================================"
echo "Config:    $CONFIG_FILE"
echo "Start:     $(date)"
echo "============================================================"

if command -v module &>/dev/null; then
    module load "$CUDA_MODULE" 2>/dev/null || echo "Warning: Could not load $CUDA_MODULE"
fi

mkdir -p "$RFD_TMP_DIR" "$RFD_CACHE_DIR"
export APPTAINER_TMPDIR="$RFD_TMP_DIR"
export APPTAINER_CACHEDIR="$RFD_CACHE_DIR"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# ── Validate inputs ─────────────────────────────────────────────────────────
if [[ ! -f "$INPUT_PDB" ]]; then
    echo "Error: Input PDB file not found: $INPUT_PDB"
    exit 1
fi

if [[ ! -f "$MPNN_SCRIPT" ]]; then
    echo "Error: ProteinMPNN script not found: $MPNN_SCRIPT"
    echo "Run setup.sh first or set proteinmpnn.install_path in config."
    exit 1
fi

# ── Create output directory structure ────────────────────────────────────────
timestamp=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$BASE_DIR/${timestamp}"
RFD_INPUT_DIR="$RUN_DIR/RFD/inputs"
RFD_OUTPUT_DIR="$RUN_DIR/RFD/outputs"
MPNN_INPUT_DIR="$RUN_DIR/MPNN/inputs"
MPNN_OUTPUT_DIR="$RUN_DIR/MPNN/outputs"
AF3_INPUT_DIR="$RUN_DIR/AF3_INPUT"
AF3_OUTPUT_DIR="$RUN_DIR/AF3"

mkdir -p "$RFD_INPUT_DIR" "$RFD_OUTPUT_DIR" \
         "$MPNN_INPUT_DIR" "$MPNN_OUTPUT_DIR" \
         "$AF3_INPUT_DIR" "$AF3_OUTPUT_DIR"

echo ""
echo "Run directory: $RUN_DIR"
echo ""

# Copy config to run directory for reproducibility
cp "$CONFIG_FILE" "$RUN_DIR/config.json"

# Copy input PDB
rsync "$INPUT_PDB" "$RFD_INPUT_DIR/" || { echo "Error syncing input PDB file."; exit 1; }

###############################################################################
# STAGE 1: RFDiffusion
###############################################################################
echo "============================================================"
echo "STAGE 1: RFDiffusion"
echo "============================================================"

container_run \
    --env TF_FORCE_UNIFIED_MEMORY=1,XLA_PYTHON_CLIENT_MEM_FRACTION=4.0,OPENMM_CPU_THREADS=10,HYDRA_FULL_ERROR=1 \
    -B "$RFD_OUTPUT_DIR:/outputs,$RFD_INPUT_DIR:/inputs,$RFD_MODEL_PATH:/models,$RFD_SIF:/sif" \
    --pwd "$RFD_PWD" \
    "$RFD_SIF" \
    hydra.run.dir=/outputs \
    inference.schedule_directory_path=/outputs \
    inference.output_prefix=/outputs/RFD \
    inference.model_directory_path=/models \
    inference.input_pdb="/inputs/$(basename "$INPUT_PDB")" \
    inference.num_designs="$RFD_NUM_DESIGNS" \
    "contigmap.contigs=$RFD_CONTIGS" \
    "ppi.hotspot_res=$RFD_HOTSPOTS" \
    $RFD_EXTRA_ARGS
check_error "RFDiffusion"

echo "RFDiffusion completed. Outputs in $RFD_OUTPUT_DIR"

# Copy RFD outputs to MPNN inputs
rsync "$RFD_OUTPUT_DIR"/*.pdb "$MPNN_INPUT_DIR/"

###############################################################################
# STAGE 2: ProteinMPNN
###############################################################################
echo ""
echo "============================================================"
echo "STAGE 2: ProteinMPNN"
echo "============================================================"

for i in $(seq 0 $((RFD_NUM_DESIGNS - 1))); do
    input_pdb="$MPNN_INPUT_DIR/RFD_${i}.pdb"
    if [[ ! -f "$input_pdb" ]]; then
        echo "Warning: Expected PDB not found: $input_pdb, skipping."
        continue
    fi

    rfd_output_num=$(basename "$input_pdb" .pdb | sed 's/RFD_//')

    # Find non-glycine residues on the designed chain for fixed positions
    non_gly_residues=$(awk -v chain="$MPNN_DESIGNED_CHAIN" \
        '$5 == chain && $4 != "GLY" {print $6}' "$input_pdb" | sort -u -n)
    unique_non_gly_residues=($(echo "${non_gly_residues[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))

    find_consecutive_sequences() {
        local arr=("$@")
        local start=${arr[0]}
        local end=$start
        local seq=($start)
        local result=()
        for val in "${arr[@]:1}"; do
            if (( val == end + 1 )); then
                end=$val
                seq+=($val)
            else
                if (( ${#seq[@]} >= 5 )); then
                    result+=("${seq[@]}")
                fi
                start=$val
                end=$val
                seq=($val)
            fi
        done
        if (( ${#seq[@]} >= 5 )); then
            result+=("${seq[@]}")
        fi
        echo "${result[@]}"
    }

    consecutive_non_gly_residues=$(find_consecutive_sequences "${unique_non_gly_residues[@]}")
    first_residue_target=$(awk -v chain="$MPNN_TARGET_CHAIN" \
        '$5 == chain {print $6}' "$input_pdb" | sort -n | head -n 1)
    chain_residues=$(awk -v chain="$MPNN_TARGET_CHAIN" -v first="$first_residue_target" \
        '$5 == chain {print $6 - first + 1}' "$input_pdb" | sort -u -n | tr '\n' ' ' | sed 's/ $//')
    fixed_positions_list="${consecutive_non_gly_residues}, ${chain_residues}"

    echo "Running ProteinMPNN for RFD_${i}..."
    parsed_chains="$MPNN_OUTPUT_DIR/parsed_pdbs.jsonl"
    assigned_chains="$MPNN_OUTPUT_DIR/assigned_pdbs.jsonl"
    fixed_positions="$MPNN_OUTPUT_DIR/fixed_pdbs.jsonl"

    python "$REPO_ROOT/scripts/parse_multiple_chains.py" \
        --input_path="$MPNN_INPUT_DIR" \
        --output_path="$parsed_chains"
    check_error "parse_multiple_chains"

    python "$REPO_ROOT/scripts/assign_fixed_chains.py" \
        --input_path="$parsed_chains" \
        --output_path="$assigned_chains" \
        --chain_list "$MPNN_CHAINS_TO_DESIGN"
    check_error "assign_fixed_chains"

    python "$REPO_ROOT/scripts/make_fixed_positions_dict.py" \
        --input_path="$parsed_chains" \
        --output_path="$fixed_positions" \
        --chain_list "$MPNN_CHAINS_TO_DESIGN" \
        --position_list "$fixed_positions_list"
    check_error "make_fixed_positions_dict"

    python "$MPNN_SCRIPT" \
        --jsonl_path "$parsed_chains" \
        --chain_id_jsonl "$assigned_chains" \
        --fixed_positions_jsonl "$fixed_positions" \
        --out_folder "$MPNN_OUTPUT_DIR" \
        --num_seq_per_target "$MPNN_NUM_SEQUENCES" \
        --backbone_noise "$MPNN_BACKBONE_NOISE" \
        --sampling_temp "$MPNN_SAMPLING_TEMP" \
        --seed "$MPNN_SEED" \
        --batch_size "$MPNN_BATCH_SIZE"
    check_error "ProteinMPNN"
done

echo "ProteinMPNN completed. Outputs in $MPNN_OUTPUT_DIR"

###############################################################################
# STAGE 3: Generate AF3 Input JSON
###############################################################################
echo ""
echo "============================================================"
echo "STAGE 3: Generate AlphaFold 3 Input JSON"
echo "============================================================"

MPNN_SEQS_DIR="$MPNN_OUTPUT_DIR/seqs"
if [[ ! -d "$MPNN_SEQS_DIR" ]]; then
    echo "Error: MPNN sequences directory not found: $MPNN_SEQS_DIR"
    exit 1
fi

python "$REPO_ROOT/scripts/generate_af3_inputs.py" \
    --mpnn_fasta_dir "$MPNN_SEQS_DIR" \
    --output_dir "$AF3_INPUT_DIR" \
    --num_seeds "$AF3_NUM_SEEDS" \
    --num_samples "$AF3_NUM_SAMPLES" \
    --designed_chain_id "$MPNN_DESIGNED_CHAIN" \
    --target_chain_id "$MPNN_TARGET_CHAIN" \
    --process_count "$MPNN_PROCESS_COUNT" \
    --config "$CONFIG_FILE"
check_error "generate_af3_inputs"

# Optionally add templates
if [[ "$AF3_USE_TEMPLATES" == "true" && -n "$AF3_TEMPLATE_CIF_DIR" ]]; then
    echo "Adding templates to AF3 inputs..."
    python "$REPO_ROOT/scripts/add_templates_to_af3.py" \
        --input_dir "$AF3_INPUT_DIR" \
        --output_dir "$AF3_INPUT_DIR" \
        --chain_template_map "$AF3_TEMPLATE_CIF_DIR" \
        --release_date "$AF3_TEMPLATE_DATE"
    check_error "add_templates_to_af3"
fi

echo "AF3 input generation completed. Inputs in $AF3_INPUT_DIR"

###############################################################################
# STAGE 4: AlphaFold 3
###############################################################################
echo ""
echo "============================================================"
echo "STAGE 4: AlphaFold 3"
echo "============================================================"

mkdir -p "$AF3_TMP_DIR"
export TMPDIR="$AF3_TMP_DIR"
export TEMP="$AF3_TMP_DIR"
export TMP="$AF3_TMP_DIR"

# Find all AF3 input JSON files
af3_json_files=$(find "$AF3_INPUT_DIR" -name "*.json" -type f | sort)
af3_count=$(echo "$af3_json_files" | wc -l)
echo "Running AF3 on $af3_count input files..."

for json_file in $af3_json_files; do
    json_basename=$(basename "$json_file")
    rfd_subdir=$(basename "$(dirname "$json_file")")
    af3_rfd_output="$AF3_OUTPUT_DIR/$rfd_subdir"
    mkdir -p "$af3_rfd_output"

    echo ""
    echo "  Processing: $rfd_subdir/$json_basename"

    container_exec \
        --bind "$AF3_INPUT_DIR/$rfd_subdir:/root/af_input" \
        --bind "$af3_rfd_output:/root/af_output" \
        --bind "$AF3_MODEL_PATH:/root/models" \
        --bind "$AF3_DB_PATH:/root/public_databases" \
        --bind "$AF3_PROGRAMS_PATH:/root/AF3" \
        --bind "$AF3_TMP_DIR:/tmp" \
        "$AF3_SIF" \
        python /root/AF3/run_alphafold.py \
            --json_path="/root/af_input/$json_basename" \
            --model_dir=/root/models \
            --db_dir=/root/public_databases \
            --output_dir=/root/af_output
    check_error "AlphaFold 3: $json_basename"

    echo "  Completed: $rfd_subdir/$json_basename"
done

###############################################################################
# Summary
###############################################################################
echo ""
echo "============================================================"
echo "Pipeline Complete"
echo "============================================================"
echo "Run directory: $RUN_DIR"
echo "  RFD outputs:   $RFD_OUTPUT_DIR"
echo "  MPNN outputs:  $MPNN_OUTPUT_DIR"
echo "  AF3 inputs:    $AF3_INPUT_DIR"
echo "  AF3 outputs:   $AF3_OUTPUT_DIR"
echo "End time:        $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  python analysis/analysis.py \\"
echo "    --rf_base_dir $RFD_OUTPUT_DIR \\"
echo "    --af_base_dir $AF3_OUTPUT_DIR \\"
echo "    --output_dir $RUN_DIR/analysis"
