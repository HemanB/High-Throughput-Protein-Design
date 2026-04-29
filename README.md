# High-Throughput Protein Design Pipeline

An automated pipeline for computational protein design integrating **RFDiffusion**, **ProteinMPNN**, and **AlphaFold 3**. Clone, configure, and run on any SLURM-managed HPC cluster.

## Overview

This pipeline automates the full protein design workflow:

1. **RFDiffusion** — Generates diverse protein backbone structures conditioned on a target binding site
2. **ProteinMPNN** — Designs amino acid sequences for each generated backbone
3. **AF3 Input Generation** — Converts designed sequences to AlphaFold 3 input format
4. **AlphaFold 3** — Predicts 3D structures of designed sequences and evaluates confidence

All configuration is centralized in a single JSON file. No hardcoded paths.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/HemanB/High-Throughput-Protein-Design.git
cd High-Throughput-Protein-Design

# 2. Setup (creates conda env, clones ProteinMPNN)
#    Requires ~8GB for the conda env (PyTorch + CUDA).
#    Use --conda-prefix to install to a directory with enough space.
./setup.sh --conda-prefix /cwork/$USER

# 3. Configure
#    Edit config.json with your paths (container images, databases, output dirs)
nano config.json

# 4. Activate environment
conda activate protein_design

# 5. Run (auto-submits to SLURM using config.json settings)
./pipeline.sh config.json
```

## Prerequisites

| Requirement | Details |
|---|---|
| **SLURM** | Job scheduler for HPC |
| **GPU** | NVIDIA GPU with CUDA support |
| **Singularity/Apptainer** | Container runtime (or Docker) |
| **Conda** | Miniconda or Anaconda |
| **RFDiffusion container** | `rfdiffusion_v1.1.0.sif` + model weights |
| **AlphaFold 3 container** | `AF3_v3.0.1.sif` + databases + model parameters |

## Installation

### Automated Setup

```bash
./setup.sh --conda-prefix /cwork/$USER   # Full setup (env installed to /cwork)
./setup.sh --skip-conda                  # Skip conda env (already exists)
./setup.sh --mpnn-path /path/to/ProteinMPNN  # Use existing install
```

> **Storage:** The conda environment requires ~8GB (PyTorch + CUDA). Use `--conda-prefix` to point to a directory with sufficient space. Without it, the environment installs to `~/.conda/envs/` which may exceed home directory quotas on HPC systems.

The setup script:
- Creates a conda environment from `environment.yml`
- Clones ProteinMPNN (or links to an existing installation)
- Creates `config.json` from the template
- Validates your environment (container runtime, GPU, jq)

### Manual Setup

```bash
conda env create -f environment.yml
conda activate protein_design
git clone https://github.com/dauparas/ProteinMPNN.git
cp config_template.json config.json
```

## Configuration

Edit `config.json` with your institution-specific paths. See `examples/example_config.json` for reference.

### Pipeline Settings

| Field | Description |
|---|---|
| `pipeline.base_dir` | Root directory for pipeline outputs |
| `pipeline.input_pdb` | Path to your target PDB structure |
| `pipeline.conda_env` | Conda environment name (default: `protein_design`) |
| `pipeline.container_runtime` | `singularity` or `docker` |
| `pipeline.cuda_module` | CUDA module to load (e.g., `CUDA/12.4`) |

### SLURM Settings

| Field | Description |
|---|---|
| `slurm.partition` | GPU partition name |
| `slurm.gres` | GPU resource specification |
| `slurm.mem` | Memory allocation |
| `slurm.time` | Wall time limit |
| `slurm.mail_user` | Email for job notifications |

### RFDiffusion

| Field | Description |
|---|---|
| `rfdiffusion.container_path` | Path to RFDiffusion `.sif` image |
| `rfdiffusion.model_path` | Path to RFDiffusion model weights |
| `rfdiffusion.num_designs` | Number of backbones to generate |
| `rfdiffusion.contigs` | Contig specification for design |
| `rfdiffusion.hotspots` | Hotspot residues for binding |

### ProteinMPNN

| Field | Description |
|---|---|
| `proteinmpnn.install_path` | Path to ProteinMPNN (use `./ProteinMPNN` for local) |
| `proteinmpnn.num_sequences` | Sequences per backbone |
| `proteinmpnn.designed_chain` | Chain ID being designed |
| `proteinmpnn.target_chain` | Chain ID of the binding target |
| `proteinmpnn.process_count` | Number of sequences to carry forward to AF3 |

### AlphaFold 3

| Field | Description |
|---|---|
| `alphafold3.container_path` | Path to AF3 `.sif` image |
| `alphafold3.database_path` | Path to AF3 sequence databases |
| `alphafold3.model_path` | Path to AF3 model parameters |
| `alphafold3.programs_path` | Path to AF3 `run_alphafold.py` directory |
| `alphafold3.num_seeds` | Model seeds per prediction |
| `alphafold3.num_samples` | Diffusion samples per seed |
| `alphafold3.use_templates` | Enable structural templates (`true`/`false`) |

## Pipeline Stages

### Stage 1: RFDiffusion

Generates protein backbone structures using RFDiffusion in a Singularity container. The input PDB defines the target binding site, and contigs/hotspots control the design space.

**Outputs:** `RFD/outputs/RFD_0.pdb`, `RFD_1.pdb`, ...

### Stage 2: ProteinMPNN

For each generated backbone, ProteinMPNN:
1. Parses PDB chain coordinates
2. Assigns designed vs. fixed chains
3. Identifies fixed positions (consecutive non-glycine stretches)
4. Designs sequences with specified diversity parameters

**Outputs:** `MPNN/outputs/seqs/RFD_0.fa`, `RFD_1.fa`, ...

### Stage 3: AF3 Input Generation

Converts MPNN FASTA output to AlphaFold 3 input JSON. Each designed sequence becomes a separate AF3 job with configurable seeds and samples.

**Outputs:** `AF3_INPUT/RFD_0/RFD_0_seq_1.json`, ...

### Stage 4: AlphaFold 3

Runs AF3 structure prediction for each designed sequence. Produces model structures (CIF), confidence metrics, and PAE matrices.

**Outputs:** `AF3/RFD_0/rfd_0_seq_1/seed-N_sample-M/` containing `*_model.cif`, `*_summary_confidences.json`, `*_confidences.json`

## Template Support

To use structural templates with AlphaFold 3:

1. Set `alphafold3.use_templates: true` in config
2. Provide template CIF files in `alphafold3.template_cif_dir`
3. The pipeline will align templates to query sequences and embed them in AF3 JSON

For manual template addition:

```bash
python scripts/add_templates_to_af3.py \
  --input_dir AF3_INPUT/ \
  --output_dir AF3_INPUT/ \
  --chain_template_map "A:/path/to/chain_a.cif,B:/path/to/chain_b.cif" \
  --release_date 2024-01-01
```

## Analysis

After the pipeline completes, run the analysis script:

```bash
python analysis/analysis.py \
  --rf_base_dir /path/to/run/RFD/outputs \
  --af_base_dir /path/to/run/AF3 \
  --output_dir /path/to/run/analysis
```

### Metrics

| Metric | Source | Description |
|---|---|---|
| **Ranking Score** | `summary_confidences.json` | AF3 overall confidence (higher = better) |
| **pLDDT** | `confidences.json` | Per-atom predicted local distance difference test |
| **pTM** | `summary_confidences.json` | Predicted template modeling score |
| **ipTM** | `summary_confidences.json` | Interface predicted TM score |
| **PAE** | `confidences.json` | Predicted aligned error matrix |
| **Clash Score** | Computed from CIF | Steric clashes (atom pairs < 2A) |
| **RMSD** | Computed vs. reference | Backbone RMSD with iterative outlier rejection |

### Outputs

- `plddt.tsv`, `ptm_iptm.tsv`, `pae.tsv`, `clash.tsv` — Individual metric tables
- `final_merged.tsv` — All metrics merged
- `top20_cifs/` — Top 20 model CIF files by ranking score
- `analysis_report.pdf` — PDF report with tables and scatter plots

## Output Structure

Each pipeline run creates a timestamped directory:

```
{base_dir}/{timestamp}/
├── config.json              # Copy of config used
├── RFD/
│   ├── inputs/              # Input PDB
│   └── outputs/             # Generated backbones (*.pdb)
├── MPNN/
│   ├── inputs/              # RFD PDBs
│   └── outputs/
│       ├── parsed_pdbs.jsonl
│       ├── assigned_pdbs.jsonl
│       ├── fixed_pdbs.jsonl
│       └── seqs/            # Designed sequences (*.fa)
├── AF3_INPUT/
│   └── RFD_N/               # AF3 JSON inputs per design
└── AF3/
    └── RFD_N/
        └── rfd_n_seq_m/
            └── seed-S_sample-T/
                ├── *_model.cif
                ├── *_summary_confidences.json
                └── *_confidences.json
```

## Docker Support

Set `pipeline.container_runtime` to `"docker"` in your config. The pipeline will use `docker run --gpus all` instead of `singularity run --nv`.

## Troubleshooting

| Issue | Solution |
|---|---|
| `jq: command not found` | `conda activate protein_design` or `conda install -c conda-forge jq` |
| `singularity: command not found` | Load your HPC's singularity module or install Apptainer |
| CUDA out of memory | Reduce `slurm.mem`, use a node with more VRAM, or reduce `num_designs` |
| ProteinMPNN not found | Run `./setup.sh` or set `proteinmpnn.install_path` correctly |
| AF3 template errors | Ensure `unpairedMsa` and `pairedMsa` are `""` (not null) when templates are set |
| Empty AF3 outputs | Check that AF3 container, databases, and model paths exist |

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Credits

- [RFDiffusion](https://github.com/RosettaCommons/RFdiffusion) — Watson et al.
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) — Dauparas et al.
- [AlphaFold 3](https://github.com/google-deepmind/alphafold3) — Abramson et al.
