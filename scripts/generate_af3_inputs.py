#!/usr/bin/env python3
"""
Generate AlphaFold 3 input JSON files from ProteinMPNN FASTA output.

Reads MPNN output FASTA files, parses the designed/target chain sequences
separated by "/", and produces AF3-compatible JSON for each sequence.

Usage:
  python generate_af3_inputs.py \
    --mpnn_fasta_dir /path/to/MPNN/outputs/seqs \
    --output_dir /path/to/AF3_INPUT \
    --num_seeds 5 \
    --num_samples 5 \
    --designed_chain_id A \
    --target_chain_id B

  # With template support:
  python generate_af3_inputs.py \
    --mpnn_fasta_dir /path/to/MPNN/outputs/seqs \
    --output_dir /path/to/AF3_INPUT \
    --num_seeds 5 \
    --num_samples 5 \
    --designed_chain_id A \
    --target_chain_id B \
    --config /path/to/config.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_mpnn_fasta(fasta_path):
    """Parse a ProteinMPNN output FASTA file.

    MPNN FASTA format:
      >RFD_0, score=..., ...
      DESIGNED_CHAIN_SEQ/TARGET_CHAIN_SEQ
      >T=0.1, sample=1, score=..., ...
      DESIGNED_CHAIN_SEQ/TARGET_CHAIN_SEQ
      ...

    The first entry is the native/input sequence. Subsequent entries are
    designed sequences. Chains are separated by "/".

    Returns:
        list of (header, chain_sequences) tuples, where chain_sequences
        is a list of sequences split by "/".
    """
    entries = []
    current_header = None
    current_seq_lines = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    full_seq = "".join(current_seq_lines)
                    chains = full_seq.split("/")
                    entries.append((current_header, chains))
                current_header = line[1:]  # strip ">"
                current_seq_lines = []
            else:
                current_seq_lines.append(line)

    # Last entry
    if current_header is not None:
        full_seq = "".join(current_seq_lines)
        chains = full_seq.split("/")
        entries.append((current_header, chains))

    return entries


def build_af3_json(name, chain_sequences, chain_ids, num_seeds, num_samples):
    """Build an AF3-compatible input JSON dict.

    Args:
        name: Job name (e.g. "RFD_0_seq_1")
        chain_sequences: List of amino acid sequences, one per chain
        chain_ids: List of chain IDs (e.g. ["A", "B"])
        num_seeds: Number of model seeds
        num_samples: Number of diffusion samples per seed

    Returns:
        dict: AF3 input JSON structure
    """
    sequences = []
    for chain_id, seq in zip(chain_ids, chain_sequences):
        sequences.append({
            "protein": {
                "id": chain_id,
                "sequence": seq
            }
        })

    return {
        "name": name,
        "modelSeeds": list(range(1, num_seeds + 1)),
        "dialect": "alphafold3",
        "version": 3,
        "sequences": sequences,
        "numDiffusionSamples": num_samples
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert ProteinMPNN FASTA output to AlphaFold 3 input JSON")
    parser.add_argument("--mpnn_fasta_dir", required=True,
                        help="Directory containing MPNN output .fa files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for AF3 JSON files")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of model seeds (default: 5)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of diffusion samples per seed (default: 5)")
    parser.add_argument("--designed_chain_id", default="A",
                        help="Chain ID for the designed chain (default: A)")
    parser.add_argument("--target_chain_id", default="B",
                        help="Chain ID for the target chain (default: B)")
    parser.add_argument("--process_count", type=int, default=0,
                        help="Max sequences to process per FASTA (0=all, default: 0)")
    parser.add_argument("--config", default=None,
                        help="Path to pipeline config.json (for template settings)")

    args = parser.parse_args()

    fasta_dir = Path(args.mpnn_fasta_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chain_ids = [args.designed_chain_id, args.target_chain_id]

    # Check for template settings in config
    use_templates = False
    template_config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            config = json.load(f)
        af3_config = config.get("alphafold3", {})
        use_templates = af3_config.get("use_templates", False)
        if use_templates:
            template_config = af3_config

    # Find all FASTA files
    fasta_files = sorted(fasta_dir.glob("*.fa"))
    if not fasta_files:
        fasta_files = sorted(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        print(f"No FASTA files found in {fasta_dir}", file=sys.stderr)
        sys.exit(1)

    total_generated = 0

    for fasta_file in fasta_files:
        rfd_name = fasta_file.stem  # e.g. "RFD_0"
        entries = parse_mpnn_fasta(fasta_file)

        if not entries:
            print(f"Warning: No sequences found in {fasta_file}")
            continue

        # Skip first entry (native/input sequence), process designed sequences
        designed_entries = entries[1:]

        if args.process_count > 0:
            designed_entries = designed_entries[:args.process_count]

        # Create subdirectory for this RFD design
        rfd_output_dir = output_dir / rfd_name
        rfd_output_dir.mkdir(parents=True, exist_ok=True)

        for seq_idx, (header, chain_seqs) in enumerate(designed_entries, start=1):
            if len(chain_seqs) < 2:
                print(f"Warning: Expected >=2 chains in {fasta_file} seq {seq_idx}, "
                      f"got {len(chain_seqs)}. Skipping.")
                continue

            # Use only the first N chains matching our chain_ids
            used_seqs = chain_seqs[:len(chain_ids)]

            job_name = f"{rfd_name}_seq_{seq_idx}"
            af3_json = build_af3_json(
                name=job_name,
                chain_sequences=used_seqs,
                chain_ids=chain_ids,
                num_seeds=args.num_seeds,
                num_samples=args.num_samples,
            )

            output_path = rfd_output_dir / f"{job_name}.json"
            with open(output_path, "w") as f:
                json.dump(af3_json, f, indent=2)

            total_generated += 1

    print(f"Generated {total_generated} AF3 input JSON files in {output_dir}")

    # If templates requested, apply them
    if use_templates and template_config.get("template_cif_dir"):
        print("Template support is enabled. Run add_templates_to_af3.py to add "
              "templates to the generated JSON files.")


if __name__ == "__main__":
    main()
