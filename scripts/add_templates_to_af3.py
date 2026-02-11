#!/usr/bin/env python3
"""
Add structural templates to AlphaFold 3 input JSON files.

This script reads AF3 input JSON files and adds mmCIF templates for
specified chains. Templates are aligned to query sequences using
semi-global alignment, and the resulting queryIndices/templateIndices
mappings are embedded in the JSON.

When templates are added, unpairedMsa and pairedMsa are set to ""
(empty string) to skip MSA search, as required by AF3.

Usage:
  python add_templates_to_af3.py \
    --input_json /path/to/af3_input.json \
    --output_json /path/to/af3_input_with_template.json \
    --chain_template_map "A:/path/to/chain_a.cif,B:/path/to/chain_b.cif" \
    --release_date 2024-01-01

  # Process a directory of JSON files:
  python add_templates_to_af3.py \
    --input_dir /path/to/af3_inputs/ \
    --output_dir /path/to/af3_inputs_with_templates/ \
    --chain_template_map "A:/path/to/chain_a.cif,B:/path/to/chain_b.cif" \
    --release_date 2024-01-01
"""

import argparse
import json
import numpy as np
from pathlib import Path

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def build_proper_mmcif(cif_path, template_name, chain_id, date="2024-01-01"):
    """Read a CIF file and rebuild it with all metadata AF3 requires.

    AF3 requires:
      - _entity with type 'polymer'
      - _entity_poly with type 'polypeptide(L)'
      - _struct_asym mapping chain to entity
      - _pdbx_audit_revision_history with revision_date
      - _chem_comp for each residue type
      - ATOM records with valid label_asym_id and label_entity_id
    """
    with open(cif_path) as f:
        raw_lines = f.readlines()

    atom_lines = []
    comp_ids = set()
    atom_type_symbols = set()
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("ATOM") or stripped.startswith("HETATM"):
            atom_lines.append(stripped)
            parts = stripped.split()
            comp_ids.add(parts[5])
            atom_type_symbols.add(parts[2])

    cell_params = {}
    symmetry_params = {}
    for line in raw_lines:
        stripped = line.strip()
        if stripped.startswith("_cell.") and not stripped.startswith("_cell.entry_id"):
            key = stripped.split()[0]
            val = stripped.split(None, 1)[1] if len(stripped.split(None, 1)) > 1 else ""
            cell_params[key] = val
        if stripped.startswith("_symmetry.") and not stripped.startswith("_symmetry.entry_id"):
            key = stripped.split()[0]
            val = stripped.split(None, 1)[1] if len(stripped.split(None, 1)) > 1 else ""
            symmetry_params[key] = val

    has_auth_seq_id = any("_atom_site.auth_seq_id" in line for line in raw_lines)

    fixed_atom_lines = []
    for line in atom_lines:
        parts = line.split()
        parts[6] = chain_id
        parts[7] = "1"

        if parts[8] == "." and has_auth_seq_id:
            parts[8] = parts[16]

        if not has_auth_seq_id:
            auth_seq_id = parts[8]
            parts.insert(16, auth_seq_id)

        fixed_atom_lines.append(" ".join(parts))

    out = []
    out.append(f"data_{template_name}")
    out.append(f"_entry.id '{template_name}'")
    out.append("#")

    out.append(f"_cell.entry_id '{template_name}'")
    for key in ["_cell.length_a", "_cell.length_b", "_cell.length_c",
                "_cell.angle_alpha", "_cell.angle_beta", "_cell.angle_gamma"]:
        if key in cell_params:
            out.append(f"{key} {cell_params[key]}")
    out.append("#")

    out.append(f"_symmetry.entry_id '{template_name}'")
    for key in ["_symmetry.space_group_name_H-M", "_symmetry.Int_Tables_number"]:
        if key in symmetry_params:
            out.append(f"{key} {symmetry_params[key]}")
    out.append("#")

    out.append("loop_")
    out.append("_pdbx_audit_revision_history.ordinal")
    out.append("_pdbx_audit_revision_history.data_content_type")
    out.append("_pdbx_audit_revision_history.major_revision")
    out.append("_pdbx_audit_revision_history.minor_revision")
    out.append("_pdbx_audit_revision_history.revision_date")
    out.append(f"1 'Structure model' 1 0 {date}")
    out.append("#")

    out.append("_entity.id 1")
    out.append("_entity.type polymer")
    out.append("#")

    out.append("_entity_poly.entity_id 1")
    out.append("_entity_poly.type 'polypeptide(L)'")
    out.append(f"_entity_poly.pdbx_strand_id {chain_id}")
    out.append("#")

    out.append("loop_")
    out.append("_struct_asym.id")
    out.append("_struct_asym.entity_id")
    out.append(f"{chain_id} 1")
    out.append("#")

    out.append("loop_")
    out.append("_chem_comp.id")
    out.append("_chem_comp.type")
    for comp in sorted(comp_ids):
        out.append(f"{comp} 'L-peptide linking'")
    out.append("#")

    out.append("loop_")
    out.append("_atom_type.symbol")
    for sym in sorted(atom_type_symbols):
        out.append(sym)
    out.append("#")

    out.append("loop_")
    out.append("_atom_site.group_PDB")
    out.append("_atom_site.id")
    out.append("_atom_site.type_symbol")
    out.append("_atom_site.label_atom_id")
    out.append("_atom_site.label_alt_id")
    out.append("_atom_site.label_comp_id")
    out.append("_atom_site.label_asym_id")
    out.append("_atom_site.label_entity_id")
    out.append("_atom_site.label_seq_id")
    out.append("_atom_site.pdbx_PDB_ins_code")
    out.append("_atom_site.Cartn_x")
    out.append("_atom_site.Cartn_y")
    out.append("_atom_site.Cartn_z")
    out.append("_atom_site.occupancy")
    out.append("_atom_site.B_iso_or_equiv")
    out.append("_atom_site.pdbx_formal_charge")
    out.append("_atom_site.auth_seq_id")
    out.append("_atom_site.auth_asym_id")
    out.append("_atom_site.pdbx_PDB_model_num")
    for atom_line in fixed_atom_lines:
        out.append(atom_line)
    out.append("#")

    return "\n".join(out)


def extract_sequence_from_cif(cif_path, seq_id_col=8, comp_id_col=5):
    """Extract amino acid sequence from a CIF file's ATOM records."""
    residues = {}
    with open(cif_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                parts = line.split()
                raw_seq_id = parts[seq_id_col]
                if raw_seq_id == ".":
                    raw_seq_id = parts[16]
                seq_id = int(raw_seq_id)
                comp_id = parts[comp_id_col]
                if seq_id not in residues:
                    residues[seq_id] = comp_id
    sorted_ids = sorted(residues.keys())
    seq = "".join(AA3TO1.get(residues[i], "X") for i in sorted_ids)
    return seq, sorted_ids


def semi_global_align(query, template, match=2, mismatch=-1, gap=-2):
    """Semi-global alignment: query aligns end-to-end, template has free
    leading/trailing gaps. Returns (query_indices, template_indices) lists
    of matched positions (both 0-indexed)."""
    n, m = len(query), len(template)
    score = np.zeros((n + 1, m + 1), dtype=int)
    traceback = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(1, n + 1):
        score[i][0] = gap * i
        traceback[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = 0
        traceback[0][j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if query[i - 1] == template[j - 1] else mismatch
            diag = score[i - 1][j - 1] + s
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap

            if diag >= up and diag >= left:
                score[i][j] = diag
                traceback[i][j] = 0
            elif up >= left:
                score[i][j] = up
                traceback[i][j] = 1
            else:
                score[i][j] = left
                traceback[i][j] = 2

    best_j = int(np.argmax(score[n]))

    pairs = []
    i, j = n, best_j
    while i > 0:
        if j > 0 and traceback[i][j] == 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif traceback[i][j] == 1 or j == 0:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1

    pairs.reverse()

    query_indices = []
    template_indices = []
    for qi, ti in pairs:
        if qi is not None and ti is not None:
            query_indices.append(qi)
            template_indices.append(ti)

    return query_indices, template_indices


def add_templates_to_json(data, chain_template_map, release_date):
    """Add templates to an AF3 JSON data dict.

    Args:
        data: AF3 input JSON dict
        chain_template_map: dict of chain_id -> cif_path
        release_date: Template release date string

    Returns:
        Modified data dict with templates added
    """
    for seq_entry in data.get("sequences", []):
        if "protein" not in seq_entry:
            continue

        protein = seq_entry["protein"]
        chain_id = protein["id"]

        if chain_id not in chain_template_map:
            continue

        cif_path = Path(chain_template_map[chain_id])
        if not cif_path.exists():
            print(f"Warning: Template CIF not found for chain {chain_id}: {cif_path}")
            continue

        query_seq = protein["sequence"]
        tmpl_seq, _ = extract_sequence_from_cif(cif_path)

        qi, ti = semi_global_align(query_seq, tmpl_seq)

        identity = sum(1 for q, t in zip(qi, ti)
                       if query_seq[q] == tmpl_seq[t])
        print(f"  Chain {chain_id}: {len(query_seq)} aa query, "
              f"{len(tmpl_seq)} aa template, "
              f"{identity}/{len(qi)} identity ({100*identity/max(len(qi),1):.1f}%)")

        template_name = f"{cif_path.stem}_chain{chain_id}"
        mmcif_str = build_proper_mmcif(cif_path, template_name, "A", release_date)

        protein["templates"] = [{
            "mmcif": mmcif_str,
            "queryIndices": qi,
            "templateIndices": ti,
        }]
        protein["unpairedMsa"] = ""
        protein["pairedMsa"] = ""

    return data


def process_single_file(input_path, output_path, chain_template_map, release_date):
    """Process a single AF3 JSON file."""
    with open(input_path) as f:
        data = json.load(f)

    data = add_templates_to_json(data, chain_template_map, release_date)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def parse_chain_template_map(map_str):
    """Parse chain template map string like 'A:/path/a.cif,B:/path/b.cif'."""
    result = {}
    for pair in map_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        chain_id, cif_path = pair.split(":", 1)
        result[chain_id.strip()] = cif_path.strip()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Add structural templates to AF3 input JSON files")
    parser.add_argument("--input_json", default=None,
                        help="Single input AF3 JSON file")
    parser.add_argument("--output_json", default=None,
                        help="Output path for single file mode")
    parser.add_argument("--input_dir", default=None,
                        help="Directory of AF3 JSON files to process")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for batch mode")
    parser.add_argument("--chain_template_map", required=True,
                        help="Chain-to-CIF mapping, e.g. 'A:/path/a.cif,B:/path/b.cif'")
    parser.add_argument("--release_date", default="2024-01-01",
                        help="Template release date (default: 2024-01-01)")

    args = parser.parse_args()
    chain_map = parse_chain_template_map(args.chain_template_map)

    if not chain_map:
        print("Error: No valid chain template mappings provided.", file=sys.stderr)
        sys.exit(1)

    print(f"Chain template mappings: {chain_map}")
    print(f"Release date: {args.release_date}")

    if args.input_json:
        output = args.output_json or args.input_json.replace(".json", "_with_template.json")
        print(f"\nProcessing: {args.input_json}")
        process_single_file(args.input_json, output, chain_map, args.release_date)
        print(f"Output: {output}")

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else input_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(input_dir.rglob("*.json"))
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            sys.exit(1)

        print(f"\nProcessing {len(json_files)} files from {input_dir}")
        for json_file in json_files:
            rel_path = json_file.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"\n  {rel_path}")
            process_single_file(json_file, out_path, chain_map, args.release_date)

        print(f"\nDone. Output in {output_dir}")

    else:
        print("Error: Provide either --input_json or --input_dir", file=sys.stderr)
        sys.exit(1)


import sys

if __name__ == "__main__":
    main()
