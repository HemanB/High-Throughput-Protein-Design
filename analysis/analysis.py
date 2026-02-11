#!/usr/bin/env python3
"""
Post-pipeline analysis for RFDiffusion + ProteinMPNN + AlphaFold 3 outputs.

Extracts confidence metrics (pLDDT, pTM, ipTM, PAE, ranking_score),
calculates clash scores and RMSD against reference structures, and
generates a PDF report.

Updated for AlphaFold 3 output format:
  - CIF models instead of PDB (uses MMCIFParser)
  - summary_confidences.json for ranking_score, ptm, iptm, has_clash
  - confidences.json for atom_plddts and pae matrix
  - seed-N_sample-M/ subdirectory structure

Usage:
  python analysis.py \
    --rf_base_dir /path/to/RFD/outputs \
    --af_base_dir /path/to/AF3/outputs \
    --output_dir /path/to/analysis_output
"""

import os
import json
import shutil
import math
import argparse

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.rms import rmsd
from Bio.PDB import MMCIFParser
from Bio import pairwise2
from Bio.SeqUtils import seq1
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table,
                                TableStyle, Spacer, Image as RLImage)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


###############################################################################
# 1) Chain-based Sequence Extraction (CIF)
###############################################################################
def get_chain_sequences(cif_path):
    """Extract chain sequences from an mmCIF file."""
    parser = MMCIFParser(QUIET=True)
    chain_dict = {}
    try:
        structure = parser.get_structure("temp", cif_path)
    except Exception as e:
        print(f"[get_chain_sequences] Error parsing {cif_path}: {e}")
        return {}

    for model in structure:
        for chain in model:
            cid = chain.id.strip()
            if cid in chain_dict:
                continue
            seq_list = []
            for residue in chain:
                if not hasattr(residue, "resname"):
                    continue
                res3 = residue.resname.upper().strip()
                try:
                    one_letter = seq1(res3)
                except Exception:
                    one_letter = "X"
                seq_list.append(one_letter)
            chain_dict[cid] = "".join(seq_list)
        break  # only first model
    return chain_dict


###############################################################################
# 2) Kabsch Algorithm
###############################################################################
def kabsch(P, Q):
    P_cent = P - P.mean(axis=0)
    Q_cent = Q - Q.mean(axis=0)
    H = P_cent.T @ Q_cent
    U, s, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = Q.mean(axis=0) - P.mean(axis=0) @ R
    return R, t


###############################################################################
# 3) Iterative RMSD with outlier rejection
###############################################################################
def iterative_rmsd_pymol(model_coords, ref_coords, max_iter=7, cutoff=2.0):
    if model_coords.shape[0] < 2:
        return np.nan, []

    indices = np.arange(len(model_coords))
    for _ in range(max_iter):
        if len(indices) < 2:
            break
        R, t = kabsch(model_coords[indices], ref_coords[indices])
        model_sub = model_coords[indices] - model_coords[indices].mean(axis=0)
        model_aligned = model_sub @ R + ref_coords[indices].mean(axis=0)
        dist = np.linalg.norm(model_aligned - ref_coords[indices], axis=1)
        current_rms = np.sqrt((dist**2).mean())
        new_indices = indices[dist <= (cutoff * current_rms)]
        if np.array_equal(new_indices, indices):
            break
        indices = new_indices

    if len(indices) < 2:
        return np.nan, []
    final_r = rmsd(model_coords[indices], ref_coords[indices],
                   center=True, superposition=True)
    return final_r, indices


def chain_rmsd_single_pair(ref_pdb, mod_cif, chain_ref_id, chain_mod_id,
                           aligned_ref_seq, aligned_mod_seq, cutoff=2.0):
    """Calculate RMSD between a reference PDB and a model CIF for one chain pair."""
    try:
        ref_u = mda.Universe(ref_pdb)
        mod_u = mda.Universe(mod_cif, format="MMCIF")
    except Exception as e:
        print(f"[chain_rmsd_single_pair] Error loading: {e}")
        return np.nan, []

    sel_ref = ref_u.select_atoms(f"segid {chain_ref_id} and protein")
    sel_mod = mod_u.select_atoms(f"segid {chain_mod_id} and protein")
    if (len(sel_ref) < 1) or (len(sel_mod) < 1):
        # Try chainID selection if segid fails
        sel_ref = ref_u.select_atoms(f"chainID {chain_ref_id} and protein")
        sel_mod = mod_u.select_atoms(f"chainID {chain_mod_id} and protein")
    if (len(sel_ref) < 1) or (len(sel_mod) < 1):
        return np.nan, []

    ref_res = sel_ref.residues
    mod_res = sel_mod.residues

    ref_coords_list = []
    mod_coords_list = []

    pos_r = 0
    pos_m = 0
    for i in range(len(aligned_ref_seq)):
        r_aa = aligned_ref_seq[i]
        m_aa = aligned_mod_seq[i]
        if (r_aa != "-") and (m_aa != "-"):
            if (pos_r < len(ref_res)) and (pos_m < len(mod_res)):
                rr = ref_res[pos_r]
                mr = mod_res[pos_m]
                for nm in ["N", "CA", "C", "O"]:
                    a1 = rr.atoms.select_atoms(f"name {nm}")
                    a2 = mr.atoms.select_atoms(f"name {nm}")
                    if (len(a1) == 1) and (len(a2) == 1):
                        ref_coords_list.append(a1.positions[0])
                        mod_coords_list.append(a2.positions[0])
        if r_aa != "-":
            pos_r += 1
        if m_aa != "-":
            pos_m += 1

    if len(ref_coords_list) < 2:
        return np.nan, []
    ref_coords = np.array(ref_coords_list)
    mod_coords = np.array(mod_coords_list)

    chain_r, used_ix = iterative_rmsd_pymol(mod_coords, ref_coords,
                                            max_iter=5, cutoff=cutoff)
    return chain_r, used_ix


###############################################################################
# 4) RMSD multimer (Hungarian chain assignment)
###############################################################################
def calculate_rmsd_multimer_robust(ref_pdb, model_cif, exclude_chains=None):
    if exclude_chains is None:
        exclude_chains = []

    ref_chains = get_chain_sequences_pdb(ref_pdb)
    mod_chains = get_chain_sequences(model_cif)
    if not ref_chains or not mod_chains:
        return "N/A"

    ref_chain_ids = sorted(ref_chains.keys())
    mod_chain_ids = sorted(mod_chains.keys())
    if not ref_chain_ids or not mod_chain_ids:
        return "N/A"

    n_ref = len(ref_chain_ids)
    n_mod = len(mod_chain_ids)
    cost_matrix = np.full((n_ref, n_mod), 1e6, dtype=float)

    for i, rc in enumerate(ref_chain_ids):
        seq_r = ref_chains[rc]
        for j, mc in enumerate(mod_chain_ids):
            seq_m = mod_chains[mc]
            if seq_r and seq_m:
                alns = pairwise2.align.globalxx(seq_r, seq_m)
                if alns:
                    score = alns[0][2]
                    cost_matrix[i, j] = -score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_atoms = 0
    sum_rms_atoms = 0.0
    chain_comps = []

    for idx in range(len(row_ind)):
        i = row_ind[idx]
        j = col_ind[idx]
        if cost_matrix[i, j] >= 1e6:
            continue
        rc = ref_chain_ids[i]
        mc = mod_chain_ids[j]

        seq_r = ref_chains[rc]
        seq_m = mod_chains[mc]
        alns2 = pairwise2.align.globalxx(seq_r, seq_m)
        if not alns2:
            continue
        best = alns2[0]
        aref = best[0]
        amod = best[1]

        c_r, used_ix = chain_rmsd_single_pair(ref_pdb, model_cif, rc, mc,
                                              aref, amod, cutoff=2.0)
        if c_r is np.nan or math.isnan(c_r):
            continue

        n_used = len(used_ix)
        c_str = f"{c_r:.2f}({n_used}atoms)"
        chain_comps.append((rc, mc, c_str))

        total_atoms += n_used
        sum_rms_atoms += (c_r * n_used)

    if total_atoms < 1:
        return "N/A"
    overall_rms = sum_rms_atoms / total_atoms
    return {
        "chain_comparisons": chain_comps,
        "average_rmsd": overall_rms
    }


def get_chain_sequences_pdb(pdb_path):
    """Extract chain sequences from a PDB file (for reference structures)."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    chain_dict = {}
    try:
        structure = parser.get_structure("temp", pdb_path)
    except Exception as e:
        print(f"[get_chain_sequences_pdb] Error parsing {pdb_path}: {e}")
        return {}

    for model in structure:
        for chain in model:
            cid = chain.id.strip()
            if cid in chain_dict:
                continue
            seq_list = []
            for residue in chain:
                if not hasattr(residue, "resname"):
                    continue
                res3 = residue.resname.upper().strip()
                try:
                    one_letter = seq1(res3)
                except Exception:
                    one_letter = "X"
                seq_list.append(one_letter)
            chain_dict[cid] = "".join(seq_list)
        break
    return chain_dict


###############################################################################
# 5) AF3 Metric Extraction
###############################################################################
def extract_af3_scores(summary_json_path):
    """Extract ranking_score, pTM, ipTM, and has_clash from AF3 summary_confidences.json."""
    if not os.path.isfile(summary_json_path):
        return {}
    try:
        with open(summary_json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {summary_json_path}: {e}")
        return {}

    return {
        "ranking_score": data.get("ranking_score", np.nan),
        "ptm": data.get("ptm", np.nan),
        "iptm": data.get("iptm", np.nan),
        "has_clash": data.get("has_clash", np.nan),
        "fraction_disordered": data.get("fraction_disordered", np.nan),
    }


def extract_plddt_from_confidences(confidences_json_path):
    """Extract per-atom pLDDT scores from AF3 confidences.json.

    Returns the mean pLDDT across all atoms.
    """
    if not os.path.isfile(confidences_json_path):
        return np.nan
    try:
        with open(confidences_json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {confidences_json_path}: {e}")
        return np.nan

    atom_plddts = data.get("atom_plddts", [])
    if not atom_plddts:
        return np.nan
    return float(np.mean(atom_plddts))


def extract_average_pae(confidences_json_path):
    """Extract average PAE from AF3 confidences.json."""
    if not os.path.isfile(confidences_json_path):
        return np.nan
    try:
        with open(confidences_json_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {confidences_json_path}: {e}")
        return np.nan

    pae = data.get("pae", [])
    if not pae:
        return np.nan
    matrix = np.array(pae)
    return float(np.mean(matrix))


###############################################################################
# 6) Clash Score Calculation (CIF)
###############################################################################
def calculate_clash_score(cif_path, distance_threshold=2.0):
    """Calculate steric clashes from an mmCIF structure."""
    parser = MMCIFParser(QUIET=True)
    if not os.path.isfile(cif_path):
        return np.nan

    try:
        structure = parser.get_structure("structure", cif_path)
    except Exception as e:
        print(f"Error parsing {cif_path}: {e}")
        return np.nan

    atom_coords = []
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element != "H":
                        atom_coords.append(atom.get_coord())
                        residues.append(residue.get_id())

    if len(atom_coords) < 2:
        return np.nan

    coords = np.array(atom_coords)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=distance_threshold)

    total_clashes = 0
    for i, j in pairs:
        if residues[i][1] == residues[j][1]:
            continue
        total_clashes += 1

    return total_clashes


###############################################################################
# 7) Walk AF3 Output Directory
###############################################################################
def find_af3_results(af_base_dir):
    """Walk AF3 output directories and collect result paths.

    AF3 output structure:
      af_base_dir/
        RFD_0/
          rfd_0_seq_1/
            seed-1_sample-0/
              *_summary_confidences.json
              *_confidences.json
              *_model.cif
            seed-1_sample-1/
              ...

    Returns list of dicts with paths and metadata.
    """
    results = []

    for root, dirs, files in os.walk(af_base_dir):
        # Look for seed-N_sample-M directories
        dirname = os.path.basename(root)
        if not (dirname.startswith("seed-") and "_sample-" in dirname):
            continue

        # Parse seed and sample
        parts = dirname.split("_")
        seed_str = parts[0].replace("seed-", "")
        sample_str = parts[1].replace("sample-", "")

        # Find files
        summary_json = None
        confidences_json = None
        model_cif = None
        for f in files:
            if f.endswith("_summary_confidences.json"):
                summary_json = os.path.join(root, f)
            elif f.endswith("_confidences.json") and "summary" not in f:
                confidences_json = os.path.join(root, f)
            elif f.endswith("_model.cif"):
                model_cif = os.path.join(root, f)

        if not (summary_json and model_cif):
            continue

        # Build name from path hierarchy
        # e.g. af_base_dir/RFD_0/rfd_0_seq_1/seed-1_sample-0/
        job_dir = os.path.dirname(root)  # rfd_0_seq_1
        rfd_dir = os.path.dirname(job_dir)  # RFD_0
        job_name = os.path.basename(job_dir)
        rfd_name = os.path.basename(rfd_dir)
        name_col = f"{rfd_name}/{job_name}"

        results.append({
            "name": name_col,
            "rfd_name": rfd_name,
            "job_name": job_name,
            "seed": seed_str,
            "sample": sample_str,
            "seed_sample": dirname,
            "summary_json": summary_json,
            "confidences_json": confidences_json,
            "model_cif": model_cif,
        })

    return results


###############################################################################
# 8) Main Pipeline
###############################################################################
def run_pipeline(args):
    rf_base_dir = args.rf_base_dir
    af_base_dir = args.af_base_dir
    output_dir = args.output_dir

    plddt_tsv = os.path.join(output_dir, args.plddt_tsv)
    rmsd_tsv = os.path.join(output_dir, args.rmsd_tsv)
    ptm_iptm_tsv = os.path.join(output_dir, args.ptm_iptm_tsv)
    pae_tsv = os.path.join(output_dir, args.pae_tsv)
    clash_tsv = os.path.join(output_dir, args.clash_tsv)
    final_merged_tsv = os.path.join(output_dir, args.final_merged_tsv)
    downselected_dir = os.path.join(output_dir, args.downselected_dir or "top20_cifs")
    os.makedirs(output_dir, exist_ok=True)

    # Find all AF3 results
    print("Scanning AF3 output directory...")
    af3_results = find_af3_results(af_base_dir)
    print(f"Found {len(af3_results)} AF3 predictions.")

    if not af3_results:
        print("No AF3 results found. Check af_base_dir path.")
        return None

    # ── (A) Collect all metrics ──────────────────────────────────────────────
    print("\nCollecting AF3 confidence metrics...")
    all_entries = []

    for r in af3_results:
        # Summary confidences
        scores = extract_af3_scores(r["summary_json"])
        ranking_score = scores.get("ranking_score", np.nan)
        ptm_val = scores.get("ptm", np.nan)
        iptm_val = scores.get("iptm", np.nan)
        has_clash_flag = scores.get("has_clash", np.nan)

        # Per-atom pLDDT
        mean_plddt = np.nan
        if r["confidences_json"]:
            mean_plddt = extract_plddt_from_confidences(r["confidences_json"])

        # Average PAE
        avg_pae = np.nan
        if r["confidences_json"]:
            avg_pae = extract_average_pae(r["confidences_json"])

        # Clash score (from structure)
        clash_count = calculate_clash_score(r["model_cif"])

        all_entries.append({
            "Name": r["name"],
            "Job_Name": r["job_name"],
            "Seed": r["seed"],
            "Sample": r["sample"],
            "Seed_Sample": r["seed_sample"],
            "Model_CIF": r["model_cif"],
            "Ranking_Score": ranking_score,
            "pLDDT": mean_plddt,
            "pTM": ptm_val,
            "ipTM": iptm_val,
            "Average_PAE": avg_pae,
            "Has_Clash": has_clash_flag,
            "Clash_Count": clash_count,
        })

    metrics_df = pd.DataFrame(all_entries)
    metrics_df.sort_values(by="Ranking_Score", ascending=False, inplace=True)

    # Write individual metric TSVs
    plddt_df = metrics_df[["Name", "Job_Name", "Seed", "Sample", "Seed_Sample",
                           "pLDDT", "Ranking_Score"]].copy()
    plddt_df.to_csv(plddt_tsv, sep="\t", index=False)
    print(f"pLDDT => {plddt_tsv}")

    ptm_iptm_df = metrics_df[["Name", "Job_Name", "Seed", "Sample",
                              "pTM", "ipTM"]].copy()
    ptm_iptm_df.to_csv(ptm_iptm_tsv, sep="\t", index=False)
    print(f"pTM/ipTM => {ptm_iptm_tsv}")

    pae_df = metrics_df[["Name", "Job_Name", "Seed", "Sample",
                         "Average_PAE"]].copy()
    pae_df.to_csv(pae_tsv, sep="\t", index=False)
    print(f"PAE => {pae_tsv}")

    clash_df = metrics_df[["Name", "Job_Name", "Seed", "Sample",
                           "Has_Clash", "Clash_Count"]].copy()
    clash_df.to_csv(clash_tsv, sep="\t", index=False)
    print(f"Clashes => {clash_tsv}")

    # ── (B) RMSD ─────────────────────────────────────────────────────────────
    print("\nCalculating RMSD for each reference vs. AF3 predictions...")
    rmsd_results = []

    ref_files = [f for f in os.listdir(rf_base_dir) if f.endswith(".pdb")]
    for ref_file in sorted(ref_files):
        ref_filepath = os.path.join(rf_base_dir, ref_file)
        ref_name_noext = os.path.splitext(ref_file)[0]
        print(f"\nReference PDB: {ref_filepath}")

        for r in af3_results:
            # Match RFD name
            if r["rfd_name"].upper() != ref_name_noext.upper():
                continue

            model_cif = r["model_cif"]
            name_col = r["name"]

            rmsd_val = calculate_rmsd_multimer_robust(ref_filepath, model_cif)

            if isinstance(rmsd_val, dict):
                if "chain_comparisons" in rmsd_val:
                    avg_val = rmsd_val["average_rmsd"]
                    details = [f"{rc}->{mc}:{val_str}"
                               for (rc, mc, val_str) in rmsd_val["chain_comparisons"]]
                    chain_details_str = "; ".join(details)
                    final_rmsd = avg_val
                else:
                    final_rmsd = "N/A"
                    chain_details_str = "N/A"
            else:
                final_rmsd = "N/A"
                chain_details_str = "N/A"

            rmsd_results.append({
                "Name": name_col,
                "Reference": ref_name_noext,
                "Seed": r["seed"],
                "Sample": r["sample"],
                "RMSD": final_rmsd,
                "Chain_RMSD_Details": chain_details_str,
            })

    rmsd_df = pd.DataFrame(rmsd_results)
    rmsd_df.to_csv(rmsd_tsv, sep="\t", index=False)
    print(f"RMSD => {rmsd_tsv}")

    # ── (C) Merge all metrics ────────────────────────────────────────────────
    print("\nMerging all metrics...")
    merge_keys = ["Name", "Seed", "Sample"]

    if not rmsd_df.empty:
        merged_all = metrics_df.merge(rmsd_df[["Name", "Seed", "Sample", "RMSD",
                                                "Chain_RMSD_Details"]],
                                      on=merge_keys, how="left")
    else:
        merged_all = metrics_df.copy()
        merged_all["RMSD"] = "N/A"
        merged_all["Chain_RMSD_Details"] = "N/A"

    merged_all.to_csv(final_merged_tsv, sep="\t", index=False)
    print(f"Final merged => {final_merged_tsv}")

    # ── (D) Copy Top 20 by Ranking Score ─────────────────────────────────────
    print("\nCopying top 20 by Ranking Score...")
    top_20_df = merged_all.sort_values(by="Ranking_Score", ascending=False).head(20)
    os.makedirs(downselected_dir, exist_ok=True)
    for _, row in top_20_df.iterrows():
        src = row["Model_CIF"]
        name_val = row["Name"].replace("/", "_")
        seed_sample = row["Seed_Sample"]
        tgt_name = f"{name_val}_{seed_sample}_model.cif"
        tgt_path = os.path.join(downselected_dir, tgt_name)
        if os.path.isfile(src):
            shutil.copy(src, tgt_path)
        else:
            print(f"Warning: Missing {src}")

    print(f"Top 20 copied to => {downselected_dir}")
    print("\nPipeline completed!\n")

    return {
        "af_base_dir": af_base_dir,
        "output_dir": output_dir,
        "plddt_tsv": plddt_tsv,
        "rmsd_tsv": rmsd_tsv,
        "ptm_iptm_tsv": ptm_iptm_tsv,
        "pae_tsv": pae_tsv,
        "clash_tsv": clash_tsv,
        "final_merged_tsv": final_merged_tsv,
    }


###############################################################################
# 9) PDF Reporting
###############################################################################
def run_report(config, pdf_name="analysis_report.pdf"):
    """Generate a PDF report with tables and scatter plots."""
    import sys
    import io
    import plotly.express as px

    final_merged_file = config["final_merged_tsv"]
    output_dir = config["output_dir"]
    af_dir = config["af_base_dir"]

    pdf_filepath = os.path.join(output_dir, pdf_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Generating PDF report ---")
    print(f"Output PDF => {pdf_filepath}")

    try:
        merged = pd.read_csv(final_merged_file, sep="\t")
    except Exception as e:
        print(f"Error reading merged TSV: {e}")
        sys.exit(1)

    if merged.empty:
        print("No data to report.")
        return

    merged["pLDDT"] = pd.to_numeric(merged["pLDDT"], errors="coerce")
    merged["RMSD"] = pd.to_numeric(merged["RMSD"], errors="coerce")
    merged["Ranking_Score"] = pd.to_numeric(merged["Ranking_Score"], errors="coerce")

    # Sort by ranking score
    merged_sorted = merged.sort_values("Ranking_Score", ascending=False)
    top_20 = merged_sorted.head(20).reset_index(drop=True)

    # Color by seed
    seed_colors = {
        "1": "blue", "2": "green", "3": "orange",
        "4": "red", "5": "purple"
    }

    # Scatter: All (pLDDT vs RMSD)
    plot_df = merged.dropna(subset=["pLDDT", "RMSD"])
    if not plot_df.empty:
        fig_all = px.scatter(
            plot_df,
            x="pLDDT", y="RMSD",
            color=plot_df["Seed"].astype(str),
            color_discrete_map=seed_colors,
            hover_data={"Name": True, "Ranking_Score": ":.3f",
                        "pLDDT": ":.2f", "RMSD": ":.2f"},
            title="pLDDT vs RMSD (All Predictions)",
            labels={"pLDDT": "Mean pLDDT", "RMSD": "RMSD (A)", "color": "Seed"}
        )
        fig_all.update_layout(template="plotly_white")

        # Top 20
        top_20_plot = top_20.dropna(subset=["pLDDT", "RMSD"])
        fig_top20 = px.scatter(
            top_20_plot,
            x="pLDDT", y="RMSD",
            color=top_20_plot["Seed"].astype(str),
            color_discrete_map=seed_colors,
            hover_data={"Name": True, "Ranking_Score": ":.3f",
                        "pLDDT": ":.2f", "RMSD": ":.2f"},
            title="pLDDT vs RMSD (Top 20 by Ranking Score)",
            labels={"pLDDT": "Mean pLDDT", "RMSD": "RMSD (A)", "color": "Seed"}
        )
        fig_top20.update_layout(template="plotly_white")

        try:
            fig_all_png = fig_all.to_image(format="png", scale=2)
            fig_top20_png = fig_top20.to_image(format="png", scale=2)
        except Exception as e:
            print(f"Plotly figure conversion error: {e}")
            fig_all_png = None
            fig_top20_png = None
    else:
        fig_all_png = None
        fig_top20_png = None

    # Build PDF
    doc = SimpleDocTemplate(pdf_filepath, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    title_par = Paragraph(f"AF3 Analysis Report", styles["Title"])
    elements.append(title_par)
    elements.append(Spacer(1, 6))
    subtitle = Paragraph(f"AF3 dir: {af_dir}", styles["Normal"])
    elements.append(subtitle)
    elements.append(Spacer(1, 12))

    # Top 20 table
    heading_top20 = Paragraph("Top 20 by Ranking Score", styles["Heading2"])
    elements.append(heading_top20)
    elements.append(Spacer(1, 12))

    def fmt(x):
        if isinstance(x, float) and not math.isnan(x):
            return f"{x:.3f}"
        return str(x)

    table_data = [["#", "Name", "Seed/Sample", "Ranking", "pLDDT", "ipTM", "RMSD"]]
    for i, row in top_20.iterrows():
        table_data.append([
            i + 1,
            str(row.get("Name", "")),
            str(row.get("Seed_Sample", "")),
            fmt(row.get("Ranking_Score", np.nan)),
            fmt(row.get("pLDDT", np.nan)),
            fmt(row.get("ipTM", np.nan)),
            fmt(row.get("RMSD", np.nan)),
        ])

    tbl = Table(table_data, colWidths=[25, 120, 80, 55, 50, 50, 50])
    tbl_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#D9E1F2")),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ])
    tbl.setStyle(tbl_style)
    elements.append(tbl)
    elements.append(Spacer(1, 24))

    # Scatter plots
    if fig_all_png:
        elements.append(Paragraph("pLDDT vs RMSD (All)", styles["Heading2"]))
        elements.append(Spacer(1, 12))
        img_buf = io.BytesIO(fig_all_png)
        elements.append(RLImage(img_buf, width=6.5 * inch, height=4.5 * inch))
        elements.append(Spacer(1, 24))

    if fig_top20_png:
        elements.append(Paragraph("pLDDT vs RMSD (Top 20)", styles["Heading2"]))
        elements.append(Spacer(1, 12))
        img_buf = io.BytesIO(fig_top20_png)
        elements.append(RLImage(img_buf, width=6.5 * inch, height=4.5 * inch))
        elements.append(Spacer(1, 24))

    # Additional metrics table
    elements.append(Paragraph("Additional Metrics (Top 20)", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    metric_data = [["Name", "Seed/Sample", "pTM", "PAE", "Has_Clash", "Clashes"]]
    for _, row in top_20.iterrows():
        metric_data.append([
            str(row.get("Name", "")),
            str(row.get("Seed_Sample", "")),
            fmt(row.get("pTM", np.nan)),
            fmt(row.get("Average_PAE", np.nan)),
            fmt(row.get("Has_Clash", np.nan)),
            fmt(row.get("Clash_Count", np.nan)),
        ])

    tbl_metrics = Table(metric_data, colWidths=[120, 80, 50, 50, 60, 50])
    style_metrics = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7030A0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E2D4F0")),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ])
    tbl_metrics.setStyle(style_metrics)
    elements.append(tbl_metrics)

    try:
        doc.build(elements)
        print(f"\nPDF report generated at {pdf_filepath}\n")
    except Exception as e:
        print(f"Error building PDF: {e}")


###############################################################################
# 10) main()
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Post-pipeline analysis for RFDiffusion + ProteinMPNN + AlphaFold 3")
    parser.add_argument("--rf_base_dir", required=True,
                        help="Path to RFDiffusion output PDBs")
    parser.add_argument("--af_base_dir", required=True,
                        help="Path to AlphaFold 3 output directories")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for analysis TSVs/PDF")
    parser.add_argument("--plddt_tsv", default="plddt.tsv")
    parser.add_argument("--rmsd_tsv", default="rmsd.tsv")
    parser.add_argument("--ptm_iptm_tsv", default="ptm_iptm.tsv")
    parser.add_argument("--pae_tsv", default="pae.tsv")
    parser.add_argument("--clash_tsv", default="clash.tsv")
    parser.add_argument("--final_merged_tsv", default="final_merged.tsv")
    parser.add_argument("--downselected_dir", default="top20_cifs",
                        help="Subfolder for top 20 CIF files")
    parser.add_argument("--pdf_name", default="analysis_report.pdf",
                        help="PDF filename")

    args = parser.parse_args()
    config = run_pipeline(args)
    if config:
        run_report(config, pdf_name=args.pdf_name)


if __name__ == "__main__":
    main()
