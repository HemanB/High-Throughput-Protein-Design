#!/usr/bin/env python3

import os
import json
import shutil
import numpy as np
import pandas as pd
import argparse
import math

# -----------------------------------------------------------------------
# MD / RMSD Imports
# -----------------------------------------------------------------------
import MDAnalysis as mda
from MDAnalysis.analysis.rms import rmsd
from Bio.PDB import PDBParser
from Bio import pairwise2 
from Bio.SeqUtils import seq1
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# -----------------------------------------------------------------------
# For PDF generation
# -----------------------------------------------------------------------
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table,
                                TableStyle, Spacer, Image as RLImage)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

###############################################################################
# 1) Chain-based Sequence Extraction
###############################################################################
def get_chain_sequences(pdb_path):
    parser = PDBParser(QUIET=True)
    chain_dict = {}
    try:
        structure = parser.get_structure("temp", pdb_path)
    except Exception as e:
        print(f"[get_chain_sequences] Error parsing {pdb_path}: {e}")
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
                    one_letter = seq1(res3)  # standard AA -> 1-letter
                except:
                    one_letter = "X"
                seq_list.append(one_letter)
            chain_dict[cid] = "".join(seq_list)
        break  # only first model
    return chain_dict

###############################################################################
# 2) Kabsch
###############################################################################
def kabsch(P, Q):
    P_cent = P - P.mean(axis=0)
    Q_cent = Q - Q.mean(axis=0)
    H = P_cent.T @ Q_cent
    U, s, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1,1,d])
    R = Vt.T @ D @ U.T
    t = Q.mean(axis=0) - P.mean(axis=0) @ R
    return R, t

###############################################################################
# 3) PyMOL-like iterative rejection
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
        # remove outliers > cutoff*current_rms
        new_indices = indices[dist <= (cutoff * current_rms)]
        if np.array_equal(new_indices, indices):
            break
        indices = new_indices

    if len(indices)<2:
        return np.nan, []
    final_r = rmsd(model_coords[indices], ref_coords[indices], center=True, superposition=True)
    return final_r, indices

def chain_rmsd_single_pair(ref_pdb, mod_pdb, chain_ref_id, chain_mod_id,
                           aligned_ref_seq, aligned_mod_seq, cutoff=2.0):
    """
    1) Load both PDBs with MDAnalysis
    2) Select chain_ref_id in reference, chain_mod_id in model
    3) Based on alignment strings (including gaps), gather matched backbone coords
    4) Do iterative RMSD, return (chain_rms, used_indices)
    """
    try:
        ref_u = mda.Universe(ref_pdb)
        mod_u = mda.Universe(mod_pdb)
    except Exception as e:
        print(f"[chain_rmsd_single_pair] Error loading: {e}")
        return np.nan, []

    sel_ref = ref_u.select_atoms(f"segid {chain_ref_id} and protein")
    sel_mod = mod_u.select_atoms(f"segid {chain_mod_id} and protein")
    if (len(sel_ref)<1) or (len(sel_mod)<1):
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
                # gather backbone
                for nm in ["N","CA","C","O"]:
                    a1 = rr.atoms.select_atoms(f"name {nm}")
                    a2 = mr.atoms.select_atoms(f"name {nm}")
                    if (len(a1)==1) and (len(a2)==1):
                        ref_coords_list.append(a1.positions[0])
                        mod_coords_list.append(a2.positions[0])
        if r_aa != "-":
            pos_r+=1
        if m_aa != "-":
            pos_m+=1

    if len(ref_coords_list)<2:
        return np.nan, []
    ref_coords = np.array(ref_coords_list)
    mod_coords = np.array(mod_coords_list)

    chain_r, used_ix = iterative_rmsd_pymol(mod_coords, ref_coords, max_iter=5, cutoff=cutoff)
    return chain_r, used_ix

###############################################################################
# 5)  RMSD multimer run
###############################################################################
def calculate_rmsd_multimer_robust(ref_pdb, model_pdb, exclude_chains=None):
    if exclude_chains is None:
        exclude_chains=[]

    # 1) chain sequences
    ref_chains = get_chain_sequences(ref_pdb)
    mod_chains = get_chain_sequences(model_pdb)
    if not ref_chains or not mod_chains:
        return "N/A"

    ref_chain_ids = sorted(ref_chains.keys())
    mod_chain_ids = sorted(mod_chains.keys())
    if (len(ref_chain_ids)==0) or (len(mod_chain_ids)==0):
        return "N/A"

    # 2) cost matrix from -alignment_score
    n_ref = len(ref_chain_ids)
    n_mod = len(mod_chain_ids)
    cost_matrix = np.full((n_ref,n_mod),1e6,dtype=float)

    for i,rc in enumerate(ref_chain_ids):
        seq_r = ref_chains[rc]
        for j,mc in enumerate(mod_chain_ids):
            seq_m = mod_chains[mc]
            if seq_r and seq_m:
                alns = pairwise2.align.globalxx(seq_r, seq_m)
                if alns:
                    score = alns[0][2]
                    cost_matrix[i,j] = -score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    chain_comps = []
    total_atoms = 0
    sum_rms_atoms = 0.0

    for idx in range(len(row_ind)):
        i = row_ind[idx]
        j = col_ind[idx]
        if cost_matrix[i,j]>=1e6:
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

        # 4) chain-level RMSD
        c_r, used_ix = chain_rmsd_single_pair(ref_pdb, model_pdb, rc, mc, aref, amod, cutoff=2.0)
        if (c_r is np.nan) or math.isnan(c_r):
            continue

        n_used = len(used_ix)
        # store e.g. "2.30(1256atoms)"
        c_str = f"{c_r:.2f}({n_used}atoms)"
        chain_comps.append((rc, mc, c_str))

        total_atoms += n_used
        sum_rms_atoms += (c_r * n_used)

    if total_atoms<1:
        return "N/A"
    overall_rms = sum_rms_atoms / total_atoms
    return {
        "chain_comparisons": chain_comps,
        "average_rmsd": overall_rms
    }

###############################################################################
# 2) pLDDT Extraction
###############################################################################
def extract_plddt_scores(ranking_debug_json_path):
    results = []
    if not os.path.isfile(ranking_debug_json_path):
        return results

    object_name = os.path.basename(os.path.dirname(ranking_debug_json_path))
    folder_two_up = os.path.basename(os.path.dirname(os.path.dirname(ranking_debug_json_path)))
    name_col = folder_two_up + "/" + object_name

    try:
        with open(ranking_debug_json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {ranking_debug_json_path}: {e}")
        return results

    if "iptm+ptm" not in data:
        return results

    iptm_ptm = data["iptm+ptm"]
    items = list(iptm_ptm.items())
    for rank_idx, (model_key, plddt_val) in enumerate(items):
        ranked_file = f"ranked_{rank_idx}.pdb"
        results.append((name_col, object_name, model_key, ranked_file, plddt_val))
    return results

###############################################################################
# 3) pTM/ipTM Extraction
###############################################################################
def extract_ptm_iptm(ranking_debug_filepath, model_identifier):
    pTM = np.nan
    ipTM = np.nan
    if not os.path.isfile(ranking_debug_filepath):
        return pTM, ipTM

    try:
        with open(ranking_debug_filepath, 'r') as f_rank:
            rank_data = json.load(f_rank)
    except Exception as e:
        print(f"Error reading {ranking_debug_filepath}: {e}")
        return pTM, ipTM

    if "iptm+ptm" in rank_data:
        iptm_ptm_dict = rank_data["iptm+ptm"]
        if model_identifier in iptm_ptm_dict:
            pTM = iptm_ptm_dict[model_identifier] * 100
            ipTM = pTM
    return pTM, ipTM

###############################################################################
# 4) PAE Extraction
###############################################################################
def extract_average_pae(pae_json_filepath):
    average_pae = np.nan
    if not os.path.isfile(pae_json_filepath):
        return average_pae

    try:
        with open(pae_json_filepath, 'r') as f_pae:
            pae_data = json.load(f_pae)
    except Exception as e:
        print(f"Error reading PAE JSON file {pae_json_filepath}: {e}")
        return average_pae

    if isinstance(pae_data, dict):
        if "predicted_aligned_error" in pae_data:
            matrix = pae_data["predicted_aligned_error"]
            if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
                matrix = np.array(matrix)
                average_pae = np.mean(matrix)
    elif isinstance(pae_data, list) and len(pae_data) > 0:
        first_element = pae_data[0]
        if isinstance(first_element, dict) and "predicted_aligned_error" in first_element:
            matrix = first_element["predicted_aligned_error"]
            if isinstance(matrix, list) and all(isinstance(row, list) for row in matrix):
                matrix = np.array(matrix)
                average_pae = np.mean(matrix)

    return average_pae

###############################################################################
# 5) Clash Score Calculation
###############################################################################
def calculate_clash_score(pdb_filepath, distance_threshold=2.0):
    parser = PDBParser(QUIET=True)
    if not os.path.isfile(pdb_filepath):
        return np.nan

    try:
        structure = parser.get_structure('structure', pdb_filepath)
    except Exception as e:
        print(f"Error parsing {pdb_filepath}: {e}")
        return np.nan

    atom_coords = []
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element != 'H':  # skip hydrogens
                        atom_coords.append(atom.get_coord())
                        residues.append(residue.get_id())

    if len(atom_coords) < 2:
        return np.nan

    coords = np.array(atom_coords)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=distance_threshold)

    total_clashes = 0
    for i, j in pairs:
        # If they belong to the same residue index, skip
        if residues[i][1] == residues[j][1]:
            continue
        total_clashes += 1

    return total_clashes

###############################################################################
# 6) Main Pipeline
###############################################################################
def run_pipeline(args):
    rf_base_dir  = args.rf_base_dir
    af_base_dir  = args.af_base_dir
    output_dir   = args.output_dir

    # TSV outputs
    plddt_tsv       = os.path.join(output_dir, args.plddt_tsv)
    rmsd_tsv        = os.path.join(output_dir, args.rmsd_tsv)
    ptm_iptm_tsv    = os.path.join(output_dir, args.ptm_iptm_tsv)
    pae_tsv         = os.path.join(output_dir, args.pae_tsv)
    clash_tsv       = os.path.join(output_dir, args.clash_tsv)
    final_merged_tsv= os.path.join(output_dir, args.final_merged_tsv)

    downselected_dir= os.path.join(output_dir, args.downselected_dir or "top20_pdbs")
    os.makedirs(output_dir, exist_ok=True)

    # (A) Collect pLDDT
    print("Collecting pLDDT scores from ranking_debug.json ...")
    all_plddt_entries = []
    for root, dirs, files in os.walk(af_base_dir):
        for f in files:
            if f == "ranking_debug.json":
                json_path = os.path.join(root, f)
                results = extract_plddt_scores(json_path)
                all_plddt_entries.extend(results)

    plddt_df = pd.DataFrame(all_plddt_entries, columns=[
        "Name", "Object_Name", "Model_Key", "Ranked_File", "pLDDT_Score"
    ])
    plddt_df["Rank"] = plddt_df["Ranked_File"].str.extract(r"ranked_(\d+)\.pdb").astype(int)
    plddt_df.sort_values(by=["pLDDT_Score"], ascending=False, inplace=True)
    plddt_df.to_csv(plddt_tsv, sep="\t", index=False)
    print(f"pLDDT => {plddt_tsv}")

    # (B) RMSD
    print("\nCalculating RMSD for each reference vs. AF predictions ...")
    rmsd_results = []
    for ref_file in os.listdir(rf_base_dir):
        if not ref_file.endswith(".pdb"):
            continue
        ref_filepath = os.path.join(rf_base_dir, ref_file)
        ref_name_noext = os.path.splitext(ref_file)[0]
        print(f"\nReference PDB: {ref_filepath}")
        
       	for root, dirs, files in os.walk(af_base_dir):
            relative = os.path.relpath(root, af_base_dir)
            top_part = relative.split(os.sep)[0]
          
            if top_part.lower() != ref_name_noext.lower():
                continue

            for f in files:
                if f.startswith("ranked_") and f.endswith(".pdb"):
                    model_path = os.path.join(root, f)
                 
                    try:
                        rank_str = f.split("_")[1].split(".")[0]
                        rank = int(rank_str)
                    except:
                        rank = -1

                    folder_two_up = os.path.basename(os.path.dirname(root))
                    name_col = folder_two_up + "/" + os.path.basename(root)
                    print(f"  => {model_path} (Rank {rank})")

                    rmsd_val = calculate_rmsd_multimer_robust(ref_filepath, model_path)
                  
                    if isinstance(rmsd_val, dict):
                        if "chain_comparisons" in rmsd_val:
                            avg_val = rmsd_val["average_rmsd"]
                            details = []
                            for (rc, mc, val_str) in rmsd_val["chain_comparisons"]:
                                # e.g. val_str = "2.30(1256atoms)"
                                # we do "A->B:2.30(1256atoms)"
                                details.append(f"{rc}->{mc}:{val_str}")
                            chain_details_str = "; ".join(details)
                            final_rmsd = avg_val
                        elif "fallback_rmsd" in rmsd_val:
                            final_rmsd = rmsd_val["fallback_rmsd"]
                            chain_details_str = "Fallback"
                        else:
                            final_rmsd = "N/A"
                            chain_details_str = "N/A"
                    else:
                        final_rmsd = "N/A"
                        chain_details_str = "N/A"

                    rmsd_results.append((
                        name_col,
                        ref_name_noext,
                        f.replace(".pdb", ""),
                        rank,
                        final_rmsd,
                        chain_details_str                    
                    ))

    rmsd_df = pd.DataFrame(rmsd_results, columns=[
        "Name", "Reference", "Model_Name", "Rank", "RMSD", "Chain_RMSD_Details"
    ])
    rmsd_df.to_csv(rmsd_tsv, sep="\t", index=False)
    print(f"RMSD => {rmsd_tsv}")

    # (C) pTM/ipTM, PAE, Clash
    print("\nGathering pTM/ipTM, PAE, and Clash ...")
    ptm_iptm_entries = []
    pae_entries = []
    clash_entries = []

    for root, dirs, files in os.walk(af_base_dir):
        for f in files:
            if f.startswith("ranked_") and f.endswith(".pdb"):
                pdb_path = os.path.join(root, f)
                model_name = os.path.splitext(f)[0]
                try:
                    rank_num = int(model_name.split("_")[1])
                except:
                    rank_num = -1

                folder_two_up = os.path.basename(os.path.dirname(root))
                name_col = folder_two_up + "/" + os.path.basename(root)

                ranking_debug_path = os.path.join(root, "ranking_debug.json")
                if not os.path.exists(ranking_debug_path):
                    continue
                model_id = f"model_{rank_num+1}_multimer_v3_pred_0"
                pae_json = os.path.join(root, f"pae_model_{rank_num+1}_multimer_v3_pred_0.json")

                pTM_val, ipTM_val = extract_ptm_iptm(ranking_debug_path, model_id)
                avg_pae = extract_average_pae(pae_json)
                clash_val = calculate_clash_score(pdb_path, distance_threshold=2.0)

                ptm_iptm_entries.append({
                    "Name": name_col,
                    "Object_Name": os.path.basename(root),
                    "Model_Name": model_name,
                    "Rank": rank_num,
                    "pTM (%)": round(pTM_val, 2) if not math.isnan(pTM_val) else "N/A",
                    "ipTM (%)": round(ipTM_val, 2) if not math.isnan(ipTM_val) else "N/A"
                })
                pae_entries.append({
                    "Name": name_col,
                    "Object_Name": os.path.basename(root),
                    "Model_Name": model_name,
                    "Rank": rank_num,
                    "Average_PAE": round(avg_pae, 2) if not math.isnan(avg_pae) else "N/A"
                })
                clash_entries.append({
                    "Name": name_col,
                    "Object_Name": os.path.basename(root),
                    "Model_Name": model_name,
                    "Rank": rank_num,
                    "Clashes": round(clash_val, 2) if not math.isnan(clash_val) else "N/A"
                })

    ptm_iptm_df = pd.DataFrame(ptm_iptm_entries)
    pae_df = pd.DataFrame(pae_entries)
    clash_df = pd.DataFrame(clash_entries)

    ptm_iptm_df.to_csv(ptm_iptm_tsv, sep="\t", index=False)
    pae_df.to_csv(pae_tsv, sep="\t", index=False)
    clash_df.to_csv(clash_tsv, sep="\t", index=False)
    print(f"pTM/ipTM => {ptm_iptm_tsv}")
    print(f"PAE => {pae_tsv}")
    print(f"Clashes => {clash_tsv}")

    # (D) Merge
    print("\nMerging all metrics ...")
    merged_all = plddt_df.merge(rmsd_df, on=["Name", "Rank"], how="left", suffixes=("", "_rmsd"))
    if "Object_Name_rmsd" in merged_all.columns:
        merged_all.drop(columns=["Object_Name_rmsd"], inplace=True)

    merged_all = merged_all.merge(ptm_iptm_df, on=["Name", "Rank"], how="left", suffixes=("", "_ptm"))
    if "Object_Name_ptm" in merged_all.columns:
        merged_all.drop(columns=["Object_Name_ptm"], inplace=True)

    merged_all = merged_all.merge(pae_df, on=["Name", "Rank"], how="left", suffixes=("", "_pae"))
    if "Object_Name_pae" in merged_all.columns:
        merged_all.drop(columns=["Object_Name_pae"], inplace=True)

    merged_all = merged_all.merge(clash_df, on=["Name", "Rank"], how="left", suffixes=("", "_clash"))
    if "Object_Name_clash" in merged_all.columns:
        merged_all.drop(columns=["Object_Name_clash"], inplace=True)

    merged_all.to_csv(final_merged_tsv, sep="\t", index=False)
    print(f"Final merged => {final_merged_tsv}")

    # (E) Copy Top 20 by pLDDT
    print("\nCopying top 20 by pLDDT ...")
    top_20_df = plddt_df.sort_values(by="pLDDT_Score", ascending=False).head(20)
    os.makedirs(downselected_dir, exist_ok=True)
    for _, row in top_20_df.iterrows():
        name_val = row["Name"]
        ranked_file = row["Ranked_File"]
        src = os.path.join(af_base_dir, name_val, ranked_file)
        tgt_subdir = os.path.join(downselected_dir, name_val)
        os.makedirs(tgt_subdir, exist_ok=True)
        if os.path.isfile(src):
            shutil.copy(src, tgt_subdir)
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
        "final_merged_tsv": final_merged_tsv
    }

###############################################################################
# 7) PDF Reporting
###############################################################################
def run_report(config, pdf_name="analysis_report.pdf"):
    """
    Generate a PDF report:
      1) Table of top 20 by pLDDT & RMSD
      2) Scatter: pLDDT vs RMSD (All)
      3) Scatter: pLDDT vs RMSD (Top 20)
      4) Table of additional metrics (Clashes, PAE, pTM, ipTM)
    """
    import sys
    import io
    import plotly.express as px

    plddt_file        = config["plddt_tsv"]
    rmsd_file         = config["rmsd_tsv"]
    clash_file        = config["clash_tsv"]
    pae_file          = config["pae_tsv"]
    ptm_file          = config["ptm_iptm_tsv"]
    final_merged_file = config["final_merged_tsv"]
    output_dir        = config["output_dir"]
    af_dir            = config["af_base_dir"]

    pdf_filepath = os.path.join(output_dir, pdf_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Generating PDF report ---")
    print(f"Output PDF => {pdf_filepath}")

    # 1) Load pLDDT data
    try:
        plddt_df = pd.read_csv(plddt_file, sep="\t")
    except Exception as e:
        print(f"Error reading pLDDT TSV: {e}")
        sys.exit(1)

    if "Ranked_File" in plddt_df.columns:
        plddt_df["Rank"] = plddt_df["Ranked_File"].str.extract(r"ranked_(\d+)\.pdb").astype(int)

    if plddt_df["pLDDT_Score"].max() <= 1.0:
        plddt_df["pLDDT_Score"] = plddt_df["pLDDT_Score"] * 100

    # 2) Load RMSD
    try:
        rmsd_df = pd.read_csv(rmsd_file, sep="\t")
    except Exception as e:
        print(f"Error reading RMSD TSV: {e}")
        sys.exit(1)

    merged = pd.merge(plddt_df, rmsd_df, on=["Name", "Rank"], how="inner")
    if merged.empty:
        print("No overlapping (Name,Rank) between pLDDT and RMSD. Stopping PDF.")
        return

    merged["pLDDT_Score"] = pd.to_numeric(merged["pLDDT_Score"], errors="coerce")
    merged["RMSD"] = pd.to_numeric(merged["RMSD"], errors="coerce")
    merged.dropna(subset=["pLDDT_Score", "RMSD"], inplace=True)

    merged_sorted = merged.sort_values(["pLDDT_Score", "RMSD"], ascending=[False, True])
    top_20 = merged_sorted.head(20).reset_index(drop=True)

    rank_colors = {"0": "blue", "1": "green", "2": "orange", "3": "red", "4": "purple"}

    # Scatter: All
    fig_all = px.scatter(
        merged,
        x="pLDDT_Score",
        y="RMSD",
        color=merged["Rank"].astype(str),
        color_discrete_map=rank_colors,
        hover_data={
            "Name":True, "Model_Name":True,
            "pLDDT_Score":":.2f","RMSD":":.2f","Chain_RMSD_Details":True
        },
        title="pLDDT vs RMSD (All Predictions)",
        labels={"pLDDT_Score":"pLDDT","RMSD":"RMSD(Å)","color":"Rank"}
    )
    fig_all.update_layout(template="plotly_white")

    # Scatter: Top 20
    fig_top20 = px.scatter(
        top_20,
        x="pLDDT_Score",
        y="RMSD",
        color=top_20["Rank"].astype(str),
        color_discrete_map=rank_colors,
        hover_data={
            "Name":True, "Model_Name":True,
            "pLDDT_Score":":.2f","RMSD":":.2f","Chain_RMSD_Details":True
        },
        title="pLDDT vs RMSD (Top 20)",
        labels={"pLDDT_Score":"pLDDT","RMSD":"RMSD(Å)","color":"Rank"}
    )
    fig_top20.update_layout(template="plotly_white")

    try:
        fig_all_png = fig_all.to_image(format="png", scale=2)
        fig_top20_png = fig_top20.to_image(format="png", scale=2)
    except Exception as e:
        print(f"Plotly figure conversion error: {e}")
        return

    # Table data
    table_data_top20 = [["#","Name","Model_Name","pLDDT","RMSD"]]
    for i, row in top_20.iterrows():
        table_data_top20.append([
            i+1,
            row["Name"],
            row["Model_Name"],
            f"{row['pLDDT_Score']:.2f}",
            f"{row['RMSD']:.2f}"
        ])

    # Additional metrics
    try:
        final_merged_df = pd.read_csv(final_merged_file, sep="\t")
    except Exception as e:
        print(f"Error reading final merged TSV: {e}")
        return

    top20_extra = pd.merge(top_20, final_merged_df, on=["Name","Rank"], suffixes=("", "_merged"), how="left")
    metric_table_data = [["Name","Model_Name","Clashes","Average PAE","pTM (%)","ipTM (%)"]]
    for _, row in top20_extra.iterrows():
        c = row.get("Clashes","N/A")
        p = row.get("Average_PAE","N/A")
        pTMval = row.get("pTM (%)","N/A")
        ipTMval= row.get("ipTM (%)","N/A")
        def fmt(x):
            if isinstance(x, (float,int)):
                return f"{x:.2f}"
            return str(x)
        metric_table_data.append([
            row["Name"],
            row["Model_Name"],
            fmt(c),
            fmt(p),
            fmt(pTMval),
            fmt(ipTMval)
        ])

    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch

    doc = SimpleDocTemplate(pdf_filepath, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    title_par = Paragraph(f"Analysis Report (AF dir: {af_dir})", styles["Title"])
    elements.append(title_par)
    elements.append(Spacer(1, 12))

    heading_top20 = Paragraph("Top 20 by pLDDT & RMSD", styles["Heading2"])
    elements.append(heading_top20)
    elements.append(Spacer(1, 12))

    tbl_top20 = Table(table_data_top20, colWidths=[30,130,130,60,60])
    tbl_style_top20 = TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#4F81BD")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,0),12),
        ("BOTTOMPADDING",(0,0),(-1,0),8),
        ("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#D9E1F2")),
        ("GRID",(0,0),(-1,-1),1,colors.black),
    ])
    tbl_top20.setStyle(tbl_style_top20)
    elements.append(tbl_top20)
    elements.append(Spacer(1, 24))

    scatter_all_heading = Paragraph("Scatter Plot: All Predictions", styles["Heading2"])
    elements.append(scatter_all_heading)
    elements.append(Spacer(1, 12))

    import io
    img_all_buf = io.BytesIO(fig_all_png)
    img_all = RLImage(img_all_buf, width=6.5*inch, height=4.5*inch)
    elements.append(img_all)
    elements.append(Spacer(1, 24))

    scatter_top20_heading = Paragraph("Scatter Plot: Top 20", styles["Heading2"])
    elements.append(scatter_top20_heading)
    elements.append(Spacer(1, 12))

    img_t20_buf = io.BytesIO(fig_top20_png)
    img_t20 = RLImage(img_t20_buf, width=6.5*inch, height=4.5*inch)
    elements.append(img_t20)
    elements.append(Spacer(1, 24))

    heading_metrics = Paragraph("Additional Metrics (Top 20)", styles["Heading2"])
    elements.append(heading_metrics)
    elements.append(Spacer(1, 12))

    tbl_metrics = Table(metric_table_data, colWidths=[130,130,60,60,60,60])
    style_metrics = TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#7030A0")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,0),12),
        ("BOTTOMPADDING",(0,0),(-1,0),8),
        ("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#E2D4F0")),
        ("GRID",(0,0),(-1,-1),1,colors.black),
    ])
    tbl_metrics.setStyle(style_metrics)
    elements.append(tbl_metrics)
    elements.append(Spacer(1, 24))

    try:
        doc.build(elements)
        print(f"\nPDF report generated at {pdf_filepath}\n")
    except Exception as e:
        print(f"Error building PDF: {e}")

###############################################################################
# 8) main()
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Full pipeline with chain-based RMSD (Hungarian + sequence) and PDF report.")
    parser.add_argument("--rf_base_dir", required=True, help="Path to reference PDBs")
    parser.add_argument("--af_base_dir", required=True, help="Path to AlphaFold output directories")
    parser.add_argument("--output_dir", required=True, help="Output directory for TSVs/PDF")
    parser.add_argument("--plddt_tsv", default="plddt.tsv")
    parser.add_argument("--rmsd_tsv", default="rmsd.tsv")
    parser.add_argument("--ptm_iptm_tsv", default="ptm_iptm.tsv")
    parser.add_argument("--pae_tsv", default="pae.tsv")
    parser.add_argument("--clash_tsv", default="clash.tsv")
    parser.add_argument("--final_merged_tsv", default="final_merged.tsv")
    parser.add_argument("--downselected_dir", default="top20_pdbs", help="Subfolder for top 20 PDBs")
    parser.add_argument("--pdf_name", default="analysis_report.pdf", help="PDF filename")

    args = parser.parse_args()
    config = run_pipeline(args)
    run_report(config, pdf_name=args.pdf_name)

if __name__ == "__main__":
    main()
