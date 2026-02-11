#!/usr/bin/env python
import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Assign fixed and designed chains for ProteinMPNN input.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON lines file (e.g., parsed_pdbs.jsonl).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON lines file (e.g., assigned_pdbs.jsonl).")
    parser.add_argument("--chain_list", type=str, required=True, help="Space-separated list of chains to design, e.g. 'A B'.")

    args = parser.parse_args()

    chain_list = args.chain_list.strip().split()
    if not chain_list:
        print("No chains provided in --chain_list.", file=sys.stderr)
        sys.exit(1)

    # In this example, we assume all chains provided in chain_list are "designed"
    # and no chains are considered fixed. If you need a different logic, adjust below.
    designed_chains = chain_list
    fixed_chains = []  # Adjust if necessary

    with open(args.input_path, 'r') as infile, open(args.output_path, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Add or update chain_id_dict field
            # The expected format for ProteinMPNN (check documentation if different):
            # {"fixed": [...], "designed": [...]}
            chain_id_dict = {
                "fixed": fixed_chains,
                "designed": designed_chains
            }

            # Merge this into the data
            data["chain_id_dict"] = chain_id_dict

            outfile.write(json.dumps(data) + "\n")

    print("Assigned designed and fixed chains. Output written to:", args.output_path)


if __name__ == "__main__":
    main()

