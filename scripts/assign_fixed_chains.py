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

    designed_chain_list = args.chain_list.strip().split()
    if not designed_chain_list:
        print("No chains provided in --chain_list.", file=sys.stderr)
        sys.exit(1)

    my_dict = {}
    with open(args.input_path, 'r') as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pdb_name = data['name']

            all_chain_list = [item[-1:] for item in list(data) if item[:9] == 'seq_chain']
            fixed_chain_list = [c for c in all_chain_list if c not in designed_chain_list]

            my_dict[pdb_name] = [designed_chain_list, fixed_chain_list]

    with open(args.output_path, 'w') as outfile:
        outfile.write(json.dumps(my_dict) + "\n")

    print("Assigned designed and fixed chains. Output written to:", args.output_path)


if __name__ == "__main__":
    main()
