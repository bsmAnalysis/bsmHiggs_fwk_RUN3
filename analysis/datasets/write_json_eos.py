import json
import sys
from pathlib import Path

def write_json_eos(input_path, output_path=None):
    with open(input_path, "r") as f:
        data = json.load(f)

    for key, content in data.items():
        sample = key
        new_files = [(
            #f"root://eosuser.cern.ch//eos/user/g/georgia/RUN3/{sample}/"
            f"root://eosuser.cern.ch//eos/user/k/kpaschos/NTuples_2024/{sample}/"
            #f"root://eosuser.cern.ch//eos/user/a/ataxeidi/NTuples_2024/{sample}/"
            f"{sample}_{i}.root" )
                     
            for i in range(len(content["files"]))
        ]
        content["files"] = new_files

    output_path = output_path or f"eos_{Path(input_path).name}"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2] 
    write_json_eos(input_file, output_file)
