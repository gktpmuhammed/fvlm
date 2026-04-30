import json
import os
from pathlib import Path
import csv
import os

def combine_desc_conc(desc_file, conc_file, output_file):
    # read descriptions and conclusion files
    with open(desc_file, 'r') as df:
        descriptions = json.load(df)
    with open(conc_file, 'r') as cf:
        conclusions = json.load(cf)

    combined_data = {}

    for patient_id in set(descriptions.keys()).union(conclusions.keys()):
        desc_organs = descriptions.get(patient_id, {})
        conc_organs = conclusions.get(patient_id, {})

        # union of all organs that appear in either file
        all_organs = set(desc_organs.keys()).union(conc_organs.keys())

        combined_data[patient_id] = {}

        for organ in all_organs:
            organ_desc = desc_organs.get(organ, "")
            organ_conc = conc_organs.get(organ, "")

            combined_text = f"{organ_desc} {organ_conc}".strip()
            combined_data[patient_id][organ] = combined_text

    # write combined data to output file
    with open(output_file, 'w') as of:
        json.dump(combined_data, of, indent=4)

# example usage
if __name__ == "__main__":
    project_root = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
    data_root = Path(os.getenv("DATA_ROOT", project_root / "data"))

    desc_file = str(data_root / "val_train_combined_desc.json")
    conc_file = str(data_root / "val_train_combined_conc.json")
    output_file = str(data_root / "combined_desc_conc.json")
    combine_desc_conc(desc_file, conc_file, output_file)
