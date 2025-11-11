import pandas as pd
import json

def cross_check_retention(csv_path, findings_json, impressions_json, threshold=70):
    df = pd.read_csv(csv_path)
    organ_columns = [col.replace('_retained_pct', '') for col in df.columns if '_retained_pct' in col]
    organ_stats = {organ: {"mentioned": 0, "lost": 0} for organ in organ_columns}

    for idx, row in df.iterrows():
        scan_id = str(row["patient_id"]) if "patient_id" in row else str(idx)
        organs_in_report = set()
        for dct in [findings_json.get(scan_id, {}), impressions_json.get(scan_id, {})]:
            for k in dct.keys():
                if k in organ_columns:
                    organs_in_report.add(k)
        for organ in organs_in_report:
            col = f"{organ}_retained_pct"
            organ_stats[organ]["mentioned"] += 1
            retained = row[col] if col in row else None
            if retained is not None and retained < threshold:
                organ_stats[organ]["lost"] += 1

    results = []
    for organ, counts in organ_stats.items():
        mentioned = counts["mentioned"]
        lost = counts["lost"]
        percent_lost = (lost / mentioned * 100) if mentioned > 0 else 0
        results.append({"organ": organ, "mentioned_in_report": mentioned, "lost_by_threshold": lost, "percent_lost": percent_lost})

    result_df = pd.DataFrame(results)
    result_df.to_csv("organ_report_loss_percentages.csv", index=False)
    print(result_df)
    return result_df

# Usage:
find_path = "/home/muhammedg/fvlm/data/conc_info.json"
imp_path = "/home/muhammedg/fvlm/data/desc_info.json"
csv_file = "/home/muhammedg/fvlm/output/organ_voxel_pct_analysis_debug/per_scan_organ_voxel_percentages.csv"
findings = json.load(open(find_path))
impressions = json.load(open(imp_path))
cross_check_retention(csv_file, findings, impressions, threshold=90)

