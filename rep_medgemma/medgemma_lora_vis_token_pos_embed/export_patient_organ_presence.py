#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Resized,
    SpatialPadd,
    Transposed,
)

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


ALL_TARGET_KEYS = [
    "lung",
    "heart",
    "esophagus",
    "liver",
    "gallbladder",
    "stomach",
    "pancreas",
    "spleen",
    "kidney",
    "aorta",
    "trachea",
    "rib",
]


def get_organ_ids_for_key(report_key):
    key = report_key.lower().strip()
    if "lung" in key:
        return [10, 11, 12, 13, 14]
    if "heart" in key:
        return [51, 61]
    if "aorta" in key:
        return [52]
    if "esophagus" in key:
        return [15]
    if "trachea" in key:
        return [16]
    if "rib" in key:
        return list(range(92, 116))
    if "liver" in key:
        return [5]
    if "gallbladder" in key:
        return [4]
    if "stomach" in key:
        return [6]
    if "pancreas" in key:
        return [7]
    if "spleen" in key:
        return [1]
    if "kidney" in key:
        return [2, 3]
    return []


def build_mask_transform():
    # Mirror train.py preprocessing path for mask visibility.
    return Compose(
        [
            LoadImaged(keys=["mask"], reader="ITKReader", image_only=True),
            EnsureChannelFirstd(keys=["mask"]),
            Transposed(keys=["mask"], indices=(0, 3, 2, 1)),
            SpatialPadd(
                keys=["mask"],
                spatial_size=(112, 256, 352),
                mode="constant",
                constant_values=0,
            ),
            Resized(keys=["mask"], spatial_size=(112, 256, 352), mode="nearest"),
            EnsureTyped(keys=["mask"]),
        ]
    )


def normalize_image_path(image_path):
    return image_path.replace("/data_sym_sym/", "/data_sym/")


def image_to_mask_path(image_path):
    return normalize_image_path(image_path).replace("images", "masks")


def base_id_from_image_path(image_path):
    fname = os.path.basename(image_path)
    return fname.replace(".nii.gz", "").replace(".nii", "")


def resolve_patient_id(image_path, reports_json):
    base_id = base_id_from_image_path(image_path)
    if base_id in reports_json:
        return base_id
    if len(base_id.split("_")) > 1:
        parent_id = base_id.rsplit("_", 1)[0]
        if parent_id in reports_json:
            return parent_id
    return None


def empty_organ_dict():
    return {organ: 0 for organ in ALL_TARGET_KEYS}


def report_presence_from_patient_data(patient_data):
    out = empty_organ_dict()
    for organ in ALL_TARGET_KEYS:
        text = patient_data.get(organ, "")
        if isinstance(text, str) and len(text.strip()) >= 3:
            out[organ] = 1
    return out


def mask_presence_from_tensor(mask_tensor):
    out = empty_organ_dict()
    for organ in ALL_TARGET_KEYS:
        organ_ids = get_organ_ids_for_key(organ)
        for organ_id in organ_ids:
            if torch.any(mask_tensor == organ_id):
                out[organ] = 1
                break
    return out


def init_patient_record(split, patient_id):
    return {
        "split": split,
        "patient_id": patient_id,
        "scan_count": 0,
        "report": empty_organ_dict(),
        "mask": empty_organ_dict(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export per-patient organ presence (0/1) for report and mask."
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default=str(PROJECT_ROOT / 'data_sym/image_first_dataset.csv'),
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default=str(PROJECT_ROOT / 'data_sym/combined_desc_conc_v2.json'),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["training", "validation"],
        help="Dataset splits to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / 'rep_medgemma/medgemma_lora_vis_token_pos_embed/analysis_outputs/patient_organ_presence'),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Progress print interval over valid rows.",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Optional cap per split for quick smoke tests.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading CSV: {args.csv_file}")
    df = pd.read_csv(args.csv_file)

    print(f"Loading JSON: {args.json_file}")
    with open(args.json_file, "r") as f:
        reports_json = json.load(f)

    transform = build_mask_transform()

    results = {}
    meta_rows = []
    error_rows = []

    for split in args.splits:
        print(f"\n=== Split: {split} ===")
        df_split = df[df["split"] == split].reset_index(drop=True)

        split_results = {}
        valid_seen = 0
        mask_success = 0
        mask_errors = 0

        for _, row in df_split.iterrows():
            image_path = row["image_path"]
            patient_id = resolve_patient_id(image_path, reports_json)
            if patient_id is None:
                continue

            if args.max_scans is not None and valid_seen >= args.max_scans:
                break

            valid_seen += 1

            record = split_results.get(patient_id)
            if record is None:
                record = init_patient_record(split, patient_id)
                split_results[patient_id] = record

            record["scan_count"] += 1

            patient_data = reports_json.get(patient_id, {})
            report_presence = report_presence_from_patient_data(patient_data)
            for organ in ALL_TARGET_KEYS:
                if report_presence[organ] == 1:
                    record["report"][organ] = 1

            mask_path = image_to_mask_path(image_path)
            try:
                data = transform({"mask": mask_path})
                mask = data["mask"]
                mask_tensor = (
                    mask.as_tensor() if hasattr(mask, "as_tensor") else torch.as_tensor(mask)
                )
                mask_presence = mask_presence_from_tensor(mask_tensor)
                for organ in ALL_TARGET_KEYS:
                    if mask_presence[organ] == 1:
                        record["mask"][organ] = 1
                mask_success += 1
            except Exception as exc:
                mask_errors += 1
                error_rows.append(
                    {
                        "split": split,
                        "patient_id": patient_id,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "error": str(exc),
                    }
                )

            if valid_seen % args.progress_every == 0:
                print(
                    f"  processed {valid_seen} valid rows "
                    f"(unique patients={len(split_results)}, "
                    f"mask_success={mask_success}, mask_errors={mask_errors})"
                )

        print(
            f"Done split={split}: valid_rows={valid_seen}, "
            f"unique_patients={len(split_results)}, "
            f"mask_success={mask_success}, mask_errors={mask_errors}"
        )

        results[split] = split_results
        meta_rows.append(
            {
                "split": split,
                "rows_in_split": int(len(df_split)),
                "valid_rows_processed": int(valid_seen),
                "unique_patients": int(len(split_results)),
                "mask_success": int(mask_success),
                "mask_errors": int(mask_errors),
            }
        )

    json_path = os.path.join(args.output_dir, "patient_organ_presence.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    wide_rows = []
    long_rows = []

    for split, patient_dict in results.items():
        for patient_id, rec in patient_dict.items():
            wide_row = {
                "split": split,
                "patient_id": patient_id,
                "scan_count": int(rec["scan_count"]),
            }

            for organ in ALL_TARGET_KEYS:
                wide_row[f"report_{organ}"] = int(rec["report"][organ])
                wide_row[f"mask_{organ}"] = int(rec["mask"][organ])

                long_rows.append(
                    {
                        "split": split,
                        "patient_id": patient_id,
                        "organ": organ,
                        "report_present": int(rec["report"][organ]),
                        "mask_present": int(rec["mask"][organ]),
                    }
                )

            wide_rows.append(wide_row)

    wide_df = pd.DataFrame(wide_rows).sort_values(["split", "patient_id"]).reset_index(
        drop=True
    )
    long_df = pd.DataFrame(long_rows).sort_values(
        ["split", "patient_id", "organ"]
    ).reset_index(drop=True)

    wide_csv_path = os.path.join(args.output_dir, "patient_organ_presence_wide.csv")
    long_csv_path = os.path.join(args.output_dir, "patient_organ_presence_long.csv")
    meta_csv_path = os.path.join(args.output_dir, "run_metadata.csv")

    wide_df.to_csv(wide_csv_path, index=False)
    long_df.to_csv(long_csv_path, index=False)
    pd.DataFrame(meta_rows).to_csv(meta_csv_path, index=False)

    print(f"Saved wide CSV: {wide_csv_path}")
    print(f"Saved long CSV: {long_csv_path}")
    print(f"Saved metadata CSV: {meta_csv_path}")

    if error_rows:
        err_csv_path = os.path.join(args.output_dir, "mask_processing_errors.csv")
        pd.DataFrame(error_rows).to_csv(err_csv_path, index=False)
        print(f"Saved errors CSV: {err_csv_path}")


if __name__ == "__main__":
    main()
