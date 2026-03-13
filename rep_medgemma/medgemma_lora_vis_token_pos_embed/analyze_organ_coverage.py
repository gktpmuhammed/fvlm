#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    # Mirror train.py preprocessing path for masks.
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


def resolve_patient_id(image_path, reports_json):
    fname = os.path.basename(image_path)
    base_id = fname.replace(".nii.gz", "").replace(".nii", "")
    if base_id in reports_json:
        return base_id
    if len(base_id.split("_")) > 1:
        parent_id = base_id.rsplit("_", 1)[0]
        if parent_id in reports_json:
            return parent_id
    return None


def normalize_image_path(image_path):
    return image_path.replace("/data_sym_sym/", "/data_sym/")


def image_to_mask_path(image_path):
    return normalize_image_path(image_path).replace("images", "masks")


def build_valid_rows(df_split, reports_json, max_scans=None):
    valid_rows = []
    for _, row in df_split.iterrows():
        pid = resolve_patient_id(row["image_path"], reports_json)
        if pid is None:
            continue
        valid_rows.append((row["image_path"], pid))
        if max_scans is not None and len(valid_rows) >= max_scans:
            break
    return valid_rows


def organ_visible_in_mask(mask_tensor, organ_ids):
    for organ_id in organ_ids:
        if torch.any(mask_tensor == organ_id):
            return True
    return False


def count_split(valid_rows, reports_json, transform, progress_every=500):
    report_counts = Counter({k: 0 for k in ALL_TARGET_KEYS})
    visible_counts = Counter({k: 0 for k in ALL_TARGET_KEYS})
    load_errors = []

    total = len(valid_rows)
    mask_success = 0

    for idx, (image_path, pid) in enumerate(valid_rows, start=1):
        patient_data = reports_json.get(pid, {})

        for organ in ALL_TARGET_KEYS:
            text = patient_data.get(organ, "")
            if isinstance(text, str) and len(text.strip()) >= 3:
                report_counts[organ] += 1

        mask_path = image_to_mask_path(image_path)
        try:
            data = transform({"mask": mask_path})
            mask = data["mask"]
            mask_tensor = (
                mask.as_tensor() if hasattr(mask, "as_tensor") else torch.as_tensor(mask)
            )
            mask_success += 1

            for organ in ALL_TARGET_KEYS:
                organ_ids = get_organ_ids_for_key(organ)
                if organ_visible_in_mask(mask_tensor, organ_ids):
                    visible_counts[organ] += 1
        except Exception as exc:
            load_errors.append({"mask_path": mask_path, "error": str(exc)})

        if idx % progress_every == 0 or idx == total:
            print(
                f"  processed {idx}/{total} "
                f"(mask_success={mask_success}, errors={len(load_errors)})"
            )

    return report_counts, visible_counts, total, mask_success, load_errors


def counts_to_df(split, report_counts, visible_counts, total, mask_success):
    rows = []
    for organ in ALL_TARGET_KEYS:
        rcount = int(report_counts[organ])
        vcount = int(visible_counts[organ])
        rows.append(
            {
                "split": split,
                "organ": organ,
                "num_scans": total,
                "num_masks_processed": mask_success,
                "report_count": rcount,
                "report_pct": (100.0 * rcount / total) if total else 0.0,
                "visible_count": vcount,
                "visible_pct": (100.0 * vcount / mask_success) if mask_success else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_split(df_split, output_dir, split):
    sns.set_theme(style="whitegrid", context="talk")

    counts_long = df_split.melt(
        id_vars=["organ"],
        value_vars=["report_count", "visible_count"],
        var_name="source",
        value_name="count",
    )
    counts_long["source"] = counts_long["source"].map(
        {
            "report_count": "Report (non-empty)",
            "visible_count": "Visible in preprocessed mask",
        }
    )

    pct_long = df_split.melt(
        id_vars=["organ"],
        value_vars=["report_pct", "visible_pct"],
        var_name="source",
        value_name="percent",
    )
    pct_long["source"] = pct_long["source"].map(
        {
            "report_pct": "Report (non-empty)",
            "visible_pct": "Visible in preprocessed mask",
        }
    )

    palette = {
        "Report (non-empty)": "#2A6F97",
        "Visible in preprocessed mask": "#D1495B",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), constrained_layout=True)

    sns.barplot(
        data=counts_long,
        x="organ",
        y="count",
        hue="source",
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_title(f"{split.capitalize()}: Organ Coverage Counts")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Number of scans")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(title="")

    sns.barplot(
        data=pct_long,
        x="organ",
        y="percent",
        hue="source",
        palette=palette,
        ax=axes[1],
    )
    axes[1].set_title(f"{split.capitalize()}: Organ Coverage Percentages")
    axes[1].set_xlabel("Organ")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylim(0, 105)
    axes[1].legend(title="")

    out_png = os.path.join(output_dir, f"{split}_organ_coverage.png")
    out_pdf = os.path.join(output_dir, f"{split}_organ_coverage.pdf")
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)

    return out_png, out_pdf


def main():
    parser = argparse.ArgumentParser(description="Analyze report and mask organ coverage.")
    parser.add_argument(
        "--csv-file",
        type=str,
        default="/home/muhammedg/fvlm/data_sym/image_first_dataset.csv",
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default="/home/muhammedg/fvlm/data_sym/combined_desc_conc_v2.json",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["training", "validation"],
        help="Dataset splits to analyze (e.g., training validation).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/muhammedg/fvlm/rep_medgemma/medgemma_lora_vis_token_pos_embed/analysis_outputs/organ_coverage",
    )
    parser.add_argument("--progress-every", type=int, default=500)
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

    all_df = []
    run_meta = []

    for split in args.splits:
        print(f"\n=== Split: {split} ===")
        df_split = df[df["split"] == split].reset_index(drop=True)
        valid_rows = build_valid_rows(
            df_split, reports_json, max_scans=args.max_scans
        )
        print(f"Rows in split: {len(df_split)}")
        print(f"Valid rows after report id matching: {len(valid_rows)}")

        report_counts, visible_counts, total, mask_success, load_errors = count_split(
            valid_rows=valid_rows,
            reports_json=reports_json,
            transform=transform,
            progress_every=args.progress_every,
        )

        split_df = counts_to_df(
            split=split,
            report_counts=report_counts,
            visible_counts=visible_counts,
            total=total,
            mask_success=mask_success,
        )
        all_df.append(split_df)

        png_path, pdf_path = plot_split(split_df, args.output_dir, split)
        print(f"Saved plot: {png_path}")
        print(f"Saved plot: {pdf_path}")

        if load_errors:
            err_file = os.path.join(args.output_dir, f"{split}_mask_load_errors.csv")
            pd.DataFrame(load_errors).to_csv(err_file, index=False)
            print(f"Saved mask load errors: {err_file}")

        run_meta.append(
            {
                "split": split,
                "rows_in_split": int(len(df_split)),
                "rows_with_report_match": int(len(valid_rows)),
                "mask_success": int(mask_success),
                "mask_errors": int(len(load_errors)),
            }
        )

    result_df = pd.concat(all_df, ignore_index=True)
    result_csv = os.path.join(args.output_dir, "organ_coverage_counts.csv")
    result_df.to_csv(result_csv, index=False)
    print(f"\nSaved counts table: {result_csv}")

    meta_csv = os.path.join(args.output_dir, "run_metadata.csv")
    pd.DataFrame(run_meta).to_csv(meta_csv, index=False)
    print(f"Saved metadata table: {meta_csv}")


if __name__ == "__main__":
    main()
