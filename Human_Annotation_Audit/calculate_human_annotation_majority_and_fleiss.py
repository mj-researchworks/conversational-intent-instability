#!/usr/bin/env python3
"""Reproduce the final three-annotator CV1 audit workbook and statistics."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = ["Chitchat", "Interpret", "Reveal", "Revise", "Inquire"]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "annotation_folder",
        type=Path,
        help="Folder containing the final audit workbook and three annotator workbooks.",
    )
    parser.add_argument(
        "output_workbook",
        type=Path,
        help="Destination for the regenerated final audit workbook.",
    )
    args = parser.parse_args()

    folder = args.annotation_folder
    final_reference_path = (
        folder
        / "CV1_100_human_audit_FINAL_3annotators_with_majority_and_source_preservation.xlsx"
    )
    annotator_paths = [
        folder / f"CV1_100_human_audit_ANNOTATOR_{i}.xlsx"
        for i in (1, 2, 3)
    ]

    # The final workbook supplies the source/provenance fields that were hidden
    # from annotators. Original ratings are always reloaded from the three
    # separate annotator workbooks below.
    master = pd.read_excel(final_reference_path, sheet_name="Final_Annotations")
    annotations = []
    for index, path in enumerate(annotator_paths, start=1):
        annotator = pd.read_excel(path, sheet_name="Annotation")
        # The public annotation sheets intentionally retain only the fields
        # needed to reproduce the audit.  Confidence, ambiguity, and notes
        # were unused optional columns and are not required for the analysis.
        annotator = annotator[["audit_id", "annotator_label"]].copy()
        annotator = annotator.rename(
            columns={
                "annotator_label": f"annotator_{index}_label",
            }
        )
        annotations.append(annotator)

    derived_columns = {
        "human_majority_label",
        "unanimous_3of3",
        "three_way_disagreement",
        "source_intent_preserved_by_majority",
    }
    final = master.drop(
        columns=[
            column
            for column in master.columns
            if column.startswith("annotator_") or column in derived_columns
        ],
        errors="ignore",
    ).copy()
    for annotator in annotations:
        final = final.merge(annotator, on="audit_id", how="left")

    label_columns = [f"annotator_{i}_label" for i in (1, 2, 3)]
    if len(final) != 100:
        raise ValueError(f"Expected 100 audit rows; found {len(final)}")
    for column in label_columns:
        if final[column].isna().any():
            raise ValueError(f"Missing ratings in {column}")
        invalid = sorted(set(final[column]) - set(LABELS))
        if invalid:
            raise ValueError(f"Invalid ratings in {column}: {invalid}")

    def majority(row: pd.Series) -> str:
        values = [row[column] for column in label_columns if pd.notna(row[column])]
        counts = Counter(values)
        if not counts:
            return np.nan
        most_common = counts.most_common()
        if most_common[0][1] >= 2:
            return most_common[0][0]
        return "NO_MAJORITY"

    def unanimous(row: pd.Series) -> bool:
        values = [row[column] for column in label_columns if pd.notna(row[column])]
        return len(values) == 3 and len(set(values)) == 1

    final["human_majority_label"] = final.apply(majority, axis=1)
    final["unanimous_3of3"] = final.apply(unanimous, axis=1)
    final["three_way_disagreement"] = (
        final[label_columns].nunique(axis=1) == 3
    )
    final["source_intent_preserved_by_majority"] = final[
        "human_majority_label"
    ].eq(final["source_gt_intent_hidden"])

    rating_counts = []
    for _, row in final.iterrows():
        values = [row[column] for column in label_columns]
        rating_counts.append(
            [sum(value == label for value in values) for label in LABELS]
        )
    rating_counts = np.array(rating_counts, dtype=float)
    item_count = rating_counts.shape[0]
    ratings_per_item = rating_counts.sum(axis=1)[0]
    agreement_per_item = (
        (rating_counts**2).sum(axis=1) - ratings_per_item
    ) / (ratings_per_item * (ratings_per_item - 1))
    observed_agreement = agreement_per_item.mean()
    label_proportions = rating_counts.sum(axis=0) / (
        item_count * ratings_per_item
    )
    expected_agreement = (label_proportions**2).sum()
    fleiss_kappa = (observed_agreement - expected_agreement) / (
        1 - expected_agreement
    )

    summary = pd.DataFrame(
        [
            {
                "n_items": len(final),
                "fleiss_kappa": fleiss_kappa,
                "unanimous_count": int(final["unanimous_3of3"].sum()),
                "unanimous_percent": final["unanimous_3of3"].mean() * 100,
                "three_way_disagreement_count": int(
                    final["three_way_disagreement"].sum()
                ),
                "majority_source_preserved_count": int(
                    final["source_intent_preserved_by_majority"].sum()
                ),
                "majority_source_preserved_percent": final[
                    "source_intent_preserved_by_majority"
                ].mean()
                * 100,
            }
        ]
    )
    class_summary = (
        final.groupby("source_gt_intent_hidden")
        .agg(
            audited_count=("audit_id", "count"),
            preserved_count=("source_intent_preserved_by_majority", "sum"),
        )
        .reset_index()
    )
    class_summary["preserved_percent"] = (
        class_summary["preserved_count"]
        / class_summary["audited_count"]
        * 100
    )
    confusion = pd.crosstab(
        final["source_gt_intent_hidden"], final["human_majority_label"]
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)

    args.output_workbook.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output_workbook, engine="openpyxl") as writer:
        final.to_excel(writer, sheet_name="Final_Annotations", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        class_summary.to_excel(
            writer, sheet_name="Class_Preservation", index=False
        )
        confusion.to_excel(writer, sheet_name="Source_vs_Human_Majority")

    print("Saved", args.output_workbook)
    print(summary.to_string(index=False))
    print("\nClass preservation")
    print(class_summary.to_string(index=False))
    print("\nSource vs human majority")
    print(confusion.to_string())


if __name__ == "__main__":
    main()
