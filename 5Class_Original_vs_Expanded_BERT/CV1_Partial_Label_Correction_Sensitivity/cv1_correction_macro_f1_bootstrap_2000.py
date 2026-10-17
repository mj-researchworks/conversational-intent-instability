#!/usr/bin/env python3
"""Reproduce the Round 3 CV1 correction paired bootstrap analysis."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


LABELS = ["Chitchat", "Interpret", "Reveal", "Revise", "Inquire"]
LABEL_TO_I = {label: i for i, label in enumerate(LABELS)}


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(["conversation_id", "turn_index"]).reset_index(drop=True)


def table_for(df: pd.DataFrame) -> np.ndarray:
    cm = np.zeros((5, 5), dtype=np.int64)
    for true_label, predicted_label in zip(df.true_label, df.predicted_label):
        cm[LABEL_TO_I[true_label], LABEL_TO_I[predicted_label]] += 1
    return cm


def macro_from_cm(cm: np.ndarray) -> float:
    f1 = []
    for i in range(5):
        tp = cm[i, i]
        row = cm[i, :].sum()
        col = cm[:, i].sum()
        denominator = row + col
        f1.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(f1))


def conversation_tables(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(conversation_id): table_for(group)
        for conversation_id, group in df.groupby("conversation_id", sort=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_package_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--package-root",
        type=Path,
        default=default_package_root,
        help=(
            "Reproducibility-package root. Defaults to the package containing "
            "this script."
        ),
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help=(
            "Destination for regenerated results. Defaults to the packaged "
            "Results_Analysis folder."
        ),
    )
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    experiment_root = package_root / "5Class_Original_vs_Expanded_BERT"
    predictions_root = experiment_root / "Original_vs_Expanded_Predictions"
    correction_root = experiment_root / "CV1_Partial_Label_Correction_Sensitivity"
    output = (
        args.output_folder.resolve()
        if args.output_folder is not None
        else experiment_root / "Results_Analysis"
    )
    output.mkdir(parents=True, exist_ok=True)

    original = load(predictions_root / "original" / "original_cv1_test_predictions.csv")
    expanded = load(predictions_root / "expanded" / "expanded_cv1_test_predictions.csv")
    corrected = load(correction_root / "corrected_cv1_test_predictions.csv")

    match_columns = ["conversation_id", "turn_index", "true_label"]
    for name, df in [("expanded", expanded), ("corrected", corrected)]:
        assert len(df) == len(original), name
        assert (
            df[match_columns].astype(str).values
            == original[match_columns].astype(str).values
        ).all(), name

    original_tables = conversation_tables(original)
    expanded_tables = conversation_tables(expanded)
    corrected_tables = conversation_tables(corrected)
    conversation_ids = np.array(sorted(original_tables.keys()))
    assert set(conversation_ids) == set(expanded_tables) == set(corrected_tables)
    conversation_count = len(conversation_ids)

    point = {
        "Original": macro_from_cm(table_for(original)),
        "Expanded": macro_from_cm(table_for(expanded)),
        "Corrected": macro_from_cm(table_for(corrected)),
    }
    print("Point macro F1:", point)
    print("Corrected-Expanded", point["Corrected"] - point["Expanded"])
    print("Corrected-Original", point["Corrected"] - point["Original"])

    rng = np.random.default_rng(42)
    bootstrap_count = 2000
    corrected_minus_expanded = []
    corrected_minus_original = []
    all_draws = np.empty(
        (bootstrap_count, conversation_count), dtype=np.int16
    )

    for bootstrap_index in range(bootstrap_count):
        indices = rng.integers(
            0, conversation_count, size=conversation_count
        )
        all_draws[bootstrap_index] = indices
        sampled_ids = conversation_ids[indices]
        zero = np.zeros((5, 5), dtype=np.int64)
        original_cm = sum(
            (original_tables[cid] for cid in sampled_ids), start=zero.copy()
        )
        expanded_cm = sum(
            (expanded_tables[cid] for cid in sampled_ids), start=zero.copy()
        )
        corrected_cm = sum(
            (corrected_tables[cid] for cid in sampled_ids), start=zero.copy()
        )
        original_macro = macro_from_cm(original_cm)
        expanded_macro = macro_from_cm(expanded_cm)
        corrected_macro = macro_from_cm(corrected_cm)
        corrected_minus_expanded.append(corrected_macro - expanded_macro)
        corrected_minus_original.append(corrected_macro - original_macro)

    corrected_minus_expanded = np.array(corrected_minus_expanded)
    corrected_minus_original = np.array(corrected_minus_original)
    rows = []
    for comparison, observed, values in [
        (
            "Corrected - Expanded",
            point["Corrected"] - point["Expanded"],
            corrected_minus_expanded,
        ),
        (
            "Corrected - Original",
            point["Corrected"] - point["Original"],
            corrected_minus_original,
        ),
    ]:
        lower, upper = np.percentile(values, [2.5, 97.5])
        rows.append(
            {
                "Comparison": comparison,
                "Observed Macro F1 difference": observed,
                "Lower 95% limit": lower,
                "Upper 95% limit": upper,
                "resamples": bootstrap_count,
                "seed": 42,
                "test_conversations": conversation_count,
                "test_turns": len(original),
            }
        )

    result = pd.DataFrame(rows)
    print("\nRESULT")
    print(result.to_string(index=False))
    result.to_csv(output / "cv1_correction_macro_f1_bootstrap_2000.csv", index=False)
    np.save(
        output / "cv1_correction_bootstrap_draw_indices_seed42_2000.npy",
        all_draws,
    )
    pd.DataFrame([point]).to_csv(
        output / "cv1_macro_f1_point_scores.csv", index=False
    )
    print("\nSaved", output)


if __name__ == "__main__":
    main()
