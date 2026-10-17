# %% [markdown]
# # Part 3 Turn-Wise Reanalysis
#
# This post-hoc script follows the latest advisor document exactly for the
# turn-wise analysis:
#
# - Uses saved held-out BERT test predictions; no model retraining.
# - Uses fixed five-label Macro F1 for aggregate turn-wise Macro F1:
#   `labels=VALID_LABELS`, `average="macro"`, `zero_division=0`.
# - Uses whole-conversation bootstrap blocks with 5,000 replicates and seed 42.
# - Displays class-wise F1 as NA when an intent has zero true support and no
#   true/false positives in that turn bucket.
# - Saves turn support/count tables, bootstrap intervals, corrected figures,
#   and bootstrap metadata.
#
# Input expected in Colab:
# `/content/bert_test_predictions_posthoc.csv`
#
# Required columns:
# `conversation_id`, `true_label`, `predicted_label`, `turn_bucket`

# %%
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    display
except NameError:
    display = print


# =========================
# CONFIG
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
PREDICTIONS_CSV = Path(
    os.environ.get(
        "TURNWISE_PREDICTIONS_CSV",
        SCRIPT_DIR / "bert_test_predictions_posthoc.csv",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "TURNWISE_OUTPUT_DIR",
        SCRIPT_DIR / "turnwise_bootstrap_5000_results",
    )
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_LABELS = ["Chitchat", "Interpret", "Reveal", "Revise", "Inquire"]
TURN_ORDER = ["1", "2", "3", "4", "5+"]

N_BOOT = int(os.environ.get("TURNWISE_BOOTSTRAP_REPLICATES", "5000"))
BOOT_SEED = int(os.environ.get("TURNWISE_BOOTSTRAP_SEED", "42"))

print("Predictions:", PREDICTIONS_CSV)
print("Output dir:", OUTPUT_DIR)
print("Bootstrap unit: conversation_id")
print("Bootstrap replicates:", N_BOOT)
print("Bootstrap seed:", BOOT_SEED)
print("Macro F1 convention: fixed five-label Macro F1, zero_division=0")


# %%
# =========================
# LOAD AND VALIDATE PREDICTIONS
# =========================

df = pd.read_csv(PREDICTIONS_CSV)

required = {"conversation_id", "true_label", "predicted_label", "turn_bucket"}
missing = required - set(df.columns)
assert not missing, f"Missing required columns: {missing}"

df["conversation_id"] = df["conversation_id"].astype(str)
df["true_label"] = df["true_label"].astype(str)
df["predicted_label"] = df["predicted_label"].astype(str)
df["turn_bucket"] = df["turn_bucket"].astype(str)

assert set(df["true_label"]).issubset(set(VALID_LABELS))
assert set(df["predicted_label"]).issubset(set(VALID_LABELS))

df["turn_bucket"] = pd.Categorical(df["turn_bucket"], categories=TURN_ORDER, ordered=True)

print("Rows:", len(df))
print("Conversations:", df["conversation_id"].nunique())
print("\nTrue-label counts:")
display(df["true_label"].value_counts().reindex(VALID_LABELS).fillna(0).astype(int))
print("\nTurn buckets:")
display(df["turn_bucket"].value_counts().sort_index())


# %%
# =========================
# METRIC HELPERS
# =========================

def fixed_five_macro_f1(data):
    """Fixed five-label Macro F1, as requested in advisor Part 3."""
    _, _, macro_f1, _ = precision_recall_fscore_support(
        data["true_label"],
        data["predicted_label"],
        labels=VALID_LABELS,
        average="macro",
        zero_division=0,
    )
    return float(macro_f1)


def accuracy(data):
    return float(accuracy_score(data["true_label"], data["predicted_label"]))


def class_counts(data, label):
    y_true = data["true_label"].to_numpy()
    y_pred = data["predicted_label"].to_numpy()
    tp = int(((y_true == label) & (y_pred == label)).sum())
    fp = int(((y_true != label) & (y_pred == label)).sum())
    fn = int(((y_true == label) & (y_pred != label)).sum())
    support = int((y_true == label).sum())
    return tp, fp, fn, support


def class_f1_or_na(data, label):
    """
    Class-wise display rule:
    - If TP=FP=FN=0, the class has no evidence in that bucket and F1 is NA.
    - If support is zero but FP>0, F1 is 0 because the model predicted the class incorrectly.
    - Otherwise use sklearn F1 with zero_division=0.
    """
    tp, fp, fn, support = class_counts(data, label)
    if tp == 0 and fp == 0 and fn == 0:
        return np.nan
    _, _, f1, _ = precision_recall_fscore_support(
        data["true_label"],
        data["predicted_label"],
        labels=[label],
        average=None,
        zero_division=0,
    )
    return float(f1[0])


def metric_value(data, metric_name, label=None):
    if len(data) == 0:
        return np.nan
    if metric_name == "accuracy":
        return accuracy(data)
    if metric_name == "macro_f1_fixed5":
        return fixed_five_macro_f1(data)
    if metric_name == "class_f1":
        return class_f1_or_na(data, label)
    raise ValueError(metric_name)


def conversation_bootstrap_ci(data, metric_name, label=None, n_boot=N_BOOT, seed=BOOT_SEED):
    """
    Whole-conversation bootstrap:
    draw the original number of conversation IDs with replacement and include
    every row for each sampled conversation. Repeated conversation draws are
    retained as repeated rows.
    """
    grouped_indices = {
        cid: idx.to_numpy()
        for cid, idx in data.groupby("conversation_id").groups.items()
    }
    cids = np.array(list(grouped_indices.keys()))
    if len(cids) == 0:
        return np.nan, np.nan, 100.0

    rng = np.random.default_rng(seed)
    vals = []
    missing_intent_draws = 0

    for _ in range(n_boot):
        sampled_cids = rng.choice(cids, size=len(cids), replace=True)
        sampled_idx = np.concatenate([grouped_indices[cid] for cid in sampled_cids])
        sample = data.loc[sampled_idx]

        if label is not None and int((sample["true_label"] == label).sum()) == 0:
            missing_intent_draws += 1

        vals.append(metric_value(sample, metric_name, label=label))

    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, 100.0 * missing_intent_draws / n_boot

    low, high = np.percentile(vals, [2.5, 97.5])
    missing_pct = 100.0 * missing_intent_draws / n_boot if label is not None else np.nan
    return float(low), float(high), float(missing_pct)


# %%
# =========================
# TURN SUPPORT AND COUNTS
# =========================

support_rows = []
count_rows = []

for tb in TURN_ORDER:
    g = df[df["turn_bucket"] == tb]
    row = {
        "Turn": tb,
        "Conversations_n": g["conversation_id"].nunique(),
        "Turns_n": len(g),
    }
    for label in VALID_LABELS:
        tp, fp, fn, support = class_counts(g, label)
        row[f"{label}_support"] = support
        count_rows.append({
            "Turn": tb,
            "Intent": label,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "True_support": support,
        })
    support_rows.append(row)

turn_support_df = pd.DataFrame(support_rows)
turn_counts_df = pd.DataFrame(count_rows)

display(turn_support_df)
display(turn_counts_df)

turn_support_path = os.path.join(OUTPUT_DIR, "turn_support_counts_by_bucket.csv")
turn_counts_path = os.path.join(OUTPUT_DIR, "turn_intent_tp_fp_fn_counts.csv")
turn_support_df.to_csv(turn_support_path, index=False)
turn_counts_df.to_csv(turn_counts_path, index=False)

print("Saved:", turn_support_path)
print("Saved:", turn_counts_path)


# %%
# =========================
# AGGREGATE TURN METRICS WITH 5000 CONVERSATION BOOTSTRAP CIs
# =========================

aggregate_rows = []

for tb in TURN_ORDER:
    g = df[df["turn_bucket"] == tb]
    acc = metric_value(g, "accuracy")
    mf1 = metric_value(g, "macro_f1_fixed5")
    acc_low, acc_high, _ = conversation_bootstrap_ci(g, "accuracy")
    f1_low, f1_high, _ = conversation_bootstrap_ci(g, "macro_f1_fixed5")

    aggregate_rows.extend([
        {
            "Turn": tb,
            "Metric": "Accuracy",
            "Conversations_n": g["conversation_id"].nunique(),
            "Turns_n": len(g),
            "Intent_support_n": np.nan,
            "Score": acc,
            "CI_95_low": acc_low,
            "CI_95_high": acc_high,
            "Draws_without_intent_percent": np.nan,
            "Bootstrap_unit": "conversation_id",
            "Bootstrap_replicates": N_BOOT,
            "Bootstrap_seed": BOOT_SEED,
        },
        {
            "Turn": tb,
            "Metric": "Macro F1 fixed five-label",
            "Conversations_n": g["conversation_id"].nunique(),
            "Turns_n": len(g),
            "Intent_support_n": np.nan,
            "Score": mf1,
            "CI_95_low": f1_low,
            "CI_95_high": f1_high,
            "Draws_without_intent_percent": np.nan,
            "Bootstrap_unit": "conversation_id",
            "Bootstrap_replicates": N_BOOT,
            "Bootstrap_seed": BOOT_SEED,
        },
    ])

aggregate_df = pd.DataFrame(aggregate_rows)
display(aggregate_df)

aggregate_path = os.path.join(OUTPUT_DIR, "turn_aggregate_accuracy_macroF1_fixed5_bootstrap5000.csv")
aggregate_df.to_csv(aggregate_path, index=False)
print("Saved:", aggregate_path)


# %%
# =========================
# CLASS-WISE TURN F1 WITH NA DISPLAY RULE AND BOOTSTRAP CIs
# =========================

class_rows = []

for tb in TURN_ORDER:
    g = df[df["turn_bucket"] == tb]
    for label in VALID_LABELS:
        tp, fp, fn, support = class_counts(g, label)
        f1 = class_f1_or_na(g, label)

        if pd.isna(f1):
            low, high, missing_pct = np.nan, np.nan, 100.0
        else:
            low, high, missing_pct = conversation_bootstrap_ci(g, "class_f1", label=label)

        class_rows.append({
            "Turn": tb,
            "Metric": f"{label} F1",
            "Intent": label,
            "Conversations_n": g["conversation_id"].nunique(),
            "Turns_n": len(g),
            "Intent_support_n": support,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Score": f1,
            "CI_95_low": low,
            "CI_95_high": high,
            "Draws_without_intent_percent": missing_pct,
            "Zero_support_display_NA": bool(pd.isna(f1)),
            "Bootstrap_unit": "conversation_id",
            "Bootstrap_replicates": N_BOOT,
            "Bootstrap_seed": BOOT_SEED,
        })

class_df = pd.DataFrame(class_rows)
display(class_df)

class_path = os.path.join(OUTPUT_DIR, "turn_classwise_F1_support_CI_zero_support_NA_bootstrap5000.csv")
class_df.to_csv(class_path, index=False)
print("Saved:", class_path)


# %%
# =========================
# COMBINED TURN RESULTS TABLE
# =========================

combined_df = pd.concat([aggregate_df, class_df], ignore_index=True, sort=False)
combined_path = os.path.join(OUTPUT_DIR, "turn_results_aggregate_and_intent_rows_bootstrap5000.csv")
combined_df.to_csv(combined_path, index=False)
display(combined_df)
print("Saved:", combined_path)


# %%
# =========================
# FIGURE 2: ACCURACY + FIXED FIVE-LABEL MACRO F1
# =========================

agg_plot = aggregate_df.pivot(index="Turn", columns="Metric", values="Score").reindex(TURN_ORDER)
ci_low = aggregate_df.pivot(index="Turn", columns="Metric", values="CI_95_low").reindex(TURN_ORDER)
ci_high = aggregate_df.pivot(index="Turn", columns="Metric", values="CI_95_high").reindex(TURN_ORDER)

x = np.arange(len(TURN_ORDER))
fig, ax = plt.subplots(figsize=(8.5, 5.0))

for metric, marker in [("Accuracy", "o"), ("Macro F1 fixed five-label", "s")]:
    y = agg_plot[metric].to_numpy()
    yerr = np.vstack([
        y - ci_low[metric].to_numpy(),
        ci_high[metric].to_numpy() - y,
    ])
    ax.errorbar(x, y, yerr=yerr, marker=marker, linewidth=2, capsize=4, label=metric)
    for i, val in enumerate(y):
        ax.text(i, min(val + 0.045, 1.02), f"{val:.2f}", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(TURN_ORDER)
ax.set_xlabel("User turn bucket")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("Turn-wise Accuracy and Fixed Five-label Macro F1")
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()

fig2_path = os.path.join(OUTPUT_DIR, "Figure2_turnwise_accuracy_macroF1_fixed5_bootstrap5000.png")
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", fig2_path)


# %%
# =========================
# FIGURE 3: CLASS-WISE TURN-WISE F1 WITH NA GAPS
# =========================

fig, ax = plt.subplots(figsize=(9.0, 5.2))

palette = {
    "Chitchat": "#1f77b4",
    "Interpret": "#2ca02c",
    "Reveal": "#ff7f0e",
    "Revise": "#d62728",
    "Inquire": "#9467bd",
}

for label in VALID_LABELS:
    temp = class_df[class_df["Intent"] == label].set_index("Turn").reindex(TURN_ORDER).reset_index()
    ax.plot(
        x,
        temp["Score"],
        marker="o",
        linewidth=2,
        label=label,
        color=palette[label],
    )

ax.set_xticks(x)
ax.set_xticklabels(TURN_ORDER)
ax.set_xlabel("User turn bucket")
ax.set_ylabel("F1 score")
ax.set_ylim(0, 1.05)
ax.set_title("Class-wise Turn-wise F1")
ax.grid(alpha=0.25)
ax.legend(loc="lower left")

# Mark NA display locations near baseline.
na_rows = class_df[class_df["Zero_support_display_NA"]].copy()
turn_to_x = {tb: i for i, tb in enumerate(TURN_ORDER)}
intent_offsets = {
    "Chitchat": -0.18,
    "Interpret": -0.09,
    "Reveal": 0.0,
    "Revise": 0.09,
    "Inquire": 0.18,
}
for _, row in na_rows.iterrows():
    ax.text(
        turn_to_x[row["Turn"]] + intent_offsets.get(row["Intent"], 0.0),
        0.035,
        "NA",
        ha="center",
        va="bottom",
        fontsize=8,
        color=palette[row["Intent"]],
        rotation=90,
    )

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "Figure3_classwise_turnwise_F1_zero_support_NA_bootstrap5000.png")
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", fig3_path)


# %%
# =========================
# OPTIONAL HEATMAP FIGURE 3 VERSION
# =========================

heat = class_df.pivot(index="Intent", columns="Turn", values="Score").reindex(VALID_LABELS)[TURN_ORDER]
support = class_df.pivot(index="Intent", columns="Turn", values="Intent_support_n").reindex(VALID_LABELS)[TURN_ORDER]

annot = heat.copy().astype(object)
for intent in VALID_LABELS:
    for tb in TURN_ORDER:
        val = heat.loc[intent, tb]
        n = support.loc[intent, tb]
        annot.loc[intent, tb] = "" if pd.isna(val) else f"{val:.2f}\n(n={int(n)})"

fig, ax = plt.subplots(figsize=(8.0, 4.8))
sns.heatmap(
    heat,
    annot=annot,
    fmt="",
    cmap="YlGnBu",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "F1 score"},
    ax=ax,
)
ax.set_title("Class-wise Turn-wise F1 with Support")
ax.set_xlabel("User turn bucket")
ax.set_ylabel("Intent")
plt.tight_layout()
heat_path = os.path.join(OUTPUT_DIR, "Figure3_heatmap_classwise_turnwise_F1_zero_support_blank_bootstrap5000.png")
plt.savefig(heat_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", heat_path)


# %%
# =========================
# BOOTSTRAP METHOD METADATA
# =========================

metadata = {
    "input_predictions_csv": str(PREDICTIONS_CSV),
    "valid_labels": VALID_LABELS,
    "turn_order": TURN_ORDER,
    "macro_f1_convention": "fixed five-label Macro F1 with zero_division=0",
    "classwise_zero_support_display_rule": "F1 displayed as NA only when TP=FP=FN=0; false positives with no true support give F1=0",
    "bootstrap_unit": "conversation_id",
    "bootstrap_replicates": N_BOOT,
    "bootstrap_seed": BOOT_SEED,
    "ci_method": "percentile 2.5 and 97.5",
}

metadata_path = os.path.join(OUTPUT_DIR, "turnwise_bootstrap_method_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Saved:", metadata_path)


print("\nOUTPUT FILES:")
for name in sorted(os.listdir(OUTPUT_DIR)):
    print("-", os.path.join(OUTPUT_DIR, name))
