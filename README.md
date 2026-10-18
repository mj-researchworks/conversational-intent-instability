# Reproducibility Package

This package accompanies the manuscript *Understanding Conversational Intent Instability in Product-Search Dialogue: A Diagnostic Study of Semantic Expansion and Context*. It contains the data snapshots, conversation-level fold assignments, generated training rows, training notebooks, held-out predictions, human-annotation materials, and post-hoc analysis scripts used for the reported results.

## Scope

The complete five-class task is the primary evaluation. Four-class experiments are separate diagnostic analyses and should not be compared as a continuous performance progression from the five-class task.

Most training notebooks were executed in Google Colab and retain their original Colab paths. Upload or mount the corresponding package inputs when rerunning those notebooks. The standalone post-hoc scripts listed below use package-relative paths and can be run locally from the package root.

## Data and experiment inputs

- Original English PSCon data: `DATA/ORIGINAL DATA/conversation_en.json`
- Original Chinese PSCon data used for translation-quality evaluation: `Translation_Quality/conversation_cn.json`
- Authoritative translated Chinese data: `DATA/TRANSLATED DATA/conversation_cn_translated_en_deepseek_v4flash.json`
- Chinese-to-English translation notebook: `Translation_Quality/translation_chinese_to_english.ipynb`
- Four-class five-fold assignments: `4Class_5cv/4class_5cv_fold_creation/5cv_4class_data/`
- Five-class five-fold assignments: `5Class_5cv/5class_5cv_fold_creation/data_5cv/`
- Fold-specific generated rows are stored in the corresponding `4Class_5cv` and `5Class_5cv` generated-data directories.

The DeepSeek V4-Flash JSON above is the only translated-English file used by the final translation-quality analysis. The COMET-QE output contains 3,056 scored turns and an overall score of 0.810 ± 0.060.

DeepSeek-V4-Flash was used only for Chinese-to-English translation. Qwen2.5-1.5B, Llama-3.2-3B, and Gemma-3-1B were evaluated separately as decoder-based intent classifiers.

## Manuscript table map

Table numbering refers to the current revised manuscript.

| Manuscript results | Package location |
|---|---|
| Table 8, classical machine-learning baselines | `ML BASELINE/ML_baselines.ipynb` |
| Tables 9 and 14, unaugmented encoder baselines | `5Class_single_split_baseline/` and `4Class_single_split_baseline/` |
| Table 10, decoder configurations | `DECODER_ALLEXPTS/` |
| Table 11, complete five-class single split | `5Class_single_split/` |
| Table 12, expanded-only five-class five-fold results | `5Class_5cv/5class_5cv_Training_scripts/` |
| Tables 12a–12c, matched Original versus Expanded five-fold comparison | `5Class_Original_vs_Expanded_BERT/` |
| Table 13, RoBERTa seed sensitivity | `5Class_5cv/5class_5cv_Training_scripts/5class_5cv_Roberta_seed_expt/` |
| Table 15, four-class full single-split configuration | `4Class_single_split/` |
| Table 16, four-class five-fold results | `4Class_5cv/` |
| Tables 17–18, factorial ablation and Shapley attribution | `Ablation/Ablation_4Class_single_split_Shapley/` |
| Table 19, context ablation | `Ablation/Ablation_4Class_single_split_context/` |
| Tables 20–20a and turn-wise figures | `Bootstrap/` |
| Tables 22–23b, three-annotator human audit | `Human_Annotation_Audit/` |
| Table 23c, CV1 partial label-correction sensitivity | `5Class_Original_vs_Expanded_BERT/CV1_Partial_Label_Correction_Sensitivity/` and `5Class_Original_vs_Expanded_BERT/Results_Analysis/` |
| Tables 24–25b, translation-quality evaluation | `Translation_Quality/` |
| Table 26, within-PSCon cross-lingual comparison | English: `4Class_5cv/`; Chinese original: `4Class_5cv/Chinese_Original_5fold/4class_chinese_original_5cv_bertbase.ipynb` |

## Post-hoc analyses that do not retrain a classifier

The following commands reproduce calculations from saved annotations or held-out predictions. They do not train BERT or another intent classifier.

### Human annotation majority labels and Fleiss kappa

```bash
python Human_Annotation_Audit/calculate_human_annotation_majority_and_fleiss.py \
  Human_Annotation_Audit \
  reproduced_results/CV1_human_audit_final.xlsx
```

Expected headline results are 100 items, Fleiss’ kappa 0.893418, 89% unanimous agreement, no three-way disagreements, and 84% human-majority agreement with inherited labels.

### CV1 paired whole-conversation bootstrap

```bash
python 5Class_Original_vs_Expanded_BERT/CV1_Partial_Label_Correction_Sensitivity/cv1_correction_macro_f1_bootstrap_2000.py \
  --output-folder reproduced_results/cv1_correction_bootstrap
```

The script automatically locates Original, Expanded, and Corrected CV1 prediction files inside the package. It uses 2,000 whole-conversation bootstrap resamples and seed 42.

### Exact three-factor Shapley calculation

```bash
python Ablation/Ablation_4Class_single_split_Shapley/calculate_shapley_values.py
```

This calculation uses the eight saved four-class factorial experiment scores and performs no training. The reported rounded values are A = −0.011, B = +0.015, and C = +0.025.

### Turn-wise whole-conversation bootstrap

```bash
TURNWISE_OUTPUT_DIR=reproduced_results/turnwise_bootstrap \
python Bootstrap/analyze_turnwise_fixed5label_conversation_bootstrap_5000.py
```

The default input is `Bootstrap/bert_test_predictions_posthoc.csv`. The analysis uses 5,000 whole-conversation bootstrap resamples, seed 42, and a fixed five-label Macro F1 convention with `zero_division=0`.

Optional environment variables are `TURNWISE_PREDICTIONS_CSV`, `TURNWISE_OUTPUT_DIR`, `TURNWISE_BOOTSTRAP_REPLICATES`, and `TURNWISE_BOOTSTRAP_SEED`.

### Translation-quality calculations

- `Translation_Quality/translation_chinese_to_english.ipynb` generates the retained English translation of the original Chinese PSCon conversations using DeepSeek-V4-Flash.
- `Translation_Quality/01_comet_qe_translation_quality_conversation_cn_translated_en.ipynb` performs COMET-QE inference. It does not train the intent classifier.
- `Translation_Quality/02_llm_judge_translation_quality_v4flash_50samples_seed42.ipynb` contains the 50-item model-judge sampling and scoring procedure.
- The LLM-judge sample uses seed 42, `gpt-4o-mini`, temperature 0, and `max_tokens=300`.

## Random seeds and resampling

| Analysis | Setting |
|---|---|
| Primary experiment seed | 42 |
| Additional RoBERTa seeds | 123 and 256 |
| Human-audit stratified sampling | 42 |
| Translation judge 50-item sampling | 42 |
| CV1 correction bootstrap | 2,000 whole-conversation resamples, seed 42 |
| Turn-wise bootstrap | 5,000 whole-conversation resamples, seed 42 |

## Software environments

`requirements.txt` is the single reviewer-facing dependency file for the package. It includes the common analysis and COMET-QE dependencies and the libraries required by the supplied training notebooks. CUDA-specific PyTorch builds should be installed using the index appropriate for the available GPU runtime.

The shared file intentionally pins Transformers 4.46.3 because this version was used by the successful final COMET-QE run and satisfies the `<5` constraints retained in the supplied experiment notebooks. Historical Colab notebooks retain their original installation cells as the record of their run-specific environments. The exact historical scikit-learn patch version was not preserved; this limitation is stated in `requirements.txt` rather than replaced with an invented version.

## Known reproduction limitations

- Exact API snapshot identifiers used during the original generation runs were not retained.
- Some run-level operational records, including duplicate-removal logs, were not retained.
- The exact historical scikit-learn patch version was not retained.
- These omissions limit exact recreation of the original API-based data-construction runs, but the retained prompts, notebooks, generated rows, fold assignments, provenance fields, predictions, and saved outputs support verification of the reported analyses.

## Package verification

`PACKAGE_TEST_LOG.txt` records verification of the human-audit calculation, CV1 paired bootstrap, exact Shapley calculation, and turn-wise whole-conversation bootstrap. These checks regenerate the reported post-hoc results from the packaged annotations and prediction files without rerunning model training.

## Reproduction notes

- Fold identifiers represent source conversations. Train, validation, and test conversation identifiers must remain disjoint.
- Generated rows retain their source `conversation_id`, turn index, inherited intent, augmentation type, source user text, and source system text.
- The human audit covers 100 CV1 generated examples from Chitchat, Revise, and Inquire only; it is not an estimate over all generated rows.
- The corrected CV1 experiment changes only the 16 audited labels that differ from the human majority. It does not modify validation or test labels.
- A post-hoc token-length check confirmed that none of the 652 held-out inputs exceeded the 256-token limit; therefore, no test input was truncated.
- Saved prediction CSVs permit verification of headline metrics without rerunning model training.
- COMET-QE and model-judge scores are translation diagnostics and do not replace bilingual human validation.

