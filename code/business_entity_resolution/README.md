# Business Entity Resolution — Amazon ML Challenge 2026

## Overview

End-to-end pipeline resolving noisy business records from Source 2 and Source 3 against a clean Source 1 reference, using only `business_name`, `business_address`, and `country`. Optimised for **macro-averaged F_0.5** (precision weighted 2x over recall).

---

## How to Reproduce Results

### 1. Prerequisites

Install dependencies (Python 3.9+):

```bash
pip install -r requirements.txt
```

### 2. Data Setup

Place the competition dataset under `student_resource/dataset/`:

```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
└── utils/
    └── validate_submission.py
```

### 3. Run the Pipeline

From the **project root** (folder containing `student_resource/`):

```bash
python src/run_pipeline.py
```

The script is fully self-contained — it splits data by country, trains the model, runs inference on all three test countries (France, India, US), validates output, and builds the submission zip.

**Expected runtime:** ~60-70 minutes on a 16GB RAM / 14-core CPU laptop.

### 4. Outputs

| File | Location | Description |
|------|----------|-------------|
| `matching_results.tsv` | `student_resource/output/` | Final scored predictions (1 row per S1 entity) |
| `candidate_pairs.tsv` | `student_resource/output/` | Blocking candidate set (top-20 per S1 entity) |
| `amazon_ml_hackathon_submission.zip` | `student_resource/` | Final submission zip |

---

## Directory Structure

```
code/business_entity_resolution/
├── src/
│   └── run_pipeline.py          # Single-file end-to-end pipeline
├── README.md                    # This file
└── requirements.txt             # Pinned dependencies
```

---

## Pipeline Architecture

```
Raw TSVs
   |
   v
[1] Country Splitting
    Split source1/2/3 TSVs by country into prepped/raw/
   |
   v
[2] Inverted Index Blocking  (numpy bincount, ~10K ent/s)
    Build token -> S23 index, match each S1 entity -> top-20 candidates
   |
   v
[3] Feature Extraction  (13 similarity features per pair)
    name_jaccard, name_sort_match, name_ngram3/4, prefix4_match,
    exact_match, length_ratio, addr_jaccard, addr_ngram4,
    numeric_overlap, pin_match, combined_jaccard, candidate_rank
   |
   v
[4] HistGradientBoostingClassifier  (sklearn, 300 iterations)
    Trained on 35K-sample India + US (stratified by S1 group)
   |
   v
[5] Adaptive Decision Policy  (tuned on validation Macro F_0.5)
    top-1 + margin-based multi-match
    tau_singleton=0.4, tau_multi=0.45, delta_margin=0.3
   |
   v
matching_results.tsv  +  candidate_pairs.tsv
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Blocking method | Inverted token index + numpy bincount | 10K+ ent/s; RAM-safe on 16GB |
| Candidate cap | Top-20 per S1 entity | Balances recall vs feature-extraction cost |
| Stopwords filtered | inc, ltd, pvt, road, st, ... | Reduces common-token noise |
| Doc-frequency cap | 6,000 | Prevents hub tokens from flooding candidates |
| Model | HistGradientBoostingClassifier | Handles missing values natively, fast, robust |
| Threshold policy | Adaptive (tau_s / tau_m / delta) | Directly optimises Macro F_0.5, not pairwise |
| Training sample | 35K per country (India + US) | Full data = OOM; sample is representative |

---

## Validation Results

```
matching_results.tsv : 1,732,544 rows
  - Non-empty (matched) : 701,402  (40.48%)
  - Empty (singletons)  : 1,031,142 (59.52%)
Blocking coverage    : 72.16% of S1 entities have >=1 candidate
Validation Macro F_0.5 (held-out): 0.5306
Submission validator : PASS
```

---

## License

All custom code in this submission is released under the MIT License.
The underlying model (HistGradientBoostingClassifier from scikit-learn) is BSD-3-Clause licensed, well within the MIT/Apache 2.0 constraint (<=8B parameters).
