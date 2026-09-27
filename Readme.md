# ML Challenge 2026: Business Entity Resolution Solution

<h3>Unstop Team ID:<b> UJ1K56I7 </b>
<br>Unstop Team Link: https://unstop.com/competitions/1743604/register?invitedId=UJ1K56I7
<br><br></h3>

**Team Name:**<b><h2> AGS</h2></b>
<h2>Teammates:<center>
<br><b>Akshaya Sri B - VM17000 - 113124UG07008 - <a href="https://www.linkedin.com/in/akshaya-sri/">Linkedin🔗</a>
<br>Gokul Rathinam R - VM16908 - 113124UG07036 - <a href="https://www.linkedin.com/in/gokulrathinamr/">Linkedin🔗</a>
<br>Sanjay Kumaran D - VM17014 - 113124UG07100 - <a href="https://www.linkedin.com/in/sanjaykumarand/">Linkedin🔗</a>
</b></center></h2><br>
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We built a high-precision entity resolution pipeline using an inverted-index blocking stage followed by a HistGradientBoosting classifier with 13 hand-crafted similarity features, optimised end-to-end for macro-averaged F_0.5. The key innovation is replacing the conventional Python Counter-based blocking with numpy bincount operations, achieving ~10,000 entities/sec throughput, making full-scale inference on 1.7M entities feasible on a 16GB laptop in under 70 minutes.

---

## 2. Methodology

### 2.1 Problem Analysis

Key insights from EDA:
- ~5.6% of Source 1 entities are true singletons (no match in S2/S3); predicting empty for these is critical for precision.
- Business names vary significantly: abbreviations (Pvt/Private, Ltd/Limited), spelling errors, character corruption (OCR noise), and transliteration differences (particularly for Indian names in Tamil/Hindi script).
- Address fields are highly inconsistent: missing values, varying granularity (plot number, floor, building, PIN code), and regional formatting differences.
- France is an unseen country at training time, requiring the model to generalise from India + US patterns only.
- Numeric tokens (PIN codes, plot numbers) are highly discriminative when present and should be treated separately from name tokens.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage pipeline)
**Core Innovation:** numpy bincount inverted-index blocking (18x faster than Python Counter) enabling full-scale inference within RAM constraints; combined with an adaptive entity-level decision policy that directly optimises Macro F_0.5 rather than pairwise metrics.

---

## 3. Candidate Generation (Blocking)

**Method:** Token-level inverted index with numpy bincount scoring

**Process:**
1. Normalise business names: lowercase, remove punctuation, expand legal abbreviations (Pvt->pvt, Limited->ltd, etc.)
2. Build inverted index: token -> list of S23 entity indices (as numpy int32 arrays)
3. Filter stopwords (inc, ltd, road, st, nagar, colony, ...) and high-frequency tokens (doc_freq > 6,000)
4. For each S1 entity: collect all S23 hits via numpy.concatenate, count with numpy.unique + return_counts, select top-20 by count

**Blocking keys used:** Word-level unigrams from normalised business_name (stopword-filtered, frequency-capped)

**Candidate pairs generated:**
- France: 4,105,266 pairs (259,452 S1 entities, 85.94% covered)
- India: 9,520,999 pairs (809,986 S1 entities, 59.76% covered)
- US: 10,737,636 pairs (663,106 S1 entities, 81.91% covered)
- **Total: ~24.4M candidate pairs across 1,732,544 S1 entities**

**Blocking throughput:** ~10,000-20,000 entities/sec (numpy bincount vs ~1,200/sec with Python Counter)

**Ensuring true matches are not lost:**
- Top-20 cap (not top-1) maximises recall at the blocking stage
- Stopword filtering avoids common tokens diluting genuine name matches
- Frequency cap (6,000) prevents hub tokens from filling all 20 slots with irrelevant matches

---

## 4. Matching Model

**Features used (13 total):**

Name features:
- `name_jaccard` — word-level Jaccard similarity of normalised names
- `name_sort_match` — binary: token-sorted names are identical (handles word-order variation)
- `name_ngram3` — character 3-gram Jaccard (robust to spelling errors)
- `name_ngram4` — character 4-gram Jaccard (higher precision than 3-gram)
- `prefix4_match` — binary: first 4 characters match (catches abbreviations)
- `exact_match` — binary: normalised names are identical
- `length_ratio` — min/max name length ratio (filters length-mismatch pairs)

Address features:
- `addr_jaccard` — word-level Jaccard of normalised addresses
- `addr_ngram4` — character 4-gram Jaccard of addresses
- `numeric_overlap` — Jaccard of all numeric tokens in addresses
- `pin_match` — binary: 5-6 digit numeric codes match (PIN code / ZIP code)
- `combined_jaccard` — Jaccard of (name_words UNION addr_words)

Ranking feature:
- `candidate_rank` — 1/rank (top candidate gets highest weight)

**Model type:** HistGradientBoostingClassifier (scikit-learn 1.5.1)
- max_iter=300, learning_rate=0.05, max_depth=8, min_samples_leaf=30, l2_regularization=0.5
- Trained on 70K pairs (35K India + 35K US sampled S1 entities) × up to 20 candidates each
- GroupShuffleSplit (80/20) by S1 entity to avoid data leakage

**Threshold selection method:**
- Grid search over (tau_singleton ∈ [0.25..0.50], tau_multi ∈ [0.45..0.65], delta_margin ∈ [0.10..0.30])
- Evaluated directly on **Macro F_0.5** (not pairwise F1) on held-out validation entities
- Optimal: tau_singleton=0.40, tau_multi=0.45, delta_margin=0.30

**Decision policy:**
- If top candidate probability >= tau_singleton → assign it as match
- Also include additional candidates where p >= tau_multi AND (top_p - p) <= delta_margin
- If no candidate exceeds tau_singleton → predict empty (singleton)

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, validation):** 0.5306
- **Entities matched (test set):** 701,402 / 1,732,544 (40.48%)
- **Blocking coverage:** 1,250,188 / 1,732,544 (72.16%) entities have at least one candidate

**Common false positives (wrong merges):**
- Businesses with identical or very similar names in the same city but different locations (e.g., chain stores with the same name, no disambiguating address tokens)
- Short generic names (e.g., "Tech Solutions") that match many unrelated businesses after stopword filtering

**Common false negatives (missed matches):**
- Entities with only stopwords or very common words in their name — blocked at the frequency-filter stage
- Heavy transliteration differences between sources (English → Devanagari script, etc.) where token overlap is zero
- Entities with zero name overlap but matching addresses — our blocking is name-only so address-only matches are missed

---

## 6. Conclusion

We developed a memory-efficient, high-throughput entity resolution pipeline that processes 1.73M S1 entities across three countries (France unseen at training) in ~70 minutes on commodity hardware. The key contribution is the numpy bincount blocking approach which achieves 18x speedup over conventional Python Counter methods without any external dependencies. The adaptive entity-level decision policy (rather than fixed pairwise threshold) is critical for optimising the Macro F_0.5 metric. Future improvements would include address-augmented blocking (to recover entities with zero name overlap), TF-IDF weighting of blocking tokens, and phonetic encoding for handling transliteration noise.

---

## Appendix

### A. Code Artefacts

Complete runnable pipeline is in `code/business_entity_resolution/`:

```
src/run_pipeline.py     # Single self-contained script (~500 lines)
README.md               # Reproduction instructions
requirements.txt        # Pinned dependencies (numpy, pandas, scikit-learn)
```

**Entry point:** `python src/run_pipeline.py` (run from project root containing `student_resource/`)

**Reproduces:** `student_resource/output/matching_results.tsv` and `candidate_pairs.tsv`

The pipeline auto-detects pre-processed files and skips already-completed steps, so partial re-runs are supported.

### B. Additional Results

| Country | S1 Entities | Covered | Matched | Singletons |
|---------|-------------|---------|---------|------------|
| France  | 259,452     | 85.94%  | ~105K   | ~154K      |
| India   | 809,986     | 59.76%  | ~292K   | ~518K      |
| US      | 663,106     | 81.91%  | ~304K   | ~359K      |
| **Total** | **1,732,544** | **72.16%** | **701,402** | **1,031,142** |

Blocking timing (numpy bincount):
- France matching: 12.5s (20,714 ent/s)
- India matching: 80.4s (10,068 ent/s)
- US matching: 65.5s (10,124 ent/s)

---

**Note:** This pipeline uses only MIT/Apache 2.0 licensed libraries (numpy, pandas, scikit-learn). The model is a HistGradientBoostingClassifier with ~300 trees — well within the 8B parameter constraint.
