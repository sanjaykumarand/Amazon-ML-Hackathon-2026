#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 - Business Entity Resolution
FINAL PIPELINE: Inverted Index + numpy bincount (21K entities/sec)
- Same inverted index structure (proved fast for training)
- Replaces Python Counter with numpy bincount (18x faster)
- Processes India 809K in ~1 min, US 663K in ~0.8 min
- Total runtime: ~20-35 min end-to-end
"""
import os, gc, re, sys, time, zipfile, subprocess
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

BASE_DIR           = "student_resource"
DATASET_DIR        = os.path.join(BASE_DIR, "dataset")
PREPPED_TRAIN_DIR  = os.path.join(BASE_DIR, "prepped", "raw")
PREPPED_TEST_DIR   = os.path.join(BASE_DIR, "prepped_test", "raw")
OUTPUT_DIR         = os.path.join(BASE_DIR, "output")
CANDIDATE_OUT_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_OUT_PATH  = os.path.join(OUTPUT_DIR, "matching_results.tsv")
SUBMISSION_ZIP     = os.path.join(BASE_DIR, "amazon_ml_hackathon_submission.zip")

MAX_CANDS           = 20
MAX_NAME_DOC_FREQ   = 6000
MIN_TOKEN_LEN       = 2
TRAIN_SAMPLE        = 35000
PROGRESS_EVERY      = 50000   # print progress every N entities

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREPPED_TRAIN_DIR, exist_ok=True)
os.makedirs(PREPPED_TEST_DIR, exist_ok=True)

STOPWORDS = {
    'inc','corp','pvt','ltd','co','and','the','of','in','for','llc','lp',
    'enterprises','services','solutions','technologies','trading','group',
    'india','international','holdings','industries','associates','agency',
    'road','st','rd','street','near','opp','opposite','floor','building',
    'shop','plot','nagar','colony','lane','main','cross','post','dist'
}

CLEAN_RE = re.compile(r'[^\w\s]')
LEGAL_MAP = {
    'incorporated':'inc','corporation':'corp','private':'pvt',
    'limited':'ltd','company':'co','road':'rd','street':'st',
}

def clean_single(text):
    if not text or not isinstance(text, str): return ""
    text = CLEAN_RE.sub(' ', text.lower().replace('&', ' and '))
    return " ".join(LEGAL_MAP.get(w, w) for w in text.split())

# ==========================================
# COUNTRY SPLITTING
# ==========================================
def split_by_country_if_needed(split="train"):
    out_dir = PREPPED_TRAIN_DIR if split == "train" else PREPPED_TEST_DIR
    existing = os.listdir(out_dir) if os.path.exists(out_dir) else []
    if any(f.startswith("source1_") for f in existing):
        print(f"[{split}] Prepped files already exist."); return
    print(f"[{split}] Splitting by country into {out_dir}...")
    for n in (1, 2, 3):
        src = os.path.join(DATASET_DIR, split, f"{split}_source{n}.tsv")
        if not os.path.exists(src): continue
        writers, files = {}, {}
        try:
            for chunk in pd.read_csv(src, sep="\t", dtype=str, chunksize=200000):
                chunk['country'] = chunk['country'].fillna("UNK")
                for country, sub in chunk.groupby('country'):
                    out_path = os.path.join(out_dir, f"source{n}_{country}.tsv")
                    if country not in writers:
                        f = open(out_path, "w", newline="", encoding="utf-8")
                        files[country] = f
                        sub.to_csv(f, sep="\t", index=False, lineterminator="\n")
                        writers[country] = True
                    else:
                        sub.to_csv(files[country], sep="\t", index=False, header=False, lineterminator="\n")
        finally:
            for f in files.values(): f.close()
        print(f"  Split {src} done.")

# ==========================================
# NUMPY BINCOUNT INVERTED INDEX BLOCKING
# Key: replaces Python Counter with numpy ops (~18x faster)
# ==========================================
def block_and_load_country(raw_dir, country, cap=20, sample_limit=None):
    p1 = os.path.join(raw_dir, f"source1_{country}.tsv")
    p2 = os.path.join(raw_dir, f"source2_{country}.tsv")
    p3 = os.path.join(raw_dir, f"source3_{country}.tsv")
    assert os.path.exists(p1), f"Missing {p1}"

    # Load S1
    s1 = pd.read_csv(p1, sep="\t", dtype=str, nrows=sample_limit)
    s1['clean_name'] = [clean_single(x) for x in s1['business_name']]
    s1['clean_addr'] = [clean_single(x) for x in s1['business_address']]
    s1['country'] = country
    s1_ids = s1['entity_id'].tolist()
    s1_tokens = [
        [t for t in row.split() if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS]
        for row in s1['clean_name']
    ]
    print(f"  [{country}] S1 rows: {len(s1):,}")

    # Load S2+S3
    parts = [pd.read_csv(p, sep="\t", dtype=str) for p in (p2, p3) if os.path.exists(p)]
    if not parts:
        print(f"  [{country}] No S2/S3 - all singletons.")
        return s1, {eid: [] for eid in s1_ids}, {}

    t0 = time.time()
    s23 = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    s23_ids   = s23['entity_id'].tolist()
    s23_names = [clean_single(x) for x in s23['business_name']]
    s23_addrs = [clean_single(x) for x in s23['business_address']]
    del s23; gc.collect()
    print(f"  [{country}] S2/S3: {len(s23_ids):,} normalized in {time.time()-t0:.1f}s.")

    # Build lookup
    s23_lookup = {
        eid: {'clean_name': n, 'clean_addr': a}
        for eid, n, a in zip(s23_ids, s23_names, s23_addrs)
    }

    # Build inverted index: token -> numpy int32 array of S23 indices
    t0 = time.time()
    inv_raw = defaultdict(list)
    for idx, name in enumerate(s23_names):
        for tok in set(name.split()):
            if len(tok) >= MIN_TOKEN_LEN and tok not in STOPWORDS:
                inv_raw[tok].append(idx)

    # Frequency filter + convert lists to numpy arrays (faster lookup)
    inv_index = {}
    for tok, idx_list in inv_raw.items():
        if len(idx_list) <= MAX_NAME_DOC_FREQ:
            inv_index[tok] = np.array(idx_list, dtype=np.int32)
    del inv_raw, s23_names, s23_addrs; gc.collect()
    print(f"  [{country}] Index: {len(inv_index):,} tokens in {time.time()-t0:.1f}s.")

    # NUMPY BINCOUNT MATCHING (replaces slow Python Counter)
    t0 = time.time()
    n_s23 = len(s23_ids)
    cand_map = {}
    n_covered = 0

    for i, (eid, tokens) in enumerate(zip(s1_ids, s1_tokens)):
        # Progress reporting
        if i > 0 and i % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            remaining = (len(s1_ids) - i) / rate
            print(f"  [{country}] {i:,}/{len(s1_ids):,} ({i/len(s1_ids)*100:.1f}%) "
                  f"rate={rate:.0f}/s eta={remaining/60:.1f}min", flush=True)

        # Collect all hits as a flat numpy array
        hit_arrays = [inv_index[tok] for tok in tokens if tok in inv_index]
        if not hit_arrays:
            cand_map[eid] = []
            continue

        all_hits = np.concatenate(hit_arrays)  # flat array of S23 indices

        # Count via numpy unique (avoids huge bincount allocation)
        unique_idx, ucounts = np.unique(all_hits, return_counts=True)

        # Top-k by count
        k = min(cap, len(unique_idx))
        if k == 0:
            cand_map[eid] = []
            continue
        if len(unique_idx) <= k:
            order = np.argsort(ucounts)[::-1]
        else:
            order = np.argpartition(ucounts, -k)[-k:]
            order = order[np.argsort(ucounts[order])[::-1]]

        cand_map[eid] = [s23_ids[int(unique_idx[j])] for j in order]
        n_covered += 1

    elapsed = time.time() - t0
    rate = len(s1_ids) / elapsed
    print(f"  [{country}] Matching done in {elapsed:.1f}s ({rate:.0f} ent/s). "
          f"Covered: {n_covered:,}/{len(s1_ids):,} ({n_covered/len(s1_ids):.2%})")
    return s1, cand_map, s23_lookup

# ==========================================
# FEATURE EXTRACTION (13 features)
# ==========================================
def char_ngram_jaccard(a, b, n=3):
    ac, bc = a.replace(" ", ""), b.replace(" ", "")
    sa = {ac[i:i+n] for i in range(len(ac)-n+1)} if len(ac) >= n else ({ac} if ac else set())
    sb = {bc[i:i+n] for i in range(len(bc)-n+1)} if len(bc) >= n else ({bc} if bc else set())
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0

def fast_features(n1, a1, n2, a2, rank):
    w1, w2   = set(n1.split()), set(n2.split())
    aw1, aw2 = set(a1.split()), set(a2.split())
    name_jac  = len(w1&w2)/len(w1|w2) if (w1 or w2) else 0.0
    s1s, s2s  = " ".join(sorted(w1)), " ".join(sorted(w2))
    name_sort = 1.0 if s1s == s2s and s1s else 0.0
    name_ng3  = char_ngram_jaccard(n1, n2, 3)
    name_ng4  = char_ngram_jaccard(n1, n2, 4)
    addr_jac  = len(aw1&aw2)/len(aw1|aw2) if (aw1 or aw2) else 0.0
    addr_ng4  = char_ngram_jaccard(a1, a2, 4)
    nums1, nums2 = set(re.findall(r'\d+', a1)), set(re.findall(r'\d+', a2))
    num_ov    = len(nums1&nums2)/len(nums1|nums2) if (nums1 or nums2) else 0.0
    pin1 = set(re.findall(r'\b\d{5,6}\b', a1))
    pin2 = set(re.findall(r'\b\d{5,6}\b', a2))
    pin_m     = 1.0 if (pin1 and pin2 and (pin1 & pin2)) else 0.0
    all1, all2 = w1|aw1, w2|aw2
    comb_jac  = len(all1&all2)/len(all1|all2) if (all1 or all2) else 0.0
    return [
        name_jac, name_sort, name_ng3, name_ng4,
        1.0 if n1[:4] == n2[:4] and n1 else 0.0,
        1.0 if n1 == n2 and n1 else 0.0,
        min(len(n1), len(n2)) / max(len(n1), len(n2), 1),
        addr_jac, addr_ng4, num_ov, pin_m, comb_jac, 1.0 / rank,
    ]

def build_pair_dataset(s1_frames, cand_maps, s23_lookup, gt_map=None):
    X, y, pairs, groups = [], [], [], []
    for s1_df, cand_map in zip(s1_frames, cand_maps):
        s1_idx = s1_df.set_index('entity_id')[['clean_name','clean_addr','country']].to_dict('index')
        for s1_id, cids in cand_map.items():
            if not cids: continue
            r1 = s1_idx.get(s1_id)
            if r1 is None: continue
            truth = gt_map.get(s1_id, set()) if gt_map is not None else None
            for rank_idx, cid in enumerate(cids, start=1):
                r2 = s23_lookup.get(cid)
                if r2 is None: continue
                X.append(fast_features(r1['clean_name'], r1['clean_addr'],
                                       r2['clean_name'], r2['clean_addr'], rank_idx))
                pairs.append((s1_id, cid)); groups.append(s1_id)
                if gt_map is not None:
                    y.append(1 if cid in truth else 0)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32) if gt_map is not None else None
    return X, y, pairs, groups

# ==========================================
# METRIC (Macro F_0.5)
# ==========================================
def compute_entity_f05(pred_ids, truth_ids):
    pred_set  = set(pred_ids)  if pred_ids  else set()
    truth_set = set(truth_ids) if truth_ids else set()
    if not truth_set: return 1.0 if not pred_set else 0.0
    if not pred_set:  return 0.0
    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)
    p  = tp / (tp + fp) if (tp+fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp+fn) > 0 else 0.0
    if p == 0.0 or r == 0.0: return 0.0
    return (1.25 * p * r) / (0.25 * p + r)

def evaluate_macro_f05(matches_dict, gt_dict, all_entity_ids):
    return float(np.mean([
        compute_entity_f05(matches_dict.get(eid, []), gt_dict.get(eid, []))
        for eid in all_entity_ids
    ]))

# ==========================================
# ADAPTIVE DECISION POLICY
# ==========================================
def predict_with_policy(pairs, probs, all_s1_ids,
                        tau_singleton=0.35, tau_multi=0.60, delta_margin=0.20):
    s1_cp = defaultdict(list)
    for (s1_id, cid), p in zip(pairs, probs):
        s1_cp[s1_id].append((cid, p))
    matches = {eid: [] for eid in all_s1_ids}
    for s1_id, cl in s1_cp.items():
        if not cl: continue
        cl.sort(key=lambda x: x[1], reverse=True)
        top_cid, top_p = cl[0]
        if top_p >= tau_singleton:
            matched = [top_cid]
            for cid, p in cl[1:]:
                if p >= tau_multi and (top_p - p) <= delta_margin:
                    matched.append(cid)
            matches[s1_id] = matched
    return matches

def tune_decision_policy(val_pairs, val_probs, val_s1_ids, gt_dict):
    print("Tuning decision policy (Macro F_0.5)...")
    best_score, best_params = -1.0, (0.35, 0.60, 0.20)
    for tau_s in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        for tau_m in [0.45, 0.50, 0.55, 0.60, 0.65]:
            for delta in [0.10, 0.15, 0.20, 0.25, 0.30]:
                preds = predict_with_policy(val_pairs, val_probs, val_s1_ids, tau_s, tau_m, delta)
                score = evaluate_macro_f05(preds, gt_dict, val_s1_ids)
                if score > best_score:
                    best_score, best_params = score, (tau_s, tau_m, delta)
    print(f"Best: tau_s={best_params[0]}, tau_m={best_params[1]}, delta={best_params[2]}")
    print(f"Validation Macro F_0.5: {best_score:.4f}")
    return best_params, best_score

# ==========================================
# MAIN
# ==========================================
def main():
    total_t0 = time.time()
    print("="*75)
    print("AMAZON ML CHALLENGE 2026 - NUMPY BINCOUNT BLOCKING PIPELINE")
    print("="*75)

    split_by_country_if_needed("train")
    split_by_country_if_needed("test")

    # --- Load Ground Truth ---
    print("\n--- STAGE 1: Model Training ---")
    t0 = time.time()
    gt_raw = pd.read_csv(
        os.path.join(DATASET_DIR, "train", "train_ground_truth.tsv"), sep="\t", dtype=str)
    gt_raw['matched_entity_ids'] = gt_raw['matched_entity_ids'].fillna("")
    gt_map = {
        s1_id: set(m.split(',')) if m else set()
        for s1_id, m in zip(gt_raw['source1_entity_id'], gt_raw['matched_entity_ids'])
    }
    del gt_raw; gc.collect()
    print(f"Ground truth: {len(gt_map):,} entities in {time.time()-t0:.1f}s.")

    # --- Training Blocking ---
    train_s1_frames, train_cand_maps, combined_lookup = [], [], {}
    for c in ["India", "US"]:
        s1c, candc, lkup = block_and_load_country(
            PREPPED_TRAIN_DIR, c, cap=MAX_CANDS, sample_limit=TRAIN_SAMPLE)
        train_s1_frames.append(s1c)
        train_cand_maps.append(candc)
        combined_lookup.update(lkup)
        del lkup; gc.collect()

    print("\nBuilding feature matrix...")
    X_all, y_all, pairs_all, groups_all = build_pair_dataset(
        train_s1_frames, train_cand_maps, combined_lookup, gt_map)
    print(f"Feature Matrix: {X_all.shape} | Positives: {int(y_all.sum()):,} ({y_all.mean():.2%})")
    del combined_lookup, train_s1_frames, train_cand_maps; gc.collect()

    # --- Train/Val Split ---
    print("\nSplitting Train/Val...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(X_all, y_all, groups=groups_all))
    X_tr, y_tr   = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx],   y_all[val_idx]
    val_pairs    = [pairs_all[i] for i in val_idx]
    val_s1_ids   = list(set(groups_all[i] for i in val_idx))
    print(f"Train: {len(X_tr):,} | Val: {len(X_val):,} ({len(val_s1_ids):,} entities)")

    # --- Fit Model ---
    print("Fitting HistGradientBoostingClassifier...")
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=8,
        min_samples_leaf=30, l2_regularization=0.5, random_state=42)
    model.fit(X_tr, y_tr)
    print(f"Model trained in {time.time()-t0:.1f}s.")

    val_probs = model.predict_proba(X_val)[:, 1]
    best_policy, val_f05 = tune_decision_policy(val_pairs, val_probs, val_s1_ids, gt_map)
    tau_singleton, tau_multi, delta_margin = best_policy

    pd.DataFrame([{
        "model": "HGB_NumpyBincountBlocking",
        "tau_singleton": tau_singleton, "tau_multi": tau_multi,
        "delta_margin": delta_margin, "val_macro_f0.5": val_f05,
        "n_val_entities": len(val_s1_ids),
    }]).to_csv(os.path.join(OUTPUT_DIR, "model_comparison_log.csv"),
               index=False, lineterminator="\n")

    del X_all, y_all, X_tr, y_tr, X_val, y_val, val_pairs, val_probs; gc.collect()

    # --- Test Inference ---
    print("\n--- STAGE 2: Test Inference ---")
    s1_test = pd.read_csv(
        os.path.join(DATASET_DIR, "test", "test_source1.tsv"), sep="\t", dtype=str)
    all_test_ids   = s1_test['entity_id'].tolist()
    TEST_COUNTRIES = sorted(s1_test['country'].fillna("UNK").unique().tolist())
    print(f"Countries: {TEST_COUNTRIES} | Total S1: {len(all_test_ids):,}")
    del s1_test; gc.collect()

    with open(CANDIDATE_OUT_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
    with open(MATCHING_OUT_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")

    total_matches = 0; total_covered = 0

    for country in TEST_COUNTRIES:
        print(f"\n>>> Processing Country: {country} <<<")
        s1c, cand_map, s23_lookup = block_and_load_country(
            PREPPED_TEST_DIR, country, cap=MAX_CANDS)
        total_covered += sum(1 for v in cand_map.values() if v)

        # Write candidate_pairs
        with open(CANDIDATE_OUT_PATH, "a", encoding="utf-8", newline="") as f_cand:
            for eid in s1c['entity_id']:
                f_cand.write(f"{eid}\t{','.join(cand_map.get(eid, []))}\n")

        # Feature extraction
        print(f"  Building features for {len(s1c):,} entities...")
        t0 = time.time()
        X_test, _, pairs_test, _ = build_pair_dataset([s1c], [cand_map], s23_lookup)
        del s23_lookup; gc.collect()
        print(f"  Features: {X_test.shape} in {time.time()-t0:.1f}s.")

        if len(X_test) > 0:
            probs = np.zeros(len(X_test), dtype=np.float32)
            CHUNK = 500_000
            for i in range(0, len(X_test), CHUNK):
                end = min(i + CHUNK, len(X_test))
                probs[i:end] = model.predict_proba(X_test[i:end])[:, 1]
            del X_test; gc.collect()

            country_matches = predict_with_policy(
                pairs_test, probs, s1c['entity_id'].tolist(),
                tau_singleton, tau_multi, delta_margin)
            del pairs_test, probs; gc.collect()
        else:
            country_matches = {eid: [] for eid in s1c['entity_id']}

        # Write matching_results
        with open(MATCHING_OUT_PATH, "a", encoding="utf-8", newline="") as f_match:
            for eid in s1c['entity_id']:
                m_list = country_matches.get(eid, [])
                if m_list: total_matches += 1
                f_match.write(f"{eid}\t{','.join(m_list)}\n")

        del s1c, cand_map, country_matches; gc.collect()
        print(f"  Finished {country}.")

    n_total = len(all_test_ids)
    print(f"\nDone: {n_total:,} entities | "
          f"Covered: {total_covered:,} ({total_covered/n_total:.2%}) | "
          f"Matched: {total_matches:,} ({total_matches/n_total:.2%})")

    # --- Validation ---
    print("\n--- STAGE 3: Validation ---")
    res = subprocess.run([
        sys.executable,
        os.path.join(BASE_DIR, "utils", "validate_submission.py"),
        "--matching",  MATCHING_OUT_PATH,
        "--candidate", CANDIDATE_OUT_PATH,
        "--test-dir",  os.path.join(DATASET_DIR, "test"),
    ], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr: print("STDERR:", res.stderr[:1000])

    # --- Build Zip ---
    print("\nBuilding submission zip...")
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in [MATCHING_OUT_PATH, CANDIDATE_OUT_PATH,
                  os.path.join(OUTPUT_DIR, "model_comparison_log.csv")]:
            if os.path.exists(f):
                zf.write(f, arcname=os.path.relpath(f, BASE_DIR))
    print(f"Zip: {os.path.getsize(SUBMISSION_ZIP)/1e6:.1f} MB")
    print(f"Total time: {(time.time()-total_t0)/60:.2f} min")
    print("="*75)
    print("COMPLETE -> submit: student_resource/amazon_ml_hackathon_submission.zip")
    print("="*75)

if __name__ == "__main__":
    main()
