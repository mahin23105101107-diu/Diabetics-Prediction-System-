#!/usr/bin/env python3
"""
Train and export the 9-feature Random Forest Classifier with 5-fold OOF Platt
calibration directly from the project's survey dataset (dataset_Update_-_new.csv).

Outputs:
  model-data.js — browser-ready static model payload for client-side predict_proba()
                  on GitHub Pages without requiring a backend server.
"""

import csv
import json
import math
import os
import random
from collections import Counter

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "public",
    "assets",
    "aistudio",
    "dataset_Update_-_new_38dcf422.csv",
)
OUTPUT_JS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model-data.js",
)

# Exact 9 questionnaire features aligned with the CSV column indices
# (0-based indices AFTER removing column 0: Timestamp).
FEATURE_SPECS = [
    {
        "key": "ageGroup",
        "csv_index": 0,
        "csv_column": "1. What is your age group? (আপনার বয়স কত?)",
        "categories": ["Under 18", "18-25", "26-35", "35-40"],
    },
    {
        "key": "exerciseDays",
        "csv_index": 5,
        "csv_column": "6. How many days per week do you exercise? ( সপ্তাহে কত দিন আপনি ব্যায়াম করেন?)",
        "categories": ["Never", "1-2 Days", "3-4 Days", "Daily"],
    },
    {
        "key": "familyHistory",
        "csv_index": 15,
        "csv_column": "16.Does anyone in your family have Diabetes? ( আপনার পরিবারের কারো কি ডায়াবেটিস আছে? )",
        "categories": ["No", "Not Sure", "Yes (Parents)", "Yes (Grandparents)"],
    },
    {
        "key": "highBloodPressure",
        "csv_index": 16,
        "csv_column": "17. Have you ever been diagnosed with high blood pressure? ( আপনার কি কখনো উচ্চ রক্তচাপ ধরা পড়েছে? )",
        "categories": ["Yes", "No"],
    },
    {
        "key": "craveSweets",
        "csv_index": 6,
        "csv_column": "7 .Do you often crave sweets? (আপনার কি প্রায়ই মিষ্টি খেতে ইচ্ছা করে?)",
        "categories": ["Never", "Rarely", "Sometimes", "Often", "Very often", "Always"],
    },
    {
        "key": "frequentUrination",
        "csv_index": 21,
        "csv_column": "22. Do you urinate frequently? (আপনার কি ঘন ঘন প্রস্রাব হয়? )",
        "categories": ["Yes", "No"],
    },
    {
        "key": "excessiveThirst",
        "csv_index": 22,
        "csv_column": "23. Do you feel very thirsty? (আপনার কি বেশি পিপাসা লাগে?)",
        "categories": ["Yes", "No"],
    },
    {
        "key": "frequentHunger",
        "csv_index": 23,
        "csv_column": "24. Do you feel hungry frequently?( আপনার কি ঘন ঘন ক্ষুধা লাগে?)",
        "categories": ["Yes", "No"],
    },
    {
        "key": "weightChange",
        "csv_index": 24,
        "csv_column": "25. Have you lost or gained weight in the last 3 months? (গত ৩ মাসে আপনার কি ওজন কমেছে বা বেড়েছে ?)",
        "categories": ["Yes", "No"],
    },
]
TARGET_INDEX = 25  # Column 26 before dropping Timestamp


def load_deduplicated_dataset(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)
        raw_rows = list(reader)

    seen = set()
    rows = []
    missing_dropped = 0
    duplicates_dropped = 0

    for r in raw_rows:
        vals = [c.strip() for c in r[1:]]  # Drop Timestamp (col 0)
        if any(v == "" for v in vals):
            missing_dropped += 1
            continue
        t = tuple(vals)
        if t in seen:
            duplicates_dropped += 1
            continue
        seen.add(t)
        rows.append(vals)

    return headers, rows, len(raw_rows), missing_dropped, duplicates_dropped


def build_onehot_schema():
    schema = []
    for spec in FEATURE_SPECS:
        for cat in spec["categories"]:
            schema.append((spec["key"], spec["csv_index"], cat))
    return schema


def encode_row(row, schema):
    return [1 if row[csv_idx] == cat else 0 for (_, csv_idx, cat) in schema]


def build_cart_tree(X, y, indices, depth, max_depth, min_samples_leaf, max_features, n_feats, rng, importances):
    n = len(indices)
    pos = sum(y[i] for i in indices)
    p1 = pos / n
    if depth >= max_depth or pos == 0 or pos == n or n < 2 * min_samples_leaf:
        return round(p1, 4)

    gini_parent = 2.0 * p1 * (1.0 - p1)
    feat_subset = rng.sample(range(n_feats), max_features)
    best_gain = 1e-9
    best_f = -1
    best_left = None
    best_right = None

    for f in feat_subset:
        l_idx = []
        r_idx = []
        l_pos = 0
        for i in indices:
            if X[i][f] == 0:
                l_idx.append(i)
                l_pos += y[i]
            else:
                r_idx.append(i)
        nl = len(l_idx)
        nr = n - nl
        if nl < min_samples_leaf or nr < min_samples_leaf:
            continue
        pl = l_pos / nl
        pr = (pos - l_pos) / nr
        gain = gini_parent - (nl * 2.0 * pl * (1.0 - pl) + nr * 2.0 * pr * (1.0 - pr)) / n
        if gain > best_gain:
            best_gain = gain
            best_f = f
            best_left = l_idx
            best_right = r_idx

    if best_f == -1:
        return round(p1, 4)

    importances[best_f] += n * best_gain

    left_node = build_cart_tree(
        X, y, best_left, depth + 1, max_depth, min_samples_leaf, max_features, n_feats, rng, importances
    )
    right_node = build_cart_tree(
        X, y, best_right, depth + 1, max_depth, min_samples_leaf, max_features, n_feats, rng, importances
    )
    if left_node == right_node:
        return left_node
    return [best_f, left_node, right_node]


def predict_tree(node, x):
    while isinstance(node, list):
        node = node[2] if x[node[0]] == 1 else node[1]
    return node


def train_random_forest(X, y, n_estimators=100, max_depth=6, min_samples_leaf=5, seed=42):
    rng = random.Random(seed)
    n = len(X)
    n_feats = len(X[0])
    max_features = max(1, int(math.sqrt(n_feats)))
    trees = []
    importances = [0.0] * n_feats

    for _ in range(n_estimators):
        boot = [rng.randrange(n) for _ in range(n)]
        tree = build_cart_tree(
            X, y, boot, 0, max_depth, min_samples_leaf, max_features, n_feats, rng, importances
        )
        trees.append(tree)

    total_imp = sum(importances) or 1.0
    importances = [v / total_imp for v in importances]
    return trees, importances


def predict_proba_rf(trees, X):
    n_t = len(trees)
    return [sum(predict_tree(t, x) for t in trees) / n_t for x in X]


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def safe_logit(p):
    p = max(1e-4, min(1.0 - 1e-4, p))
    return math.log(p / (1.0 - p))


def fit_platt_calibration(raw_probs, y):
    xs = [safe_logit(p) for p in raw_probs]
    a, b = 1.0, 0.0
    for _ in range(60):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, yi in zip(xs, y):
            q = sigmoid(a * x + b)
            w = max(1e-6, q * (1.0 - q))
            diff = q - yi
            g0 += diff * x
            g1 += diff
            h00 += w * x * x
            h01 += w * x
            h11 += w
        h00 += 1e-4
        h11 += 1e-4
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (-h01 * g0 + h00 * g1) / det
        a -= da
        b -= db
        if abs(da) + abs(db) < 1e-8:
            break
    return round(a, 6), round(b, 6)


def compute_metrics(y_true, probs):
    preds = [1 if p >= 0.5 else 0 for p in probs]
    n = len(y_true)
    tp = sum(1 for yt, yp in zip(y_true, preds) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, preds) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, preds) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, preds) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    brier = sum((p - yt) ** 2 for p, yt in zip(probs, y_true)) / n
    eps = 1e-6
    logloss = -sum(
        yt * math.log(max(eps, min(1 - eps, p))) + (1 - yt) * math.log(max(eps, min(1 - eps, 1 - p)))
        for p, yt in zip(probs, y_true)
    ) / n

    # 10-bin Expected Calibration Error (ECE)
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        bucket = [(p, yt) for p, yt in zip(probs, y_true) if (lo <= p < hi) or (b == 9 and p == 1.0)]
        if bucket:
            avg_p = sum(p for p, _ in bucket) / len(bucket)
            avg_y = sum(yt for _, yt in bucket) / len(bucket)
            ece += (len(bucket) / n) * abs(avg_p - avg_y)

    # ROC-AUC
    pairs = sorted(zip(probs, y_true), key=lambda z: -z[0])
    pos_tot = sum(y_true)
    neg_tot = n - pos_tot
    tp_c = fp_c = prev_fp = prev_tp = 0
    prev_p = None
    auc = 0.0
    for p, yt in pairs:
        if prev_p is not None and p != prev_p:
            auc += (fp_c - prev_fp) * (tp_c + prev_tp) / 2.0
            prev_fp, prev_tp = fp_c, tp_c
        if yt == 1:
            tp_c += 1
        else:
            fp_c += 1
        prev_p = p
    auc += (fp_c - prev_fp) * (tp_c + prev_tp) / 2.0
    auc /= pos_tot * neg_tot

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "rocAuc": round(auc, 4),
        "brierScore": round(brier, 4),
        "logLoss": round(logloss, 4),
        "ece": round(ece, 4),
        "confusionMatrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def main():
    _, rows, raw_count, missing_dropped, dup_dropped = load_deduplicated_dataset(DATASET_PATH)
    schema = build_onehot_schema()

    X_all = [encode_row(r, schema) for r in rows]
    y_all = [1 if r[TARGET_INDEX] == "Yes" else 0 for r in rows]

    # Stratified 80/20 split (seed=42)
    rng = random.Random(42)
    idx0 = [i for i, y in enumerate(y_all) if y == 0]
    idx1 = [i for i, y in enumerate(y_all) if y == 1]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n_test0 = round(len(idx0) * 0.2)
    n_test1 = round(len(idx1) * 0.2)
    test_idx = idx0[:n_test0] + idx1[:n_test1]
    train_idx = idx0[n_test0:] + idx1[n_test1:]
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = [X_all[i] for i in train_idx]
    y_train = [y_all[i] for i in train_idx]
    X_test = [X_all[i] for i in test_idx]
    y_test = [y_all[i] for i in test_idx]

    # 5-fold cross-validated out-of-fold probabilities for Platt calibration
    folds = 5
    oof_probs = [0.0] * len(X_train)
    for fold in range(folds):
        val_i = [i for i in range(len(X_train)) if i % folds == fold]
        trn_i = [i for i in range(len(X_train)) if i % folds != fold]
        fold_trees, _ = train_random_forest(
            [X_train[i] for i in trn_i],
            [y_train[i] for i in trn_i],
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=5,
            seed=42 + fold,
        )
        vp = predict_proba_rf(fold_trees, [X_train[i] for i in val_i])
        for i, p in zip(val_i, vp):
            oof_probs[i] = p

    platt_a, platt_b = fit_platt_calibration(oof_probs, y_train)

    # Final regularized Random Forest trained on X_train
    trees, oh_importances = train_random_forest(
        X_train, y_train, n_estimators=100, max_depth=6, min_samples_leaf=5, seed=42
    )

    p_train_raw = predict_proba_rf(trees, X_train)
    p_test_raw = predict_proba_rf(trees, X_test)
    p_test_cal = [sigmoid(platt_a * safe_logit(p) + platt_b) for p in p_test_raw]

    train_metrics = compute_metrics(y_train, p_train_raw)
    test_raw_metrics = compute_metrics(y_test, p_test_raw)
    test_cal_metrics = compute_metrics(y_test, p_test_cal)

    # Aggregate Gini feature importance per questionnaire field
    field_importance = Counter()
    for (fname, _, _), imp in zip(schema, oh_importances):
        field_importance[fname] += imp
    grouped_importance = {k: round(v, 4) for k, v in field_importance.most_common()}

    payload = {
        "modelType": "RandomForestClassifier (Gini CART, Platt-Calibrated)",
        "version": "1.0-9feat-calibrated",
        "hyperparameters": {
            "nEstimators": 100,
            "maxDepth": 6,
            "minSamplesLeaf": 5,
            "maxFeatures": int(math.sqrt(len(schema))),
            "randomState": 42,
            "encoding": "OneHotEncoder(handle_unknown='ignore')",
            "calibration": "5-fold OOF Platt Sigmoid Scaling",
        },
        "datasetSummary": {
            "sourceFile": "dataset Update - new.csv",
            "rawRows": raw_count,
            "missingRowsDropped": missing_dropped,
            "duplicateRowsDropped": dup_dropped,
            "cleanRows": len(rows),
            "trainRows": len(X_train),
            "testRows": len(X_test),
            "positiveRate": round(sum(y_all) / len(y_all), 4),
            "targetColumn": "26. Have you ever been diagnosed with diabetes?",
        },
        "evaluation": {
            "trainRaw": train_metrics,
            "testRaw": test_raw_metrics,
            "testCalibrated": test_cal_metrics,
        },
        "featureImportance": grouped_importance,
        "features": [
            {
                "key": spec["key"],
                "csvColumn": spec["csv_column"],
                "categories": spec["categories"],
            }
            for spec in FEATURE_SPECS
        ],
        "oneHotMap": [[fname, cat] for (fname, _, cat) in schema],
        "platt": {"a": platt_a, "b": platt_b},
        "trees": trees,
    }

    js_output = (
        "/* Auto-generated by train_model.py from dataset Update - new.csv.\n"
        " * Regularized 9-feature Random Forest Classifier + 5-fold OOF Platt calibration.\n"
        " * Loaded statically in prediction.html for real client-side predict_proba().\n"
        " */\n"
        "window.DIABETES_RF_MODEL = "
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        + ";\n"
    )

    with open(OUTPUT_JS_PATH, "w", encoding="utf-8") as f:
        f.write(js_output)

    print("Saved model-data.js:", os.path.getsize(OUTPUT_JS_PATH), "bytes")
    print("Train (raw)        :", train_metrics)
    print("Test  (raw)        :", test_raw_metrics)
    print("Test  (calibrated) :", test_cal_metrics)
    print("Platt params       :", {"a": platt_a, "b": platt_b})
    print("Feature importance :", grouped_importance)


if __name__ == "__main__":
    main()
