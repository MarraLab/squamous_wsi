#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_class_weight(x: str):
    x = str(x).strip().lower()
    if x in ("none", "null", ""):
        return None
    if x == "balanced":
        return "balanced"
    raise ValueError(f"Invalid class_weight: {x}")


def make_pipeline(features, X, C, class_weight):
    # Treat binary 0/1 columns as categorical too, since these are clinical flags.
    categorical = [
        c for c in features
        if (
            X[c].dtype == "object"
            or str(X[c].dtype).startswith("category")
            or set(pd.Series(X[c]).dropna().unique()).issubset({0, 1})
        )
    ]
    numeric = [c for c in features if c not in categorical]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="lbfgs",
        max_iter=5000,
    )

    return Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])


def run_oof(df, X, y, groups, features, n_splits, C, class_weight, verbose=False):
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(df), np.nan)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        y_train = y[train_idx]
        y_test = y[test_idx]

        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Fold {fold}: training set has only one class.")

        if verbose and len(np.unique(y_test)) < 2:
            print(f"WARNING: Fold {fold}: test set has only one class; AUC unstable.")

        pipe = make_pipeline(features, X, C=C, class_weight=class_weight)
        pipe.fit(X.iloc[train_idx], y_train)
        oof[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]

        if verbose:
            print(
                f"Fold {fold}: "
                f"n_test={len(test_idx)}, "
                f"pos_test={int(y_test.sum())}, "
                f"neg_test={int((1 - y_test).sum())}"
            )

    auc = roc_auc_score(y, oof)
    pr_auc = average_precision_score(y, oof)
    return oof, auc, pr_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_csv", required=True)
    parser.add_argument("--label_col", default="has_recurrence")
    parser.add_argument("--patient_col", default="patient")
    parser.add_argument(
        "--features",
        default="has_lymph_node,has_invasion,hpv_p53_group",
        help="Comma-separated clinical feature columns",
    )
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--out_csv", default=None)

    parser.add_argument(
        "--C_values",
        default="0.001,0.003,0.01,0.03,0.1,0.3,1.0",
        help="Comma-separated C values for sweep",
    )
    parser.add_argument(
        "--class_weights",
        default="balanced,none",
        help='Comma-separated class_weight values: "balanced" and/or "none"',
    )
    parser.add_argument(
        "--sweep_csv",
        default=None,
        help="Optional CSV path to save hyperparameter sweep results",
    )
    parser.add_argument(
        "--rank_metric",
        default="pr_auc",
        choices=["pr_auc", "roc_auc"],
        help="Metric used to choose best setting",
    )

    args = parser.parse_args()

    clinical_csv = Path(args.clinical_csv)
    features = [x.strip() for x in args.features.split(",") if x.strip()]
    C_values = [float(x.strip()) for x in args.C_values.split(",") if x.strip()]
    class_weights = [parse_class_weight(x) for x in args.class_weights.split(",") if x.strip()]

    df = pd.read_csv(clinical_csv)

    needed = [args.patient_col, args.label_col] + features
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns from {clinical_csv}: {missing}")

    df = df[needed].copy()
    df = df.dropna(subset=[args.patient_col, args.label_col])

    dupes = df[df[args.patient_col].duplicated(keep=False)]
    if not dupes.empty:
        print("WARNING: duplicate patients found. Keeping first row per patient.")
        print(dupes.sort_values(args.patient_col).head(20))
        df = df.drop_duplicates(subset=[args.patient_col], keep="first")

    y = df[args.label_col].astype(int).values
    groups = df[args.patient_col].astype(str).values
    X = df[features]

    if len(np.unique(y)) < 2:
        raise ValueError("Only one class present in label column.")

    print("\nCohort:")
    print(f"  n_patients: {len(df)}")
    print(f"  positives:  {int(y.sum())}")
    print(f"  negatives:  {int((1 - y).sum())}")
    print(f"  positive rate / PR baseline: {float(y.mean()):.4f}")

    print("\nFeature summaries:")
    for col in features:
        print(f"\n{col}")
        print(pd.crosstab(df[col], df[args.label_col], dropna=False))

    print("\nRunning clinical-only hyperparameter sweep...")

    rows = []
    best = None

    for C in C_values:
        for cw in class_weights:
            oof, auc, pr_auc = run_oof(
                df=df,
                X=X,
                y=y,
                groups=groups,
                features=features,
                n_splits=args.n_splits,
                C=C,
                class_weight=cw,
                verbose=False,
            )

            row = {
                "C": C,
                "class_weight": "none" if cw is None else cw,
                "roc_auc": auc,
                "pr_auc": pr_auc,
            }
            rows.append(row)

            score = pr_auc if args.rank_metric == "pr_auc" else auc
            if best is None or score > best["score"]:
                best = {
                    "C": C,
                    "class_weight": cw,
                    "roc_auc": auc,
                    "pr_auc": pr_auc,
                    "score": score,
                    "oof": oof,
                }

    sweep_df = pd.DataFrame(rows).sort_values(
        by=[args.rank_metric, "roc_auc" if args.rank_metric == "pr_auc" else "pr_auc"],
        ascending=False,
    )

    print("\nSweep results:")
    print(sweep_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nBest setting:")
    print(f"  C:            {best['C']}")
    print(f"  class_weight: {'none' if best['class_weight'] is None else best['class_weight']}")
    print(f"  ROC AUC:      {best['roc_auc']:.4f}")
    print(f"  PR AUC:       {best['pr_auc']:.4f}")

    print("\nRefitting best setting with verbose fold summaries:")
    oof, auc, pr_auc = run_oof(
        df=df,
        X=X,
        y=y,
        groups=groups,
        features=features,
        n_splits=args.n_splits,
        C=best["C"],
        class_weight=best["class_weight"],
        verbose=True,
    )

    out = df[[args.patient_col, args.label_col] + features].copy()
    out["pred_clinical"] = oof
    out["best_C"] = best["C"]
    out["best_class_weight"] = "none" if best["class_weight"] is None else best["class_weight"]

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        print(f"\nSaved predictions: {out_path}")

    if args.sweep_csv:
        sweep_path = Path(args.sweep_csv)
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(sweep_path, index=False)
        print(f"Saved sweep results: {sweep_path}")


if __name__ == "__main__":
    main()