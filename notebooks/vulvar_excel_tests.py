# %%
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# %%
EXCEL_PATH = Path(
    "/projects/marralab/mng_prj/Vulvar/Lien VSCC clinical data (selected cases from Master).xlsx"
)
OUTPUT_DIR = Path("outputs/vulvar_excel_tests")
EDA_DIR = OUTPUT_DIR / "eda"
N_SPLITS = 5
RANDOM_STATE = 42
C_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
CLASS_WEIGHTS = ["balanced", None]

# Top five raw features from the previous exploratory permutation importance run.
LABEL_COL = "Recurrence (Y/N)"
ID_COLS = ["name", "phn"]
DATE_COLS = ["DOB (M/D/Y)"]
SITE_COL = "SITE OF CANCER | Right | Left | Anterior | Posterior | Multifocal"
TREATMENT_COL = "Treatment (XRT or Chemo)?"
NUMERIC_COLS = ["treatment_xrt", "treatment_chemo"]
CATEGORICAL_COLS = [
    "Lichen Sclerosus | YES | NO",
    "site_of_cancer",
    "no in-situ identified",
]
RAW_FEATURE_COLS = DATE_COLS + ["Lichen Sclerosus | YES | NO", SITE_COL, "no in-situ identified", TREATMENT_COL]
MODEL_FEATURE_COLS = DATE_COLS + CATEGORICAL_COLS + NUMERIC_COLS


# %%
def clean_header(value):
    if pd.isna(value):
        return None
    value = str(value)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.replace(" --", " | ").replace("--", "|")
    return value or None


def make_unique(names):
    counts = {}
    out = []
    for i, name in enumerate(names):
        base = name or f"column_{i:03d}"
        counts[base] = counts.get(base, 0) + 1
        out.append(base if counts[base] == 1 else f"{base} [{counts[base]}]")
    return out


def load_vulvar_excel(path):
    raw = pd.read_excel(path, header=None)
    headers = make_unique([clean_header(x) for x in raw.iloc[2].tolist()])
    df = raw.iloc[3:].copy()
    df.columns = headers
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def normalize_label(series):
    mapped = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({"Y": 1, "YES": 1, "1": 1, "TRUE": 1, "N": 0, "NO": 0, "0": 0, "FALSE": 0})
    )
    return mapped


def date_feature_frame(df, date_cols):
    out = pd.DataFrame(index=df.index)
    for col in date_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        out[f"{col}__year"] = parsed.dt.year
        out[f"{col}__month"] = parsed.dt.month
        out[f"{col}__ordinal"] = parsed.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    return out


def sanitize_treatment(value):
    if pd.isna(value):
        return 0, 0, "neither"

    text = str(value).strip().lower()
    if not text or text in {"no", "n", "none", "(no)", "__missing__"}:
        return 0, 0, "neither"

    text = text.replace("rxt", "xrt").replace("rtx", "xrt")
    if text in {"yes", "y"}:
        return 1, 1, "xrt + chemo"

    has_xrt = "xrt" in text or "radiation" in text or "chemoradiation" in text or "chemorads" in text
    has_chemo = "chemo" in text or "chemoradiation" in text or "chemorads" in text

    if has_xrt and has_chemo:
        group = "xrt + chemo"
    elif has_xrt:
        group = "xrt only"
    elif has_chemo:
        group = "chemo only"
    else:
        group = "neither"

    return int(has_xrt), int(has_chemo), group


def sanitize_site(value):
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()
    if not text or text in {"nos", "unknown", "n/a", "na", "__missing__"}:
        return "unknown"

    if "right" in text or text == "1":
        return "right"
    if "left" in text or text == "2":
        return "left"
    if "anterior" in text or text == "3":
        return "anterior"
    if "posterior" in text or text == "4":
        return "posterior"

    return "unknown"


def add_derived_features(df):
    out = df.copy()
    treatment = out[TREATMENT_COL].map(sanitize_treatment)
    out["treatment_xrt"] = treatment.map(lambda x: x[0])
    out["treatment_chemo"] = treatment.map(lambda x: x[1])
    out["treatment_group"] = treatment.map(lambda x: x[2])
    out["site_of_cancer"] = out[SITE_COL].map(sanitize_site)
    return out


def make_model(numeric_cols, categorical_cols, C=0.1, class_weight="balanced"):
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=5000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def prepare_features(df):
    missing = [col for col in [LABEL_COL, *RAW_FEATURE_COLS] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns from workbook: {missing}")

    df = add_derived_features(df)
    date_features = date_feature_frame(df, DATE_COLS)
    X = pd.concat(
        [
            df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce"),
            date_features,
            df[CATEGORICAL_COLS]
            .astype("object")
            .where(df[CATEGORICAL_COLS].notna(), "__missing__")
            .astype(str),
        ],
        axis=1,
    )

    categorical_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    numeric_cols = [c for c in numeric_cols if not X[c].isna().all()]
    X = X[numeric_cols + categorical_cols]
    return X, numeric_cols, categorical_cols


def run_oof_auc(
    X,
    y,
    numeric_cols,
    categorical_cols,
    C=0.1,
    class_weight="balanced",
    n_splits=N_SPLITS,
    random_state=RANDOM_STATE,
):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.full(len(y), np.nan)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
        y_train = y[train_idx]
        y_test = y[test_idx]

        pipe = make_model(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            C=C,
            class_weight=class_weight,
        )
        pipe.fit(X.iloc[train_idx], y_train)
        oof[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]

        fold_auc = np.nan
        if len(np.unique(y_test)) == 2:
            fold_auc = roc_auc_score(y_test, oof[test_idx])

        fold_rows.append(
            {
                "fold": fold,
                "n_test": len(test_idx),
                "positives": int(y_test.sum()),
                "negatives": int((1 - y_test).sum()),
                "roc_auc": fold_auc,
            }
        )

    return oof, pd.DataFrame(fold_rows)


def score_oof(X, y, numeric_cols, categorical_cols, C=0.1, class_weight="balanced"):
    oof, fold_report = run_oof_auc(
        X,
        y,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        C=C,
        class_weight=class_weight,
    )
    return {
        "oof": oof,
        "fold_report": fold_report,
        "roc_auc": roc_auc_score(y, oof),
        "pr_auc": average_precision_score(y, oof),
    }


def sweep_parameters(df, y):
    X, numeric_cols, categorical_cols = prepare_features(df)
    rows = []
    best = None

    for C in C_VALUES:
        for class_weight in CLASS_WEIGHTS:
            result = score_oof(
                X,
                y,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                C=C,
                class_weight=class_weight,
            )
            row = {
                "C": C,
                "class_weight": "none" if class_weight is None else class_weight,
                "roc_auc": result["roc_auc"],
                "pr_auc": result["pr_auc"],
            }
            rows.append(row)
            if best is None or (row["roc_auc"], row["pr_auc"]) > (best["roc_auc"], best["pr_auc"]):
                best = {
                    **row,
                    "oof": result["oof"],
                    "fold_report": result["fold_report"],
                }

    sweep = pd.DataFrame(rows).sort_values(["roc_auc", "pr_auc"], ascending=False).reset_index(drop=True)
    return sweep, best


def permutation_importance(df, y, feature_cols, baseline_auc, C, class_weight):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for col in feature_cols:
        shuffled = df.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        X_perm, numeric_perm, categorical_perm = prepare_features(shuffled)
        result = score_oof(
            X_perm,
            y,
            numeric_cols=numeric_perm,
            categorical_cols=categorical_perm,
            C=C,
            class_weight=class_weight,
        )
        rows.append(
            {
                "feature": col,
                "permuted_roc_auc": result["roc_auc"],
                "auc_drop": baseline_auc - result["roc_auc"],
            }
        )

    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)


def short_label(label):
    label = label.replace("SITE OF CANCER | Right | Left | Anterior | Posterior | Multifocal", "site of cancer")
    label = label.replace("site_of_cancer", "site of cancer")
    label = label.replace("Lichen Sclerosus | YES | NO", "lichen sclerosus")
    label = label.replace("Treatment (XRT or Chemo)?", "treatment")
    label = label.replace("treatment_xrt", "treatment: XRT")
    label = label.replace("treatment_chemo", "treatment: chemo")
    return label


def plot_feature_importance(importance_df, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = importance_df.sort_values("auc_drop", ascending=True)
    ax.barh(plot_df["feature"].map(short_label), plot_df["auc_drop"], color="#4C78A8")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("OOF ROC AUC drop after permutation")
    ax.set_title("Top-5 Clinical Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true, y_pred, roc_auc, path):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#4C78A8", linewidth=2, label=f"OOF ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Top-5 Clinical Model ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pr_curve(y_true, y_pred, pr_auc, path):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, color="#F58518", linewidth=2, label=f"OOF PR AUC = {pr_auc:.3f}")
    ax.axhline(y_true.mean(), color="0.5", linestyle="--", linewidth=1, label=f"baseline = {y_true.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Top-5 Clinical Model Precision-Recall")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def recurrence_rate_by_category(df, y, column):
    df = add_derived_features(df)
    values = df[column].astype("object").where(df[column].notna(), "__missing__").astype(str).str.strip()
    values = values.replace({"": "__missing__"})
    summary = (
        pd.DataFrame({"value": values, "recurrence": y})
        .groupby("value", dropna=False)["recurrence"]
        .agg(n="count", recurrences="sum", recurrence_rate="mean")
        .reset_index()
        .sort_values(["recurrence_rate", "n"], ascending=[False, False])
    )
    return summary


def plot_recurrence_rate_by_category(summary, column, path, p_value=np.nan, test_name=""):
    plot_df = summary.sort_values(["recurrence_rate", "n"], ascending=[True, True]).reset_index(drop=True)
    fig_height = max(8, 0.42 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    x = np.arange(len(plot_df))
    bars = ax.barh(x, plot_df["recurrence_rate"], color="#54A24B")

    for bar, row in zip(bars, plot_df.itertuples(index=False)):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row.recurrences)}/{int(row.n)}",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(x)
    ax.set_yticklabels(plot_df["value"], fontsize=9)
    ax.set_xlim(0, min(1.15, max(0.2, plot_df["recurrence_rate"].max() + 0.15)))
    ax.set_xlabel("Recurrence rate")
    subtitle = f"\n{test_name}, {format_p_value(p_value)}" if test_name else ""
    ax.set_title(f"Recurrence Rate by {short_label(column).title()}{subtitle}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()


def format_p_value(p_value):
    if pd.isna(p_value):
        return "p = NA"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def numeric_p_value(values, y):
    plot_df = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "recurrence": y}).dropna()
    group_0 = plot_df.loc[plot_df["recurrence"].eq(0), "value"]
    group_1 = plot_df.loc[plot_df["recurrence"].eq(1), "value"]
    if len(group_0) == 0 or len(group_1) == 0:
        return np.nan, "Mann-Whitney U"
    if group_0.nunique() <= 1 and group_1.nunique() <= 1 and group_0.iloc[0] == group_1.iloc[0]:
        return np.nan, "Mann-Whitney U"
    return mannwhitneyu(group_0, group_1, alternative="two-sided").pvalue, "Mann-Whitney U"


def categorical_p_value(values, y):
    clean = values.astype("object").where(values.notna(), "__missing__").astype(str).str.strip()
    clean = clean.replace({"": "__missing__"})
    table = pd.crosstab(clean, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, "Fisher exact" if table.shape == (2, 2) else "chi-square"
    if table.shape == (2, 2):
        return fisher_exact(table.to_numpy()).pvalue, "Fisher exact"
    return chi2_contingency(table.to_numpy()).pvalue, "chi-square"


def plot_numeric_vs_recurrence(df, y, column, path):
    values = pd.to_numeric(df[column], errors="coerce")
    plot_df = pd.DataFrame({"value": values, "recurrence": y}).dropna()
    if plot_df.empty:
        return np.nan, "Mann-Whitney U"

    groups = [
        plot_df.loc[plot_df["recurrence"].eq(0), "value"].to_numpy(),
        plot_df.loc[plot_df["recurrence"].eq(1), "value"].to_numpy(),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.boxplot(groups, tick_labels=["No recurrence", "Recurrence"], showfliers=False)
    rng = np.random.default_rng(RANDOM_STATE)
    for i, vals in enumerate(groups, start=1):
        if len(vals):
            jitter = rng.normal(0, 0.035, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.6, s=22)
    ax.set_ylabel(short_label(column))
    p_value, test_name = numeric_p_value(values, y)
    ax.set_title(f"{short_label(column).title()} by Recurrence\n{test_name}, {format_p_value(p_value)}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return p_value, test_name


def plot_categorical_vs_recurrence(df, y, column, path):
    summary = recurrence_rate_by_category(df, y, column)
    values = add_derived_features(df)[column]
    p_value, test_name = categorical_p_value(values, y)
    plot_recurrence_rate_by_category(summary, column, path, p_value=p_value, test_name=test_name)
    return summary, p_value, test_name


def make_eda_plots(df, X, y, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for col in DATE_COLS:
        for derived_col in [f"{col}__year", f"{col}__month", f"{col}__ordinal"]:
            if derived_col in X.columns:
                path = out_dir / f"date_{safe_filename(derived_col)}.png"
                p_value, test_name = plot_numeric_vs_recurrence(X, y, derived_col, path)
                rows.append(
                    {
                        "column": derived_col,
                        "role": "date-derived",
                        "test": test_name,
                        "p_value": p_value,
                        "plot": str(path),
                    }
                )

    for col in NUMERIC_COLS:
        if col in X.columns:
            path = out_dir / f"numeric_{safe_filename(col)}.png"
            p_value, test_name = plot_numeric_vs_recurrence(X, y, col, path)
            rows.append(
                {
                    "column": col,
                    "role": "numeric",
                    "test": test_name,
                    "p_value": p_value,
                    "plot": str(path),
                }
            )

    for col in CATEGORICAL_COLS:
        path = out_dir / f"categorical_{safe_filename(col)}.png"
        summary, p_value, test_name = plot_categorical_vs_recurrence(df, y, col, path)
        summary.to_csv(out_dir / f"categorical_{safe_filename(col)}.csv", index=False)
        rows.append(
            {
                "column": col,
                "role": "categorical",
                "test": test_name,
                "p_value": p_value,
                "plot": str(path),
            }
        )

    path = out_dir / "categorical_treatment_group.png"
    treatment_summary, p_value, test_name = plot_categorical_vs_recurrence(df, y, "treatment_group", path)
    treatment_summary.to_csv(out_dir / "categorical_treatment_group.csv", index=False)
    rows.append(
        {
            "column": "treatment_group",
            "role": "categorical",
            "test": test_name,
            "p_value": p_value,
            "plot": str(path),
        }
    )

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "eda_plot_manifest.csv", index=False)
    return report


# %%
df = load_vulvar_excel(EXCEL_PATH)

y = normalize_label(df[LABEL_COL])
keep = y.notna()
df_model = df.loc[keep].reset_index(drop=True)
df_model = add_derived_features(df_model)
y = y.loc[keep].astype(int).to_numpy()

X, numeric_cols, categorical_cols = prepare_features(df_model)

print("Loaded workbook")
print(f"  path: {EXCEL_PATH}")
print(f"  rows with recurrence label: {len(df_model)}")
print(f"  recurrence positives: {int(y.sum())}")
print(f"  recurrence negatives: {int((1 - y).sum())}")
print(f"  label column: {LABEL_COL}")

print("\nExplicit feature set")
for col in MODEL_FEATURE_COLS:
    print(f"  - {col}")

if len(np.unique(y)) < 2:
    raise ValueError("Only one recurrence class is present after label cleaning.")

if min(np.bincount(y)) < N_SPLITS:
    raise ValueError(
        f"Need at least {N_SPLITS} samples in each class for {N_SPLITS}-fold CV; "
        f"class counts are {np.bincount(y).tolist()}."
    )

baseline = score_oof(X, y, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

print("\n5-fold OOF report")
print(baseline["fold_report"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nOOF ROC AUC: {baseline['roc_auc']:.4f}")
print(f"OOF PR AUC:  {baseline['pr_auc']:.4f}")
print(f"PR baseline: {y.mean():.4f}")

print("\nParameter sweep on top 5 features")
sweep_df, best = sweep_parameters(df_model, y)
print(sweep_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nBest top-5-feature model")
print(f"  C:            {best['C']}")
print(f"  class_weight: {best['class_weight']}")
print(f"  OOF ROC AUC:  {best['roc_auc']:.4f}")
print(f"  OOF PR AUC:   {best['pr_auc']:.4f}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
eda_report = make_eda_plots(df_model, X, y, EDA_DIR)

best_class_weight = None if best["class_weight"] == "none" else best["class_weight"]
importance_df = permutation_importance(
    df_model,
    y,
    feature_cols=RAW_FEATURE_COLS,
    baseline_auc=best["roc_auc"],
    C=best["C"],
    class_weight=best_class_weight,
)

print("\nTop-5 feature importance")
print(importance_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

plot_feature_importance(importance_df, OUTPUT_DIR / "vulvar_clinical_top5_feature_importance.png")
plot_roc_curve(y, best["oof"], best["roc_auc"], OUTPUT_DIR / "vulvar_clinical_top5_roc.png")
plot_pr_curve(y, best["oof"], best["pr_auc"], OUTPUT_DIR / "vulvar_clinical_top5_pr.png")

treatment_summary = recurrence_rate_by_category(df_model, y, "treatment_group")
print("\nRecurrence rate by treatment")
print(treatment_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
plot_recurrence_rate_by_category(
    treatment_summary,
    "sanitized treatment group",
    OUTPUT_DIR / "vulvar_clinical_treatment_recurrence_rate.png",
)

oof_df = df_model[[col for col in ID_COLS if col in df_model.columns]].copy()
oof_df[LABEL_COL] = y
oof_df["pred_recurrence_top5"] = best["oof"]
oof_df.to_csv(OUTPUT_DIR / "vulvar_clinical_oof.csv", index=False)

feature_report = pd.DataFrame(
    [{"column": col, "role": "date"} for col in DATE_COLS]
    + [{"column": col, "role": "numeric"} for col in NUMERIC_COLS]
    + [{"column": col, "role": "categorical"} for col in CATEGORICAL_COLS]
    + [{"column": TREATMENT_COL, "role": "sanitized into treatment_xrt and treatment_chemo"}]
)
feature_report.to_csv(OUTPUT_DIR / "vulvar_clinical_top5_features.csv", index=False)
baseline["fold_report"].to_csv(OUTPUT_DIR / "vulvar_clinical_top5_baseline_fold_report.csv", index=False)
sweep_df.to_csv(OUTPUT_DIR / "vulvar_clinical_top5_sweep.csv", index=False)
best["fold_report"].to_csv(OUTPUT_DIR / "vulvar_clinical_top5_best_fold_report.csv", index=False)
importance_df.to_csv(OUTPUT_DIR / "vulvar_clinical_top5_importance.csv", index=False)
treatment_summary.to_csv(OUTPUT_DIR / "vulvar_clinical_treatment_recurrence_rate.csv", index=False)
eda_report.to_csv(OUTPUT_DIR / "vulvar_clinical_eda_plots.csv", index=False)

print(f"\nSaved OOF predictions: {OUTPUT_DIR / 'vulvar_clinical_oof.csv'}")
print(f"Saved explicit feature list: {OUTPUT_DIR / 'vulvar_clinical_top5_features.csv'}")
print(f"Saved parameter sweep: {OUTPUT_DIR / 'vulvar_clinical_top5_sweep.csv'}")
print(f"Saved best fold report: {OUTPUT_DIR / 'vulvar_clinical_top5_best_fold_report.csv'}")
print(f"Saved feature importance: {OUTPUT_DIR / 'vulvar_clinical_top5_importance.csv'}")
print(f"Saved feature importance plot: {OUTPUT_DIR / 'vulvar_clinical_top5_feature_importance.png'}")
print(f"Saved ROC plot: {OUTPUT_DIR / 'vulvar_clinical_top5_roc.png'}")
print(f"Saved PR plot: {OUTPUT_DIR / 'vulvar_clinical_top5_pr.png'}")
print(f"Saved treatment recurrence summary: {OUTPUT_DIR / 'vulvar_clinical_treatment_recurrence_rate.csv'}")
print(f"Saved treatment recurrence plot: {OUTPUT_DIR / 'vulvar_clinical_treatment_recurrence_rate.png'}")
print(f"Saved EDA plot manifest: {OUTPUT_DIR / 'vulvar_clinical_eda_plots.csv'}")
print(f"Saved EDA plots under: {EDA_DIR}")

# %%
