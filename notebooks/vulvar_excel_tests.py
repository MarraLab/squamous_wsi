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
METADATA_PATH = Path("/projects/marralab/rcorbett_prj/vulvar/original_files/vulvar_clin.csv")
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs" / "vulvar_excel_tests"
EDA_DIR = OUTPUT_DIR / "eda"
N_SPLITS = 5
RANDOM_STATE = 42
C_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
CLASS_WEIGHTS = ["balanced", None]

# Top five raw features from the previous exploratory permutation importance run.
LABEL_COL = "Recurrence (Y/N)"
ID_COLS = ["name", "phn", "EXCISION Surgical Case Number"]
DATE_COLS = ["DOB (M/D/Y)"]
SITE_COL = "SITE OF CANCER | Right | Left | Anterior | Posterior | Multifocal"
TREATMENT_COL = "Treatment (XRT or Chemo)?"
NUMERIC_COLS = ["treatment_xrt", "treatment_chemo"]
METADATA_KEY_COL = "EXCISION Surgical Case Number"
METADATA_JOIN_COL = "patient"
METADATA_LABEL_COL = "has_recurrence"
METADATA_NUMERIC_COLS = ["has_lymph_node", "has_invasion"]
METADATA_CATEGORICAL_COLS = ["hpv_p53_group", "risk"]
MODEL_NUMERIC_COLS = NUMERIC_COLS + METADATA_NUMERIC_COLS
CATEGORICAL_COLS = [
    "Lichen Sclerosus | YES | NO",
    "site_of_cancer",
    "no in-situ identified",
] + METADATA_CATEGORICAL_COLS
RAW_FEATURE_COLS = DATE_COLS + ["Lichen Sclerosus | YES | NO", SITE_COL, "no in-situ identified", TREATMENT_COL]
METADATA_FEATURE_COLS = METADATA_NUMERIC_COLS + METADATA_CATEGORICAL_COLS
MODEL_FEATURE_COLS = DATE_COLS + CATEGORICAL_COLS + MODEL_NUMERIC_COLS
PERMUTATION_FEATURE_COLS = RAW_FEATURE_COLS + METADATA_FEATURE_COLS
REMOVED_OUTPUTS = [
    EDA_DIR / "categorical_treatment_group.png",
    EDA_DIR / "categorical_treatment_group.csv",
]


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


def clean_case_id(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return text or np.nan


def load_vulvar_metadata(path):
    metadata = pd.read_csv(path)
    cols = [METADATA_JOIN_COL, METADATA_LABEL_COL, *METADATA_NUMERIC_COLS, *METADATA_CATEGORICAL_COLS]
    missing = [col for col in cols if col not in metadata.columns]
    if missing:
        raise ValueError(f"Missing expected metadata columns from {path}: {missing}")

    metadata = metadata[cols].copy()
    metadata["metadata_case_id"] = metadata[METADATA_JOIN_COL].map(clean_case_id)
    metadata = metadata.dropna(subset=["metadata_case_id"])
    metadata = metadata.drop_duplicates("metadata_case_id", keep="first")
    metadata = metadata.drop(columns=METADATA_JOIN_COL)
    return metadata


def join_vulvar_metadata(df, metadata):
    if METADATA_KEY_COL not in df.columns:
        raise ValueError(f"Missing expected metadata join column from workbook: {METADATA_KEY_COL}")

    out = df.copy()
    out["metadata_case_id"] = out[METADATA_KEY_COL].map(clean_case_id)
    out = out.merge(metadata, on="metadata_case_id", how="left")
    out["metadata_matched"] = out[METADATA_CATEGORICAL_COLS + METADATA_NUMERIC_COLS].notna().any(axis=1)
    return out


def normalize_label(series):
    mapped = (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map(
            {
                "Y": 1,
                "YES": 1,
                "1": 1,
                "1.0": 1,
                "TRUE": 1,
                "N": 0,
                "NO": 0,
                "0": 0,
                "0.0": 0,
                "FALSE": 0,
            }
        )
    )
    return mapped


def add_combined_recurrence_label(df):
    out = df.copy()
    excel_label = normalize_label(out[LABEL_COL])
    metadata_label = normalize_label(out[METADATA_LABEL_COL]) if METADATA_LABEL_COL in out.columns else pd.Series(np.nan, index=out.index)
    combined = excel_label.where(excel_label.notna(), metadata_label)

    out["recurrence_excel_clean"] = excel_label
    out["recurrence_metadata_clean"] = metadata_label
    out["recurrence_label_disagreement"] = excel_label.notna() & metadata_label.notna() & excel_label.ne(metadata_label)
    out["recurrence_label_source"] = np.select(
        [excel_label.notna(), metadata_label.notna()],
        ["excel", "metadata"],
        default="none",
    )
    out["recurrence_combined"] = combined
    return out


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
            df[MODEL_NUMERIC_COLS].apply(pd.to_numeric, errors="coerce"),
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


def cleaned_values_frame(df, y, X):
    id_cols = [col for col in ID_COLS if col in df.columns]
    label_audit_cols = [
        LABEL_COL,
        METADATA_LABEL_COL,
        "recurrence_excel_clean",
        "recurrence_metadata_clean",
        "recurrence_label_source",
        "recurrence_label_disagreement",
    ]
    label_audit_cols = [col for col in label_audit_cols if col in df.columns]
    raw_cols = [col for col in RAW_FEATURE_COLS if col in df.columns and col not in label_audit_cols]
    derived_cols = ["treatment_xrt", "treatment_chemo", "treatment_group", "site_of_cancer"]
    derived_cols = [col for col in derived_cols if col in df.columns]
    metadata_cols = ["metadata_case_id", "metadata_matched"] + METADATA_FEATURE_COLS
    metadata_cols = [col for col in metadata_cols if col in df.columns]

    out = df[id_cols + label_audit_cols + raw_cols + derived_cols + metadata_cols].copy()
    out.insert(len(id_cols), "recurrence", y)

    model_values = X.copy()
    model_values.columns = [f"model__{col}" for col in model_values.columns]
    return pd.concat([out.reset_index(drop=True), model_values.reset_index(drop=True)], axis=1)


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
    label = label.replace("hpv_p53_group", "HPV/p53 group")
    label = label.replace("has_lymph_node", "lymph node involvement")
    label = label.replace("has_invasion", "invasion")
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


def decision_curve(y_true, y_pred, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    prevalence = y_true.mean()
    rows = []

    for threshold in thresholds:
        predicted_positive = y_pred >= threshold
        tp = int(np.sum(predicted_positive & (y_true == 1)))
        fp = int(np.sum(predicted_positive & (y_true == 0)))
        odds = threshold / (1 - threshold)
        rows.append(
            {
                "threshold": threshold,
                "net_benefit_model": (tp / n) - (fp / n) * odds,
                "net_benefit_treat_all": prevalence - (1 - prevalence) * odds,
                "net_benefit_treat_none": 0.0,
                "true_positives": tp,
                "false_positives": fp,
                "predicted_positives": int(predicted_positive.sum()),
            }
        )

    return pd.DataFrame(rows)


def plot_decision_curve(dca_df, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(dca_df["threshold"], dca_df["net_benefit_model"], color="#4C78A8", linewidth=2, label="Clinical model")
    ax.plot(
        dca_df["threshold"],
        dca_df["net_benefit_treat_all"],
        color="#F58518",
        linewidth=1.6,
        linestyle="--",
        label="Treat all",
    )
    ax.plot(
        dca_df["threshold"],
        dca_df["net_benefit_treat_none"],
        color="0.35",
        linewidth=1.2,
        linestyle=":",
        label="Treat none",
    )
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.set_xlim(0, 1)
    zoom_cols = ["net_benefit_model", "net_benefit_treat_none"]
    visible_treat_all = dca_df.loc[dca_df["net_benefit_treat_all"].ge(-0.10), "net_benefit_treat_all"]
    y_min = min(-0.05, dca_df[zoom_cols].min().min() - 0.02)
    y_max = max(0.1, dca_df[zoom_cols].max().max() + 0.03)
    if not visible_treat_all.empty:
        y_min = min(y_min, visible_treat_all.min() - 0.02)
        y_max = max(y_max, visible_treat_all.max() + 0.03)
    y_min = max(y_min, -0.12)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis")
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


def is_binary_numeric(values):
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return False
    return set(clean.unique()).issubset({0, 1})


def recurrence_fraction_by_binary_numeric(df, y, column):
    values = pd.to_numeric(df[column], errors="coerce")
    plot_df = pd.DataFrame({"value": values, "recurrence": y}).dropna()
    plot_df["value"] = plot_df["value"].astype(int)

    summary = (
        plot_df.groupby(["value", "recurrence"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    complete_index = pd.MultiIndex.from_product(
        [[0, 1], [0, 1]],
        names=["value", "recurrence"],
    )
    summary = (
        summary.set_index(["value", "recurrence"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )
    summary["value_n"] = summary.groupby("value")["n"].transform("sum")
    summary["fraction"] = np.where(summary["value_n"].gt(0), summary["n"] / summary["value_n"], np.nan)
    summary["recurrence_label"] = summary["recurrence"].map({0: "No recurrence", 1: "Recurrence"})
    summary["value_label"] = summary["value"].map(lambda x: f"{short_label(column)} = {x}")
    return summary


def plot_binary_recurrence_summary(summary, column, ax):
    value_labels = [f"{short_label(column)} = 0", f"{short_label(column)} = 1"]
    x = np.arange(len(value_labels))
    width = 0.36
    colors = {0: "#4C78A8", 1: "#F58518"}

    for offset, recurrence in [(-width / 2, 0), (width / 2, 1)]:
        recurrence_df = (
            summary.loc[summary["recurrence"].eq(recurrence)]
            .set_index("value")
            .reindex([0, 1])
        )
        bars = ax.bar(
            x + offset,
            recurrence_df["fraction"],
            width=width,
            color=colors[recurrence],
            label={0: "No recurrence", 1: "Recurrence"}[recurrence],
        )
        for bar, row in zip(bars, recurrence_df.itertuples()):
            if pd.isna(row.fraction):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{int(row.n)}/{int(row.value_n)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(value_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Fraction within value group")
    ax.legend(title="Outcome", frameon=False)
    return ax


def binary_summary_p_value(summary):
    table = summary.pivot(index="value", columns="recurrence", values="n").reindex(index=[0, 1], columns=[0, 1])
    table = table.fillna(0).astype(int)
    if table.shape == (2, 2):
        return fisher_exact(table.to_numpy()).pvalue, "Fisher exact"
    return chi2_contingency(table.to_numpy()).pvalue, "chi-square"


def plot_binary_numeric_vs_recurrence(df, y, column, path):
    summary = recurrence_fraction_by_binary_numeric(df, y, column)
    if summary["value_n"].sum() == 0:
        return np.nan, "Fisher exact", summary

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    plot_binary_recurrence_summary(summary, column, ax)
    p_value, test_name = categorical_p_value(pd.to_numeric(df[column], errors="coerce"), y)
    ax.set_title(f"Recurrence Fractions by {short_label(column).title()}\n{test_name}, {format_p_value(p_value)}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return p_value, test_name, summary


def treatment_binary_recurrence_summaries(df, y):
    rows = []
    for col in NUMERIC_COLS:
        summary = recurrence_fraction_by_binary_numeric(df, y, col)
        summary.insert(0, "column", col)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def plot_treatment_binary_recurrence_summaries(summary, path):
    fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(12, 4.5), sharey=True)
    if len(NUMERIC_COLS) == 1:
        axes = [axes]

    for ax, col in zip(axes, NUMERIC_COLS):
        col_summary = summary.loc[summary["column"].eq(col)].drop(columns="column")
        plot_binary_recurrence_summary(col_summary, col, ax)
        p_value, test_name = binary_summary_p_value(col_summary)
        ax.set_title(f"{short_label(col).title()}\n{test_name}, {format_p_value(p_value)}")

    fig.suptitle("Recurrence Fractions by Cleaned Treatment Indicators")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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

    for col in MODEL_NUMERIC_COLS:
        if col in X.columns:
            path = out_dir / f"numeric_{safe_filename(col)}.png"
            if is_binary_numeric(X[col]):
                p_value, test_name, summary = plot_binary_numeric_vs_recurrence(X, y, col, path)
                summary.to_csv(out_dir / f"numeric_{safe_filename(col)}.csv", index=False)
                role = "numeric-binary"
            else:
                p_value, test_name = plot_numeric_vs_recurrence(X, y, col, path)
                role = "numeric"
            rows.append(
                {
                    "column": col,
                    "role": role,
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

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "eda_plot_manifest.csv", index=False)
    return report


def remove_stale_outputs(paths):
    for path in paths:
        path.unlink(missing_ok=True)


# %%
df = load_vulvar_excel(EXCEL_PATH)
metadata = load_vulvar_metadata(METADATA_PATH)
df = join_vulvar_metadata(df, metadata)
df = add_combined_recurrence_label(df)

y = df["recurrence_combined"]
keep = y.notna()
df_model = df.loc[keep].reset_index(drop=True)
df_model = add_derived_features(df_model)
y = y.loc[keep].astype(int).to_numpy()

X, numeric_cols, categorical_cols = prepare_features(df_model)

print("Loaded workbook")
print(f"  path: {EXCEL_PATH}")
print(f"Loaded metadata")
print(f"  path: {METADATA_PATH}")
print(f"  matched rows with recurrence label: {int(df_model['metadata_matched'].sum())}/{len(df_model)}")
print(f"  rows with recurrence label: {len(df_model)}")
print(f"  label source counts: {df_model['recurrence_label_source'].value_counts().to_dict()}")
print(f"  label disagreements where both sources present: {int(df_model['recurrence_label_disagreement'].sum())}")
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
remove_stale_outputs(REMOVED_OUTPUTS)
cleaned_values = cleaned_values_frame(df_model, y, X)
eda_report = make_eda_plots(df_model, X, y, EDA_DIR)

best_class_weight = None if best["class_weight"] == "none" else best["class_weight"]
importance_df = permutation_importance(
    df_model,
    y,
    feature_cols=PERMUTATION_FEATURE_COLS,
    baseline_auc=best["roc_auc"],
    C=best["C"],
    class_weight=best_class_weight,
)

print("\nTop-5 feature importance")
print(importance_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

plot_feature_importance(importance_df, OUTPUT_DIR / "vulvar_clinical_top5_feature_importance.png")
plot_roc_curve(y, best["oof"], best["roc_auc"], OUTPUT_DIR / "vulvar_clinical_top5_roc.png")
plot_pr_curve(y, best["oof"], best["pr_auc"], OUTPUT_DIR / "vulvar_clinical_top5_pr.png")
dca_df = decision_curve(y, best["oof"])
plot_decision_curve(dca_df, OUTPUT_DIR / "vulvar_clinical_top5_dca.png")

treatment_summary = treatment_binary_recurrence_summaries(X, y)
print("\nRecurrence fractions by cleaned treatment indicators")
print(treatment_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
plot_treatment_binary_recurrence_summaries(
    treatment_summary,
    OUTPUT_DIR / "vulvar_clinical_treatment_recurrence_rate.png",
)

oof_df = df_model[[col for col in ID_COLS if col in df_model.columns]].copy()
oof_df[LABEL_COL] = y
oof_df["pred_recurrence_top5"] = best["oof"]
oof_df.to_csv(OUTPUT_DIR / "vulvar_clinical_oof.csv", index=False)

feature_report = pd.DataFrame(
    [{"column": col, "role": "date"} for col in DATE_COLS]
    + [{"column": col, "role": "numeric-binary treatment indicator"} for col in NUMERIC_COLS]
    + [{"column": col, "role": "metadata numeric"} for col in METADATA_NUMERIC_COLS]
    + [{"column": col, "role": "categorical"} for col in CATEGORICAL_COLS]
    + [{"column": TREATMENT_COL, "role": "sanitized into treatment_xrt and treatment_chemo"}]
)
cleaned_values.to_csv(OUTPUT_DIR / "vulvar_clinical_cleaned_values.csv", index=False)
feature_report.to_csv(OUTPUT_DIR / "vulvar_clinical_top5_features.csv", index=False)
baseline["fold_report"].to_csv(OUTPUT_DIR / "vulvar_clinical_top5_baseline_fold_report.csv", index=False)
sweep_df.to_csv(OUTPUT_DIR / "vulvar_clinical_top5_sweep.csv", index=False)
best["fold_report"].to_csv(OUTPUT_DIR / "vulvar_clinical_top5_best_fold_report.csv", index=False)
importance_df.to_csv(OUTPUT_DIR / "vulvar_clinical_top5_importance.csv", index=False)
dca_df.to_csv(OUTPUT_DIR / "vulvar_clinical_top5_dca.csv", index=False)
treatment_summary.to_csv(OUTPUT_DIR / "vulvar_clinical_treatment_recurrence_rate.csv", index=False)
eda_report.to_csv(OUTPUT_DIR / "vulvar_clinical_eda_plots.csv", index=False)

print(f"\nSaved OOF predictions: {OUTPUT_DIR / 'vulvar_clinical_oof.csv'}")
print(f"Saved cleaned values: {OUTPUT_DIR / 'vulvar_clinical_cleaned_values.csv'}")
print(f"Saved explicit feature list: {OUTPUT_DIR / 'vulvar_clinical_top5_features.csv'}")
print(f"Saved parameter sweep: {OUTPUT_DIR / 'vulvar_clinical_top5_sweep.csv'}")
print(f"Saved best fold report: {OUTPUT_DIR / 'vulvar_clinical_top5_best_fold_report.csv'}")
print(f"Saved feature importance: {OUTPUT_DIR / 'vulvar_clinical_top5_importance.csv'}")
print(f"Saved feature importance plot: {OUTPUT_DIR / 'vulvar_clinical_top5_feature_importance.png'}")
print(f"Saved ROC plot: {OUTPUT_DIR / 'vulvar_clinical_top5_roc.png'}")
print(f"Saved PR plot: {OUTPUT_DIR / 'vulvar_clinical_top5_pr.png'}")
print(f"Saved DCA values: {OUTPUT_DIR / 'vulvar_clinical_top5_dca.csv'}")
print(f"Saved DCA plot: {OUTPUT_DIR / 'vulvar_clinical_top5_dca.png'}")
print(f"Saved treatment recurrence summary: {OUTPUT_DIR / 'vulvar_clinical_treatment_recurrence_rate.csv'}")
print(f"Saved treatment recurrence plot: {OUTPUT_DIR / 'vulvar_clinical_treatment_recurrence_rate.png'}")
print(f"Saved EDA plot manifest: {OUTPUT_DIR / 'vulvar_clinical_eda_plots.csv'}")
print(f"Saved EDA plots under: {EDA_DIR}")

# %%
