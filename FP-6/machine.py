import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score, train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

_MACHINE_OUT = {}

def load_earthquakes(path="Earthquakes.csv"):
    df = pd.read_csv(path).copy()

    required = ["Deaths", "Mag", "Focal Depth (km)"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df[df["Deaths"].notna() & df["Mag"].notna() & df["Focal Depth (km)"].notna()].copy()
    df["Deaths"] = pd.to_numeric(df["Deaths"], errors="coerce")
    df = df[df["Deaths"].notna()].copy()

    df["log_deaths"] = np.log10(df["Deaths"] + 1)

    if "MMI Int" in df.columns:
        df["MMI Int"] = pd.to_numeric(df["MMI Int"], errors="coerce")

    for col in ["Latitude", "Longitude", "Year", "Tsu", "Vol"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def label_mmi_group(x):
    if pd.isna(x):
        return np.nan
    return "MMI ≥ IX" if x >= 9 else "MMI ≤ VIII"


def make_tiers(df):
    tier_A = df[(df["Mag"] >= 7.0) & (df["Mag"] <= 7.9)].copy()
    tier_B = df[(df["Mag"] >= 6.5) & (df["Mag"] <= 6.9)].copy()
    return {"Mag 7.0–7.9": tier_A, "Mag 6.5–6.9": tier_B}


def make_pipe(feature_cols, alpha=10.0, random_state=42):
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), feature_cols)
        ],
        remainder="drop"
    )
    model = Ridge(alpha=alpha, random_state=random_state)
    return Pipeline([("preprocess", preprocess), ("model", model)])

def cv_rmse(df_sub, feature_cols, n_splits=5, random_state=42):
    sub = df_sub.dropna(subset=feature_cols + ["log_deaths"]).copy()
    if len(sub) < max(10, n_splits):
        return np.array([]), sub

    X = sub[feature_cols]
    y = sub["log_deaths"]

    pipe = make_pipe(feature_cols, alpha=10.0, random_state=random_state)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rmse = -cross_val_score(pipe, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return rmse, sub


#ΔRMSE from adding MMI, by adjacent magnitude tiers
def compute_delta_rmse_by_tier(df, random_state=42):
    if "MMI Int" not in df.columns:
        raise ValueError("Column 'MMI Int' not found — cannot compute ΔRMSE with MMI.")

    BASELINE = ["Mag", "Focal Depth (km)"]
    PLUS_MMI = ["Mag", "Focal Depth (km)", "MMI Int"]

    tiers = make_tiers(df)

    rows = []
    for tier_name, tier_df in tiers.items():
        tier_df = tier_df.dropna(subset=["MMI Int"]).copy()
        if len(tier_df) < 20:
            continue

        rmse_base, used_base = cv_rmse(tier_df, BASELINE, random_state=random_state)
        rmse_mmi,  used_mmi  = cv_rmse(tier_df, PLUS_MMI, random_state=random_state)

        if len(rmse_base) == 0 or len(rmse_mmi) == 0:
            continue

        n = min(len(rmse_base), len(rmse_mmi))
        delta = rmse_base[:n] - rmse_mmi[:n]

        for i in range(n):
            rows.append({
                "tier": tier_name,
                "fold": i + 1,
                "delta_rmse": float(delta[i]),
                "rmse_baseline": float(rmse_base[i]),
                "rmse_plus_mmi": float(rmse_mmi[i]),
                "N_used": int(min(len(used_base), len(used_mmi)))
            })

    return pd.DataFrame(rows)


def plot_main_result_1_delta_rmse():
    delta_df = _MACHINE_OUT.get("delta_df")
    if delta_df is None or len(delta_df) == 0:
        raise ValueError("delta_df is empty. Run run_all_machine_analyses() first.")

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=delta_df, x="tier", y="delta_rmse")
    sns.stripplot(data=delta_df, x="tier", y="delta_rmse", color="black", alpha=0.45)
    plt.axhline(0, linestyle="--")
    plt.title("Improvement from adding MMI (cross-validation)\nΔRMSE = RMSE(baseline) − RMSE(+MMI)")
    plt.xlabel("")
    plt.ylabel("ΔRMSE (positive = better with MMI)")
    plt.tight_layout()
    plt.show()

    print("ΔRMSE summary (positive means MMI helps):")
    print(delta_df.groupby("tier")["delta_rmse"].agg(["mean", "std", "count"]))


# Absolute error by MMI group on held-out test set
def compute_error_by_mmi(
    df,
    random_state=42,
    min_group_n=8,
    min_rows_total=25,
    use_quantile_fallback=True
):


    core = ["Mag", "Focal Depth (km)", "MMI Int"]
    for c in core:
        if c not in df.columns:
            raise ValueError(f"Missing required column for Result #2: {c}")

    
    candidates = [c for c in ["Latitude", "Longitude", "Year", "Tsu", "Vol"] if c in df.columns]

    feature_sets_to_try = [core]
    for c in candidates:
        feature_sets_to_try.append(core + [c])

    if "Latitude" in candidates and "Longitude" in candidates:
        feature_sets_to_try.append(core + ["Latitude", "Longitude"])

    if len(candidates) > 0:
        feature_sets_to_try.append(core + candidates)

    best_features = core[:]
    best_n = -1

    for feats in feature_sets_to_try:
        tmp = df.dropna(subset=feats + ["log_deaths", "MMI Int"]).copy()
        if len(tmp) > best_n:
            best_n = len(tmp)
            best_features = feats

    err_df = df.dropna(subset=best_features + ["log_deaths", "MMI Int"]).copy()

    if len(err_df) < min_rows_total:
        raise ValueError(
            f"Too few rows after dropping missing values even for best feature set.\n"
            f"Best features: {best_features}\n"
            f"N rows: {len(err_df)} (need ≥ {min_rows_total})"
        )

    # fixed threshold
    err_df["mmi_group"] = err_df["MMI Int"].apply(label_mmi_group)
    err_df = err_df.dropna(subset=["mmi_group"]).copy()
    counts = err_df["mmi_group"].value_counts()

    ok = (counts.shape[0] >= 2) and (counts.min() >= min_group_n)

    # median split 
    if (not ok) and use_quantile_fallback:
        q50 = err_df["MMI Int"].median()
        err_df["mmi_group"] = np.where(err_df["MMI Int"] >= q50, "High MMI (≥ median)", "Low MMI (< median)")
        counts = err_df["mmi_group"].value_counts()
        ok = (counts.shape[0] >= 2) and (counts.min() >= min_group_n)

    if not ok:
        raise ValueError(
            f"Not enough data in both MMI groups after filtering.\nCounts:\n{counts}\n"
            f"Try lowering min_group_n or adjust grouping."
        )

    X = err_df[best_features]
    y = err_df["log_deaths"]

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=random_state,
        stratify=err_df["mmi_group"]
    )

    pipe = make_pipe(best_features, alpha=10.0, random_state=random_state)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    abs_err = np.abs(y_test.values - y_pred)

    plot_df = pd.DataFrame({
        "mmi_group": err_df.loc[X_test.index, "mmi_group"].values,
        "abs_error": abs_err
    })

    test_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred))
    }

    meta = {
        "group_counts_used": counts.to_dict(),
        "features_used": best_features,
        "n_rows_used_total": int(len(err_df)),
        "n_test": int(len(X_test)),
        "grouping_mode": "fixed_IX_threshold" if ("MMI ≥ IX" in counts.index and "MMI ≤ VIII" in counts.index) else "median_fallback"
    }

    return plot_df, test_metrics, best_features, pipe, meta


def plot_main_result_2_error_by_mmi():
    plot_df = _MACHINE_OUT.get("err_plot_df")
    test_metrics = _MACHINE_OUT.get("test_metrics")
    meta = _MACHINE_OUT.get("error_meta")

    if plot_df is None or test_metrics is None or meta is None:
        raise ValueError("Missing stored results. Run run_all_machine_analyses() first.")

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=plot_df, x="mmi_group", y="abs_error")
    sns.stripplot(data=plot_df, x="mmi_group", y="abs_error", color="black", alpha=0.35)
    plt.title("Model absolute error (test set) by MMI group\n(Ridge, adaptive features)")
    plt.xlabel("")
    plt.ylabel("|prediction error| (log10 scale)")
    plt.tight_layout()
    plt.show()

    print("Result 2 meta:")
    print("  grouping mode:", meta["grouping_mode"])
    print("  features used:", meta["features_used"])
    print("  group counts:", meta["group_counts_used"])
    print("  rows used:", meta["n_rows_used_total"], "| test size:", meta["n_test"])
    print("\nAbs error by MMI group (test set):")
    print(plot_df.groupby("mmi_group")["abs_error"].agg(["mean", "median", "count"]))
    print("\nHeld-out test metrics:", test_metrics)


# learning curve
def plot_learning_curve_extra():
    df = _MACHINE_OUT.get("df")
    feats = _MACHINE_OUT.get("FULL_features")
    pipe = _MACHINE_OUT.get("full_pipe")

    if df is None or feats is None or pipe is None:
        raise ValueError("Run run_all_machine_analyses() first.")

    lc_df = df.dropna(subset=feats + ["log_deaths"]).copy()
    if len(lc_df) < 25:
        print("Not enough rows for learning curve.")
        return

    X_lc = lc_df[feats]
    y_lc = lc_df["log_deaths"]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator=pipe,
        X=X_lc, y=y_lc,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
    )

    train_rmse = -train_scores.mean(axis=1)
    val_rmse   = -val_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_rmse, marker="o", label="Train RMSE")
    plt.plot(train_sizes, val_rmse, marker="o", label="Validation RMSE")
    plt.title("Learning curve (Ridge, adaptive features)")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_all_machine_analyses(path="Earthquakes.csv", random_state=42):
    df = load_earthquakes(path)
    _MACHINE_OUT["df"] = df

    delta_df = compute_delta_rmse_by_tier(df, random_state=random_state)
    _MACHINE_OUT["delta_df"] = delta_df

    err_plot_df, test_metrics, feats, pipe, meta = compute_error_by_mmi(
        df,
        random_state=random_state,
        min_group_n=8,
        min_rows_total=25,
        use_quantile_fallback=True
    )

    _MACHINE_OUT["err_plot_df"] = err_plot_df
    _MACHINE_OUT["test_metrics"] = test_metrics
    _MACHINE_OUT["FULL_features"] = feats
    _MACHINE_OUT["full_pipe"] = pipe
    _MACHINE_OUT["error_meta"] = meta


def machine_outputs_keys():
    return list(_MACHINE_OUT.keys())




run_all_machine_analyses("Earthquakes.csv")
plot_main_result_1_delta_rmse()
plot_main_result_2_error_by_mmi()
plot_learning_curve_extra()
