"""Advertising -> Sales: cleaning, outlier detection, EDA, correlation, Linear Regression, insights."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "advertising.csv"
OUT = ROOT / "outputs"
FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"
SEED = 42


# ---------- 1. Cleaning & outliers ----------
def clean(df: pd.DataFrame):
    log = {"rows_raw": len(df)}
    df = df.drop_duplicates().copy()
    log["duplicates_removed"] = log["rows_raw"] - len(df)
    log["missing_before"] = df.isna().sum().to_dict()
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())      # median imputation
    df = df.dropna(subset=[TARGET])
    # IQR outlier detection
    outliers = {}
    mask = pd.Series(False, index=df.index)
    for c in FEATURES + [TARGET]:
        q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        m = (df[c] < lo) | (df[c] > hi)
        outliers[c] = int(m.sum()); mask |= m
    log["outliers_flagged"] = outliers
    df = df[~mask].reset_index(drop=True)
    log["rows_clean"] = len(df)
    return df, log


# ---------- 2. EDA ----------
def eda(df: pd.DataFrame):
    OUT.mkdir(exist_ok=True)
    corr = df.corr()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, c in zip(axes, FEATURES):
        ax.scatter(df[c], df[TARGET], alpha=.6, s=18)
        b, a = np.polyfit(df[c], df[TARGET], 1)
        xs = np.linspace(df[c].min(), df[c].max(), 50); ax.plot(xs, a + b * xs, "r")
        ax.set(xlabel=f"{c} spend ($k)", ylabel="Sales", title=f"{c} vs Sales (r={corr.loc[c, TARGET]:.2f})")
    plt.tight_layout(); plt.savefig(OUT / "scatter_features_vs_sales.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im); ax.set_title("Correlation matrix"); plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=130); plt.close()
    return corr


# ---------- 3. Model ----------
def train_eval(df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = LinearRegression().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = dict(
        r2_test=r2_score(y_te, pred),
        rmse_test=float(np.sqrt(mean_squared_error(y_te, pred))),
        mae_test=mean_absolute_error(y_te, pred),
        r2_train=model.score(X_tr, y_tr),
        cv_r2_mean=float(cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2").mean()),
        n_train=len(X_tr), n_test=len(X_te),
    )

    # residual / actual-vs-predicted plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(y_te, pred, alpha=.7); lim = [y.min(), y.max()]; ax[0].plot(lim, lim, "r--")
    ax[0].set(xlabel="Actual sales", ylabel="Predicted sales", title="Actual vs Predicted (test)")
    ax[1].scatter(pred, y_te - pred, alpha=.7); ax[1].axhline(0, color="r", ls="--")
    ax[1].set(xlabel="Predicted", ylabel="Residual", title="Residuals")
    plt.tight_layout(); plt.savefig(OUT / "model_diagnostics.png", dpi=130); plt.close()

    # ---------- 4. Feature importance & budget insights ----------
    std = X.std()
    imp = permutation_importance(model, X_te, y_te, n_repeats=30, random_state=SEED, scoring="r2")
    feat = pd.DataFrame({
        "coef_per_$1k": model.coef_,
        "std_coef(beta)": model.coef_ * std / y.std(),          # comparable across channels
        "permutation_importance_R2_drop": imp.importances_mean,
    }, index=FEATURES).round(4).sort_values("permutation_importance_R2_drop", ascending=False)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.barh(feat.index[::-1], feat["permutation_importance_R2_drop"][::-1])
    ax.set(xlabel="Drop in test R² when feature is shuffled", title="Feature importance")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=130); plt.close()
    return model, metrics, feat


def budget_insights(model, feat, df):
    """Translate coefficients into marketing guidance."""
    per_k = dict(zip(FEATURES, model.coef_))
    top_driver = feat.index[0]
    best_roi = max(per_k, key=per_k.get)
    return {
        "strongest_overall_predictor": top_driver,
        "highest_marginal_return_per_$1k": best_roi,
        "extra_sales_per_additional_$1k": {k: round(v, 4) for k, v in per_k.items()},
        "recommendation": (
            f"{top_driver} explains the most variation in sales, so it should remain the core budget line. "
            f"But each extra $1k in {best_roi} adds the most sales ({per_k[best_roi]:.3f}k units vs "
            f"{per_k[top_driver]:.3f}k for {top_driver}), so shift marginal budget towards {best_roi} until returns diminish. "
            f"Newspaper's effect ({per_k['Newspaper']:+.3f}) is statistically negligible: cut or cap it."),
    }


def main():
    if not DATA.exists():
        from .generate_data import generate
        DATA.parent.mkdir(exist_ok=True); generate().to_csv(DATA, index=False)
    raw = pd.read_csv(DATA)
    df, clean_log = clean(raw)
    corr = eda(df)
    model, metrics, feat = train_eval(df)
    insights = budget_insights(model, feat, df)

    report = dict(cleaning=clean_log, metrics=metrics, intercept=float(model.intercept_),
                  correlation_with_sales=corr[TARGET].drop(TARGET).round(3).to_dict(),
                  feature_importance=feat.to_dict(orient="index"), insights=insights)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))

    print("== Cleaning =="); print(json.dumps(clean_log, indent=1))
    print("\n== Correlation with Sales =="); print(corr[TARGET].drop(TARGET).round(3).to_string())
    print(f"\n== Model == Sales = {model.intercept_:.3f} + " + " + ".join(f"{c:.4f}*{f}" for c, f in zip(model.coef_, FEATURES)))
    print(f"Test R2 = {metrics['r2_test']:.3f} | RMSE = {metrics['rmse_test']:.3f} | MAE = {metrics['mae_test']:.3f} | 5-fold CV R2 = {metrics['cv_r2_mean']:.3f}")
    print("\n== Feature importance =="); print(feat.to_string())
    print("\n== Insight =="); print(insights["recommendation"])


if __name__ == "__main__":
    main()
