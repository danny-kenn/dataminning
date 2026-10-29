"""
Does Class Attendance Affect Academic Performance? Identifying What Drives Exam Scores
COSC391 Data Mining - Group: John Kamau, Lynn Bitok, David Kiamba

Dataset: Student Habits vs Academic Performance (Kaggle), 1,000 students, 16 columns.
Place student_habits_performance.csv in ./data/ then run:  python student_performance_analysis.py
Outputs (figures, results.json) are written to ./outputs/
"""
import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")

SEED = 42
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
R = {}   # everything we report is stored here and saved to results.json

# ------------------------------------------------------------ 1. Load and clean
df = pd.read_csv("data/student_habits_performance.csv")
R["raw_shape"] = list(df.shape)
R["missing_before"] = {k: int(v) for k, v in df.isna().sum().items() if v}
R["duplicates"] = int(df.duplicated().sum())
# 91 students have no parental_education_level: keep them by labelling the group "Not reported"
df["parental_education_level"] = df["parental_education_level"].fillna("Not reported")
df = df.drop(columns=["student_id"])          # an identifier, not a predictor
R["clean_shape"] = list(df.shape)
R["exam_score_summary"] = {k: round(float(v), 2) for k, v in df["exam_score"].describe().items()}

num_cols = ["age", "study_hours_per_day", "social_media_hours", "netflix_hours", "attendance_percentage",
            "sleep_hours", "exercise_frequency", "mental_health_rating"]
cat_cols = ["gender", "part_time_job", "diet_quality", "parental_education_level",
            "internet_quality", "extracurricular_participation"]
y = df["exam_score"]

# ------------------------------------------------------------ 2. Correlation with exam score
corr = []
for c in num_cols:
    pr, pp = stats.pearsonr(df[c], y)
    sr, sp = stats.spearmanr(df[c], y)
    corr.append({"feature": c, "pearson_r": round(pr, 3), "pearson_p": float(f"{pp:.3g}"),
                 "spearman_rho": round(sr, 3), "spearman_p": float(f"{sp:.3g}")})
corr.sort(key=lambda d: -abs(d["pearson_r"]))
R["correlations"] = corr

# Attendance on its own, and after controlling for every other numeric factor (partial correlation)
others = [c for c in num_cols if c != "attendance_percentage"]
Z = np.column_stack([np.ones(len(df)), df[others].values])
res_att = df["attendance_percentage"].values - Z @ np.linalg.lstsq(Z, df["attendance_percentage"].values, rcond=None)[0]
res_y = y.values - Z @ np.linalg.lstsq(Z, y.values, rcond=None)[0]
pr, pp = stats.pearsonr(res_att, res_y)
R["attendance_partial_corr"] = {"r": round(pr, 3), "p": float(f"{pp:.3g}")}
R["attendance_r_squared_alone"] = round(corr[[d["feature"] for d in corr].index("attendance_percentage")]["pearson_r"] ** 2, 4)

# Average score by attendance band
bands = pd.cut(df["attendance_percentage"], [0, 70, 80, 90, 100], labels=["<70%", "70-79%", "80-89%", "90-100%"])
band = df.groupby(bands, observed=True)["exam_score"].agg(["count", "mean"]).round(1)
R["score_by_attendance_band"] = {k: {"students": int(v["count"]), "mean_score": float(v["mean"])} for k, v in band.iterrows()}
# Is the difference between the highest and lowest attendance groups significant?
hi = df.loc[df.attendance_percentage >= 90, "exam_score"]; lo = df.loc[df.attendance_percentage < 70, "exam_score"]
t, tp = stats.ttest_ind(hi, lo, equal_var=False)
R["attendance_ttest_90plus_vs_below70"] = {"t": round(float(t), 2), "p": float(f"{tp:.3g}")}

# Categorical factors: mean score per group
cat_means = {}
for c in cat_cols:
    g = df.groupby(c)["exam_score"].mean().round(1)
    cat_means[c] = {k: float(v) for k, v in g.items()}
R["mean_score_by_category"] = cat_means

# ------------------------------------------------------------ 3. Models (regression, predict exam_score)
X = df.drop(columns="exam_score")
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
                         ("num", StandardScaler(), num_cols)])
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=SEED),
}
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=SEED)
R["split"] = {"train": len(Xtr), "test": len(Xte), "seed": SEED}
kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
fitted, table = {}, []
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("m", m)]).fit(Xtr, ytr)
    p = pipe.predict(Xte)
    cv = cross_val_score(Pipeline([("pre", pre), ("m", m)]), Xtr, ytr, cv=kf, scoring="r2")
    table.append({"model": name, "test_r2": round(r2_score(yte, p), 4),
                  "rmse": round(float(np.sqrt(mean_squared_error(yte, p))), 2),
                  "mae": round(mean_absolute_error(yte, p), 2),
                  "cv10_r2_mean": round(cv.mean(), 4), "cv10_r2_std": round(cv.std(), 4)})
    fitted[name] = pipe
R["model_comparison"] = table
best = max(table, key=lambda d: d["test_r2"])["model"]
R["best_model"] = best

# ------------------------------------------------------------ 4. What matters most?
# (a) Permutation importance on the test set (original columns, so one-hot groups stay together)
pi = permutation_importance(fitted[best], Xte, yte, n_repeats=30, random_state=SEED, scoring="r2")
imp = pd.DataFrame({"feature": Xte.columns, "importance": pi.importances_mean, "std": pi.importances_std}) \
        .sort_values("importance", ascending=False)
R["permutation_importance"] = [{"feature": r.feature, "r2_drop": round(r.importance, 4), "std": round(r.std, 4)}
                               for r in imp.itertuples()]

# (b) Does attendance add anything? Refit the best model WITHOUT attendance and compare test R2
no_att_cols = [c for c in num_cols if c != "attendance_percentage"]
pre2 = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
                          ("num", StandardScaler(), no_att_cols)])
m2 = Pipeline([("pre", pre2), ("m", models[best].__class__(**models[best].get_params()))]).fit(Xtr.drop(columns="attendance_percentage"), ytr)
r2_without = r2_score(yte, m2.predict(Xte.drop(columns="attendance_percentage")))
r2_with = [d for d in table if d["model"] == best][0]["test_r2"]
R["attendance_ablation"] = {"model": best, "r2_with": r2_with, "r2_without": round(r2_without, 4),
                            "change": round(r2_with - r2_without, 4)}

# (c) Linear model: effect of one extra unit on the score (numeric features, standardised = comparable)
lin = fitted["Linear Regression"]
names = lin.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lin.named_steps["m"].coef_, index=names)
R["linear_standardised_coefs_numeric"] = {n.replace("num__", ""): round(float(v), 3)
                                          for n, v in coef.items() if n.startswith("num__")}
# raw effect of 1 extra study hour / 1 point of mental-health rating / 1 point of attendance (unscaled fit)
raw = LinearRegression().fit(pd.get_dummies(X, columns=cat_cols, drop_first=True).astype(float), y)
rc = dict(zip(pd.get_dummies(X, columns=cat_cols, drop_first=True).columns, raw.coef_))
R["raw_effect_per_unit"] = {k: round(float(rc[k]), 3) for k in
    ["study_hours_per_day", "mental_health_rating", "attendance_percentage", "sleep_hours",
     "exercise_frequency", "social_media_hours", "netflix_hours"]}

# ------------------------------------------------------------ 5. Figures
plt.figure(figsize=(7, 4)); c = pd.DataFrame(corr).set_index("feature")["pearson_r"].sort_values()
c.plot(kind="barh", color=["#c0392b" if v < 0 else "#2e86c1" for v in c]); plt.title("Correlation with exam score (Pearson r)")
plt.tight_layout(); plt.savefig(f"{OUT}/correlations.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4)); plt.scatter(df.attendance_percentage, y, s=10, alpha=.5)
b, a = np.polyfit(df.attendance_percentage, y, 1); xs = np.linspace(56, 100, 50); plt.plot(xs, a + b * xs, "r")
plt.xlabel("Attendance (%)"); plt.ylabel("Exam score"); plt.title(f"Attendance vs exam score (r = {corr[[d['feature'] for d in corr].index('attendance_percentage')]['pearson_r']})")
plt.tight_layout(); plt.savefig(f"{OUT}/attendance_vs_score.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4)); plt.scatter(df.study_hours_per_day, y, s=10, alpha=.5, color="green")
b, a = np.polyfit(df.study_hours_per_day, y, 1); xs = np.linspace(0, 8.3, 50); plt.plot(xs, a + b * xs, "r")
plt.xlabel("Study hours per day"); plt.ylabel("Exam score"); plt.title("Study hours vs exam score")
plt.tight_layout(); plt.savefig(f"{OUT}/study_vs_score.png", dpi=150); plt.close()

top = imp.head(8).iloc[::-1]
plt.figure(figsize=(7, 4)); plt.barh(top.feature, top.importance, xerr=top["std"], color="#2e86c1")
plt.xlabel("Drop in test R2 when shuffled"); plt.title(f"Permutation importance ({best})")
plt.tight_layout(); plt.savefig(f"{OUT}/feature_importance.png", dpi=150); plt.close()

mc = pd.DataFrame(table).set_index("model")["test_r2"]
plt.figure(figsize=(7, 4)); mc.plot(kind="bar", color="#8e44ad"); plt.ylabel("Test R2"); plt.xticks(rotation=30, ha="right")
plt.title("Model comparison"); plt.tight_layout(); plt.savefig(f"{OUT}/model_comparison.png", dpi=150); plt.close()

json.dump(R, open(f"{OUT}/results.json", "w"), indent=2)
print(json.dumps(R, indent=2))
