"""Builds outputs/dashboard.html from data/student_habits_performance.csv + outputs/results.json
Run student_performance_analysis.py first, then:  python build_dashboard.py"""
import json, pandas as pd
R = json.load(open("outputs/results.json"))
df = pd.read_csv("data/student_habits_performance.csv")
num = ["study_hours_per_day","mental_health_rating","exercise_frequency","sleep_hours","social_media_hours",
       "netflix_hours","attendance_percentage"]
explore = {}
for c in num:
    q = pd.qcut(df[c], 4, duplicates="drop")
    g = df.groupby(q, observed=True)["exam_score"].mean().round(1)
    explore[c] = {"labels": [f"{i.left:g}–{i.right:g}" for i in g.index], "values": g.tolist()}
D = {"corr": R["correlations"], "bands": R["score_by_attendance_band"], "imp": R["permutation_importance"][:8],
     "models": R["model_comparison"], "explore": explore, "abl": R["attendance_ablation"],
     "partial": R["attendance_partial_corr"], "raw": R["raw_effect_per_unit"], "tt": R["attendance_ttest_90plus_vs_below70"], "cat": R["mean_score_by_category"], "n": R["clean_shape"][0]}
cols = []
for c in df.columns:
    role = "Identifier" if c == "student_id" else "Target" if c == "exam_score" else "Feature"
    if pd.api.types.is_numeric_dtype(df[c]):
        info = f"mean {df[c].mean():.1f} (range {df[c].min():g}–{df[c].max():g})"; typ = "Numeric"
    else:
        vc = df[c].value_counts(); info = f"{len(vc)} categories: " + ", ".join(str(k) for k in vc.index[:4]); typ = "Categorical"
    cols.append({"n": c, "t": typ, "r": role, "m": int(df[c].isna().sum()), "i": info})
h, e = pd.cut(df["exam_score"], list(range(10, 101, 10)), include_lowest=True, retbins=True)
D["cols"] = cols; D["hist"] = {"labels": [f"{int(a)}–{int(b)}" for a, b in zip(e[:-1], e[1:])], "values": h.value_counts(sort=False).tolist()}
D["dup"] = R["duplicates"]; D["miss"] = sum(R["missing_before"].values()); D["score"] = R["exam_score_summary"]
html = open("dashboard_template.html", encoding="utf-8").read().replace("__DATA__", json.dumps(D))
open("outputs/dashboard.html", "w", encoding="utf-8").write(html)
print("dashboard.html written", len(html), "bytes")
