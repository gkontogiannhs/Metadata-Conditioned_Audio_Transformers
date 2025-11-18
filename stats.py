import numpy as np, pandas as pd
from scipy.stats import shapiro, levene, f_oneway, kruskal, mannwhitneyu
import itertools, warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------
# Smart patient-wise stratified sampler (1 sample per patient)
# Prioritizes abnormal (minority) classes first
# ---------------------------------------------------------------
def stratified_patient_sampling_priority(df, label_col, seed=42):
    np.random.seed(seed)
    sampled = []
    patients_sampled = set()

    # Sort labels by ascending frequency → minority first
    for lbl in df[label_col].value_counts().sort_values().index:
        sub = df[df[label_col] == lbl]
        # sample one per patient, skipping already used
        sub = sub[~sub["PID"].isin(patients_sampled)]
        s = sub.groupby("PID", group_keys=False).apply(lambda x: x.sample(1, random_state=seed))
        sampled.append(s)
        patients_sampled.update(s["PID"].unique())

    return pd.concat(sampled).reset_index(drop=True)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def stratified_patient_sampling(df, label_col, seed=42):
    np.random.seed(seed)
    parts = []
    for lbl, g in df.groupby(label_col):
        s = g.groupby("PID", group_keys=False).apply(lambda x: x.sample(1, random_state=seed))
        parts.append(s)
    return pd.concat(parts).reset_index(drop=True)

def kw_epsilon_squared(H, k, N):
    return np.nan if (N - k) <= 0 else (H - k + 1) / (N - k)

def cliffs_delta(x, y):
    nx, ny = len(x), len(y)
    diff = np.subtract.outer(x, y)
    delta = (np.sum(diff > 0) - np.sum(diff < 0)) / (nx * ny)
    return delta

# ---------------------------------------------------------------
# Adaptive test selector
# ---------------------------------------------------------------
def cycle_duration_test(df_samp, label_col):
    labels = df_samp[label_col].unique()
    groups = [df_samp.loc[df_samp[label_col]==l, "CycleDur"] for l in labels]
    
    # Normality check for each group/class
    shapiro_p = [shapiro(g)[1] for g in groups]
    # print(shapiro_p)
    normal = all(p > 0.05 for p in shapiro_p)
    # Check if equal variances
    levene_p = levene(*groups)[1]
    equal_var = levene_p > 0.05
    k, N = len(groups), sum(len(g) for g in groups)

    if normal and equal_var:
        test_type = "ANOVA"
        stat, p = f_oneway(*groups)
        eps2 = np.nan
    elif normal and not equal_var:
        test_type = "Welch_ANOVA"
        stat, p = f_oneway(*groups)  # approximate
        eps2 = np.nan
    else:
        test_type = "Kruskal_Wallis"
        stat, p = kruskal(*groups)
        eps2 = kw_epsilon_squared(stat, k, N)

    return dict(test=test_type, stat=stat, p=p, eps2=eps2)

def common_language_effect(x, y):
    nx, ny = len(x), len(y)
    diff = np.subtract.outer(x, y)
    prob = np.sum(diff > 0) / (nx * ny)
    return prob

def hodges_lehmann(x, y):
    # Median of all pairwise differences (robust typical difference)
    return np.median(np.subtract.outer(x, y))

# ---------------------------------------------------------------
# Pairwise Mann–Whitney + δ + CLES + HL_diff
# ---------------------------------------------------------------
def pairwise_posthoc(df_samp, label_col):
    pairs = list(itertools.combinations(df_samp[label_col].unique(), 2))
    out = []
    for a, b in pairs:
        x = df_samp.loc[df_samp[label_col]==a, "CycleDur"].values
        y = df_samp.loc[df_samp[label_col]==b, "CycleDur"].values
        # Test for difference in distribution using Mann–Whitney U test
        # if p < 0.05, then we reject null hypothesis and we say distributions differ
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        # print(f"Pairwise test {a} vs {b}: p={p}")
        delta = cliffs_delta(x, y)
        cles = common_language_effect(x, y)
        hl_diff = hodges_lehmann(x, y)
        out.append({"pair": f"{a} vs {b}", "p": p, "cliff": delta, "cles": cles, "hl_diff": hl_diff})
    df_out = pd.DataFrame(out)
    df_out["p_adj"] = np.minimum(df_out["p"] * len(df_out), 1.0)
    return df_out

# ---------------------------------------------------------------
# Bootstrap routine (valid for 2c or 4c)
# ---------------------------------------------------------------
def bootstrap_cycle_duration(df, label_col="label_4c", n_iter=200, seed=42):
    np.random.seed(seed)
    main_res, posthoc_res = [], []
    all_ps = []

    for i in range(n_iter):
        # samp = stratified_patient_sampling(df, label_col, seed+i)
        samp = stratified_patient_sampling_priority(df, label_col, seed+i)
        main = cycle_duration_test(samp, label_col)
        all_ps.append(main["p"])
        post = pairwise_posthoc(samp, label_col)
        post["iter"] = i
        main_res.append(main)
        posthoc_res.append(post)

    df_main = pd.DataFrame(main_res)
    df_post = pd.concat(posthoc_res, ignore_index=True)

    # =============================
    # GLOBAL SUMMARY
    # =============================
    print("\n" + "="*90)
    print(f"CYCLE DURATION TEST — {label_col.upper()}".center(90))
    print("="*90)
    print(df_main["test"].value_counts(), "\n")

    mean_p = df_main.p.mean()
    frac_sig = (df_main.p < 0.05).mean()
    mean_eps2 = df_main.loc[df_main.test=="Kruskal_Wallis", "eps2"].mean()

    print(f"Mean p-value: {mean_p:.3f}")
    print(f"Fraction significant (p<0.05): {frac_sig:.2f}")
    print(f"Mean ε² (effect size): {mean_eps2:.4f}")
    print("→ ε² represents the proportion of total variance in cycle duration\n"
          "   explained by the label categories. Values <0.01 = negligible, 0.01–0.06 = small.\n")

    # =============================
    # PAIRWISE SUMMARY
    # =============================
    print("="*90)
    print("PAIRWISE POST-HOC RESULTS (Mann–Whitney + Effect Sizes)".center(90))
    print("="*90)

    summary = (
        df_post.groupby("pair")
        .agg(mean_p=("p_adj","mean"),
             frac_sig=("p_adj", lambda x:(x<0.05).mean()),
             mean_cliff=("cliff","mean"),
             std_cliff=("cliff","std"),
             mean_cles=("cles","mean"),
             std_cles=("cles","std"),
             mean_hl=("hl_diff","mean"),
             std_hl=("hl_diff","std"))
        .reset_index()
        .sort_values("frac_sig", ascending=False)
    )

    for _,r in summary.iterrows():
        strength = (
            "negligible" if abs(r["mean_cliff"]) < 0.147 else
            "small" if abs(r["mean_cliff"]) < 0.33 else
            "medium" if abs(r["mean_cliff"]) < 0.474 else "large"
        )
        print(f"{r['pair']:<18} | sig%={r['frac_sig']*100:5.1f}% | "
              f"δ={r['mean_cliff']:+.3f} ({strength}) | "
              f"CLES={r['mean_cles']:.3f}±{r['std_cles']:.3f} | "
              f"HL_diff={r['mean_hl']:+.3f}s ±{r['std_hl']:.3f}")

    print("\nInterpretation guide:")
    print(" • δ (Cliff’s):  direction & standardized magnitude of difference (−1 to 1)")
    print(" • CLES:         probability that a random sample from A > B (0.5 = no diff)")
    print(" • HL_diff:      median difference in seconds (real-world magnitude)")
    print(" • sig%:         stability of p<0.05 across bootstraps")

    return df_main, df_post, summary, all_ps


if __name__ == "__main__":
    # For 4-class problem
    df_main, df_post, summary, all_ps = bootstrap_cycle_duration(df, label_col="label_4c", n_iter=100)