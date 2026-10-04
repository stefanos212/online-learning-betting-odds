"""
GENERATES THE THESIS APPENDIX TABLES FROM results/*.csv.

The appendix of `docs/thesis.tex` reproduces every result table in full, rather
than the abridged versions the main chapters show. Writing those by hand would
guarantee they drift out of step with the pipeline, so they are generated:
this file reads the current CSVs and emits `docs/appendix_tables.tex`, which
the thesis \\input{}s.

With `--en` it instead emits `docs/appendix_tables_en.tex` for the English
edition `docs/thesis_en.tex`: English headers and captions, and English number
formatting (period decimal separator, comma thousands separator) rather than
the Greek convention. Everything else -- which CSVs, which columns, the fitting
logic -- is shared, so the two appendices cannot drift apart.

Rerun it after any pipeline rerun. It writes nothing except that one file.
"""

import os
import sys
import numpy as np
import pandas as pd

from sleeping_experts import OUT_DIR

OUT_TEX = os.path.join("..", "docs", "appendix_tables.tex")
OUT_TEX_EN = os.path.join("..", "docs", "appendix_tables_en.tex")

# Greek column headers, so the tables read as part of the text rather than as
# a dump of the CSV schema.
HEAD_EL = {
    "name": "Σειρά", "type": "Τύπος", "phase": "Φάση", "panel": "Panel",
    "coverage_pct": "Κάλυψη \\%", "coverage_test_pct": "Κάλυψη \\%",
    "coverage_tail_pct": "Κάλυψη \\%", "rounds": "Γύροι", "n": "$n$",
    "raw_log_loss": "log-loss (raw)", "calibrated_log_loss": "log-loss (cal.)",
    "raw_brier": "Brier (raw)", "calibrated_brier": "Brier (cal.)",
    "raw_accuracy": "Acc. (raw)", "calibrated_accuracy": "Acc.",
    "train_log_loss": "train", "val_log_loss": "ουρά", "held_out": "ουρά",
    "log_loss": "log-loss", "brier": "Brier", "rps": "RPS", "accuracy": "Acc.",
    "mean_diff": "Μ. διαφορά", "ci_low": "CI κάτω", "ci_high": "CI άνω",
    "p_value": "$p$", "significant_95": "Σημ.", "verdict": "Έκβαση",
    "comparison": "Σύγκριση", "bookmaker": "Bookmaker", "config": "Διαμόρφωση",
    "vs": "Έναντι", "stage": "Στάδιο", "algorithm": "Αλγόριθμος",
    "market": "Panel", "normalization": "De-vig", "metric": "Μετρική",
    "alpha": "$\\alpha$", "season": "Σεζόν", "value": "Τιμή", "series": "Σειρά",
    "league": "Λίγκα", "league_name": "Πρωτάθλημα", "n_matches": "Αγώνες",
    "global_log_loss": "Ενιαίο", "specialist_log_loss": "Ειδικός",
    "specialist_wins": "Νίκη ειδ.", "outcome": "Έκβαση",
    "n_bets": "Στοιχ.", "final_bankroll": "Κεφάλαιο", "target_phase": "Φάση",
    "panel_type": "Panel εκπ.", "mean_log_growth_per_bet": "log-growth",
    "leakage_risk": "Διαρροή", "variant": "Παραλλαγή", "aside": "Χωριστά",
    "block_length": "Μπλοκ", "best_lr_scale": "$c$", "common_rounds": "Κοινοί",
    "bet_rate_pct": "Ρυθμός \\%", "active_rounds": "Ενεργοί",
}

HEAD_EN = {
    "name": "Series", "type": "Type", "phase": "Phase", "panel": "Panel",
    "coverage_pct": "Coverage \\%", "coverage_test_pct": "Coverage \\%",
    "coverage_tail_pct": "Coverage \\%", "rounds": "Rounds", "n": "$n$",
    "raw_log_loss": "log-loss (raw)", "calibrated_log_loss": "log-loss (cal.)",
    "raw_brier": "Brier (raw)", "calibrated_brier": "Brier (cal.)",
    "raw_accuracy": "Acc. (raw)", "calibrated_accuracy": "Acc.",
    "train_log_loss": "train", "val_log_loss": "tail", "held_out": "tail",
    "log_loss": "log-loss", "brier": "Brier", "rps": "RPS", "accuracy": "Acc.",
    "mean_diff": "Mean diff.", "ci_low": "CI low", "ci_high": "CI high",
    "p_value": "$p$", "significant_95": "Sig.", "verdict": "Verdict",
    "comparison": "Comparison", "bookmaker": "Bookmaker", "config": "Config.",
    "vs": "Against", "stage": "Stage", "algorithm": "Algorithm",
    "market": "Panel", "normalization": "De-vig", "metric": "Metric",
    "alpha": "$\\alpha$", "season": "Season", "value": "Value", "series": "Series",
    "league": "League", "league_name": "League", "n_matches": "Matches",
    "global_log_loss": "Unified", "specialist_log_loss": "Specialist",
    "specialist_wins": "Spec. wins", "outcome": "Outcome",
    "n_bets": "Bets", "final_bankroll": "Bankroll", "target_phase": "Phase",
    "panel_type": "Train panel", "mean_log_growth_per_bet": "log-growth",
    "leakage_risk": "Leakage", "variant": "Variant", "aside": "Aside",
    "block_length": "Block", "best_lr_scale": "$c$", "common_rounds": "Common",
    "bet_rate_pct": "Rate \\%", "active_rounds": "Active",
}

# Set by main(); everything below reads these rather than taking a language
# argument through five call sites.
HEAD = HEAD_EL
EN = False


def esc(x):
    """LaTeX-safe cell, with numbers rounded to a readable width."""
    if isinstance(x, (bool, np.bool_)):
        if EN:
            return "yes" if x else "no"
        return "ναι" if x else "όχι"
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "---"
        if abs(x) >= 1000:
            s = f"{x:,.0f}"
            return s if EN else s.replace(",", ".")
        s = f"{x:.4f}" if abs(x) >= 1 else f"{x:.5f}"
        return s if EN else s.replace(".", ",")
    if isinstance(x, (int, np.integer)):
        s = f"{x:,}"
        return s if EN else s.replace(",", ".")
    s = str(x)
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


# Columns carried by several CSVs that add width without adding information in
# a printed appendix: the verdict string already states what significance says,
# and the block length is an implementation detail of the bootstrap.
DROP_ALWAYS = ("significant_95", "block_length", "bet_rate_pct", "active_rounds",
               "common_rounds", "leakage_risk", "aside")


def _width(df):
    """Rough printed width in characters, used to pick a font size."""
    total = 0
    for c in df.columns:
        cells = [len(esc(v)) for v in df[c].values] or [0]
        total += max(max(cells), len(HEAD.get(c, c))) + 2
    return total


def longtable(df, caption, label, cols=None):
    """A CSV as a page-breaking longtable, sized to fit the text block."""
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    df = df[[c for c in df.columns if c not in DROP_ALWAYS]]
    # long free-text cells (verdict strings, configuration names) dominate the
    # width without carrying proportional information; clip them
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.slice(0, 20)
    # keep dropping the widest remaining column while the table cannot fit
    while _width(df) > 106 and len(df.columns) > 4:
        widest = max(df.columns,
                     key=lambda c: max([len(esc(v)) for v in df[c].values] or [0]))
        df = df.drop(columns=[widest])
    size = r"\small" if _width(df) <= 72 else (
        r"\footnotesize" if _width(df) <= 92 else r"\scriptsize")
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) and
                    not pd.api.types.is_bool_dtype(df[c]) else "l"
                    for c in df.columns)
    head = " & ".join(f"\\textbf{{{HEAD.get(c, esc(c))}}}" for c in df.columns) + r" \\"
    out = ["{" + size]
    out.append(r"\begin{longtable}{@{}" + align + r"@{}}")
    out.append(f"\\caption{{{caption}}}\\label{{{label}}}\\\\")
    out.append(r"\toprule " + head + r"\midrule\endfirsthead")
    cont = "(continued)" if EN else "(συνέχεια)"
    out.append(r"\multicolumn{" + str(len(df.columns)) +
               r"}{l}{\emph{" + cont + r"}}\\\toprule " + head + r"\midrule\endhead")
    out.append(r"\bottomrule\endfoot")
    for _, r in df.iterrows():
        out.append(" & ".join(esc(v) for v in r.values) + r" \\")
    out.append(r"\end{longtable}")
    out.append("}")
    return "\n".join(out) + "\n\n"


def read(name):
    p = os.path.join(OUT_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def T(el, en):
    """Pick the Greek or English string for the edition being generated."""
    return en if EN else el


PREAMBLE_EL = r"""% ΠΑΡΑΓΟΜΕΝΟ ΑΡΧΕΙΟ -- μην το επεξεργάζεστε με το χέρι.
% Δημιουργείται από scripts/make_appendix_tables.py με βάση τα results/*.csv.

\chapter{Πλήρεις πίνακες αποτελεσμάτων}
\label{app:tables}

Το παράρτημα αυτό παραθέτει σε πλήρη μορφή τους πίνακες που τα κεφάλαια του κυρίως κειμένου
παρουσιάζουν συνοπτικά. Παράγεται αυτόματα από τα αρχεία αποτελεσμάτων, ώστε να μην
αποκλίνει από αυτά.

\section{Κατάταξη και στατιστική σημαντικότητα}
"""

PREAMBLE_EN = r"""% GENERATED FILE -- do not edit by hand.
% Produced by scripts/make_appendix_tables.py --en from results/*.csv.

\chapter{Full result tables}
\label{app:tables}

This appendix gives in full the tables that the chapters of the main text present in abridged
form. It is generated automatically from the result files, so that it cannot drift away from
them.

\section{Ranking and statistical significance}
"""


def main():
    parts = []
    add = parts.append

    add(T(PREAMBLE_EL, PREAMBLE_EN))

    t = read("final_ranking_table.csv")
    if t is not None:
        add(longtable(t, T("Πλήρης κατάταξη Tier-A, δέκα σεζόν.",
                           "Full Tier-A ranking, ten seasons."), "app:ranking"))

    t = read("results_table.csv")
    if t is not None:
        add(longtable(t, T("Όλες οι σειρές του panel, χωρίς κατώφλι κάλυψης.",
                           "Every series in the panel, with no coverage threshold."),
                      "app:allseries"))

    t = read("significance_test_table.csv")
    if t is not None:
        add(longtable(t, T("Πλήρης μπαταρία ελέγχων σημαντικότητας.",
                           "The full battery of significance tests."), "app:sig"))

    add(T(r"\section{Panel και μέθοδος de-vig}", r"\section{Panel and de-vig method}") + "\n")
    for f, cap, cap_en in (("opening_vs_closing_table.csv",
                            "Αναλογική κανονικοποίηση", "proportional normalisation"),
                           ("opening_vs_closing_table_shin.csv",
                            "Κανονικοποίηση Shin", "Shin normalisation")):
        t = read(f)
        if t is not None:
            add(longtable(t, T(f"Panel $\\times$ αλγόριθμος --- {cap}.",
                               f"Panel $\\times$ algorithm --- {cap_en}."),
                          "app:" + f[:12] + cap[:4]))
    for f, cap, cap_en in (("opening_vs_closing_vs_uniform.csv",
                            "αναλογική", "proportional"),
                           ("opening_vs_closing_vs_uniform_shin.csv", "Shin", "Shin")):
        t = read(f)
        if t is not None:
            add(longtable(t, T(f"Κάθε αλγόριθμος έναντι του απλού μέσου όρου, {cap}.",
                               f"Each algorithm against the plain average, {cap_en}."),
                          "app:vsuni" + cap[:4]))

    add(T(r"\section{Διαμορφώσεις}", r"\section{Configurations}") + "\n")
    for f, cap, cap_en, lab in (
            ("best_configuration.csv", "Στοίβαξη διαμορφώσεων",
             "Stacking of configurations", "app:cfg"),
            ("best_configuration_significance.csv", "Ζευγαρωτοί έλεγχοι διαμορφώσεων",
             "Paired tests of configurations", "app:cfgsig")):
        t = read(f)
        if t is not None:
            add(longtable(t, T(cap, cap_en) + ".", lab))

    add(T(r"\section{Χωρίς την Pinnacle}", r"\section{Without Pinnacle}") + "\n")
    for f, cap, cap_en, lab in (
            ("no_pinnacle_ranking.csv", "Κατάταξη χωρίς Pinnacle, τρία panel",
             "Ranking without Pinnacle, three panels", "app:np"),
            ("no_pinnacle_significance.csv", "Ζευγαρωτοί έλεγχοι χωρίς Pinnacle",
             "Paired tests without Pinnacle", "app:npsig")):
        t = read(f)
        if t is not None:
            add(longtable(t, T(cap, cap_en) + ".", lab))

    add(T(r"\section{Πρόσφατο παράθυρο και έλεγχος ανάπτυξης}",
          r"\section{Recent window and deployment test}") + "\n")
    for f, cap, cap_en, lab in (
            ("recent_window_ranking.csv", "Κατάταξη, πέντε σεζόν",
             "Ranking, five seasons", "app:rw"),
            ("recent_window_significance.csv", "Έλεγχοι σημαντικότητας, πέντε σεζόν",
             "Significance tests, five seasons", "app:rwsig"),
            ("recent_window_live_market.csv", "Η αγορά της ουράς, πέντε σεζόν",
             "The market in the tail, five seasons", "app:rwlive"),
            ("deployment_test.csv", "Έλεγχος ανάπτυξης 2025--26",
             "Deployment test 2025--26", "app:dep"),
            ("deployment_test_significance.csv", "Ζευγαρωτοί έλεγχοι, ανάπτυξη 2025--26",
             "Paired tests, deployment 2025--26", "app:depsig")):
        t = read(f)
        if t is not None:
            add(longtable(t, T(cap, cap_en) + ".", lab))

    add(r"\section{Value betting}" + "\n")
    t = read("value_betting_training_panel_sweep.csv")
    if t is not None:
        add(longtable(t, T("Πλήρης σάρωση panel εκπαίδευσης $\\times$ στόχου "
                           "$\\times$ κανονικοποίησης.",
                           "Full sweep of training panel $\\times$ target "
                           "$\\times$ normalisation."), "app:vbsweep"))
    for f, cap, cap_en, lab in (
            ("value_betting_all_bookmakers_table.csv", "Σάρωση όλων των bookmakers",
             "Sweep over all bookmakers", "app:vball"),
            ("recent_window_value_betting.csv", "Value betting, πέντε σεζόν",
             "Value betting, five seasons", "app:vbrw")):
        t = read(f)
        if t is not None:
            add(longtable(t, T(cap, cap_en) + ".", lab))

    add(T(r"\section{Επεκτάσεις και αρνητικά αποτελέσματα}",
          r"\section{Extensions and negative results}") + "\n")
    for f, cap, cap_en, lab in (
            ("fixed_share_table.csv", "Fixed-Share ανά $\\alpha$",
             "Fixed-Share by $\\alpha$", "app:fs"),
            ("fixed_share_weights.csv",
             "Βάρη ανά expert και $\\alpha$ (τελευταία ξύπνια στιγμή)",
             "Weights by expert and $\\alpha$ (at each expert's last awake round)", "app:fsw"),
            ("contextual_experts_table.csv", "Εξειδίκευση ανά πρωτάθλημα",
             "Per-league specialisation", "app:ctx"),
            ("outcome_class_accuracy_table.csv", "Επίδοση ανά κατηγορία αποτελέσματος",
             "Performance by outcome class", "app:cls"),
            ("calibration_per_season_significance.csv", "Βαθμονόμηση ανά σεζόν",
             "Calibration per season", "app:calseason"),
            ("calibration_warmstart_significance.csv", "Βαθμονόμηση με warm start",
             "Calibration with a warm start", "app:calwarm"),
            ("temporal_trends_mk_test.csv", "Έλεγχοι τάσης Mann--Kendall",
             "Mann--Kendall trend tests", "app:mk"),
            ("learning_rate_study.csv", "Σάρωση συντελεστή βήματος",
             "Step-multiplier sweep", "app:lr"),
            ("pooling_window_ranking_table.csv", "Σύγκριση χρονικών παραθύρων",
             "Comparison of time windows", "app:pool"),
            ("information_arrival_significance.csv", "Άφιξη πληροφορίας ανά πεμπτημόριο",
             "Information arrival by quintile", "app:info")):
        t = read(f)
        if t is not None:
            add(longtable(t, T(cap, cap_en) + ".", lab))

    out_tex = OUT_TEX_EN if EN else OUT_TEX
    with open(out_tex, "w", encoding="utf-8", newline="") as fh:
        fh.write("".join(parts))

    n_tables = sum(p.count(r"\begin{longtable}") for p in parts)
    print(f"Saved: {out_tex}  ({n_tables} {T('πίνακες', 'tables')})")


if __name__ == "__main__":
    EN = "--en" in sys.argv[1:]
    HEAD = HEAD_EN if EN else HEAD_EL
    main()
