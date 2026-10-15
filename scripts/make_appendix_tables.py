"""
GENERATES THE THESIS APPENDIX TABLES FROM results/*.csv.

The appendix of the thesis reproduces the result tables in full, rather than
the abridged versions the main chapters show. Writing those by hand would
guarantee they drift out of step with the pipeline, so they are generated:
this file reads the current CSVs and emits `docs/appendix_tables_en.tex`
(with `--en`) or the Greek `docs/appendix_tables.tex` (without it), which the
thesis \\input{}s. The two editions differ only in headers, captions, the
lead-in of each section and number formatting (period vs. comma decimal
separator); which CSVs and which columns are shown is shared, so the two
appendices cannot drift apart.

Every table names the columns it shows. An earlier version let a width
heuristic drop the widest columns until a table fit, which on most CSVs meant
dropping the row labels themselves (series names, configurations, panels):
the numbers survived and nothing said what they belonged to. The heuristic
is kept only as a fallback for a table given no column list, and it can no
longer drop the first column.

Rerun it after any pipeline rerun. It writes nothing except that one file.
"""

import os
import sys
import numpy as np
import pandas as pd

from sleeping_experts import OUT_DIR

OUT_TEX = os.path.join("..", "docs", "appendix_tables.tex")
OUT_TEX_EN = os.path.join("..", "docs", "appendix_tables_en.tex")

# Column headers, so the tables read as part of the text rather than as a dump
# of the CSV schema.
HEAD_EL = {
    "name": "Σειρά", "type": "Τύπος", "phase": "Φάση", "panel": "Panel",
    "coverage_pct": "Κάλυψη \\%", "coverage_test_pct": "Κάλυψη \\%",
    "coverage_tail_pct": "Κάλυψη \\%", "rounds": "Γύροι", "n": "$n$",
    "raw_log_loss": "log-loss (raw)", "calibrated_log_loss": "log-loss (cal.)",
    "raw_brier": "Brier (raw)", "calibrated_brier": "Brier (cal.)",
    "raw_accuracy": "Acc. (raw)", "calibrated_accuracy": "Acc.",
    "full_log_loss": "πλήρες", "train_log_loss": "train", "val_log_loss": "ουρά",
    "held_out": "ουρά",
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
    "panel_type": "Panel εκπ.", "training_panel": "Panel εκπ.",
    "mean_log_growth_per_bet": "log-growth",
    "leakage_risk": "Διαρροή", "variant": "Παραλλαγή", "aside": "Χωριστά",
    "block_length": "Μπλοκ", "best_lr_scale": "$c$", "common_rounds": "Κοινοί",
    "bet_rate_pct": "Ρυθμός \\%", "active_rounds": "Ενεργοί",
    "n_bookmakers": "Experts", "n_seasons_active": "Σεζόν",
    "dataset": "Σύνολο", "step_multiplier": "Βήμα", "tail_log_loss": "ουρά",
    "tail_mean_diff": "Μ. διαφ. ουράς", "tail_p_value": "$p$ ουράς",
    "single_season": "Μία σεζόν", "hedge_weight_last_awake": "Βάρος Hedge",
    "rank": "\\#", "family": "Οικογένεια", "setting": "Ρύθμιση",
    "stratifier": "Διαστρωμάτωση", "stratum": "Πεμπτ.", "window": "Παράθυρο",
    "trend": "Τάση", "top3_trusted_bookmakers": "Πιο έμπιστοι", "S": "$S$", "Z": "$Z$",
}

HEAD_EN = {
    "name": "Series", "type": "Type", "phase": "Phase", "panel": "Panel",
    "coverage_pct": "Coverage \\%", "coverage_test_pct": "Coverage \\%",
    "coverage_tail_pct": "Coverage \\%", "rounds": "Rounds", "n": "$n$",
    "raw_log_loss": "log-loss (raw)", "calibrated_log_loss": "log-loss (cal.)",
    "raw_brier": "Brier (raw)", "calibrated_brier": "Brier (cal.)",
    "raw_accuracy": "Acc. (raw)", "calibrated_accuracy": "Acc.",
    "full_log_loss": "full", "train_log_loss": "train", "val_log_loss": "tail",
    "held_out": "tail",
    "log_loss": "log-loss", "brier": "Brier", "rps": "RPS", "accuracy": "Acc.",
    "mean_diff": "Mean diff.", "ci_low": "CI low", "ci_high": "CI high",
    "p_value": "$p$", "significant_95": "Sig.", "verdict": "Verdict",
    "comparison": "Comparison", "bookmaker": "Bookmaker", "config": "Configuration",
    "vs": "Against", "stage": "Stage", "algorithm": "Algorithm",
    "market": "Panel", "normalization": "De-vig", "metric": "Metric",
    "alpha": "$\\alpha$", "season": "Season", "value": "Value", "series": "Series",
    "league": "League", "league_name": "League", "n_matches": "Matches",
    "global_log_loss": "Unified", "specialist_log_loss": "Specialist",
    "specialist_wins": "Spec. wins", "outcome": "Outcome",
    "n_bets": "Bets", "final_bankroll": "Bankroll", "target_phase": "Phase",
    "panel_type": "Train panel", "training_panel": "Train panel",
    "mean_log_growth_per_bet": "log-growth",
    "leakage_risk": "Leakage", "variant": "Variant", "aside": "Aside",
    "block_length": "Block", "best_lr_scale": "$c$", "common_rounds": "Common",
    "bet_rate_pct": "Rate \\%", "active_rounds": "Active",
    "n_bookmakers": "Experts", "n_seasons_active": "Seasons",
    "dataset": "Dataset", "step_multiplier": "Step", "tail_log_loss": "tail",
    "tail_mean_diff": "Tail diff.", "tail_p_value": "Tail $p$",
    "single_season": "Single season", "hedge_weight_last_awake": "Hedge weight",
    "rank": "\\#", "family": "Family", "setting": "Setting",
    "stratifier": "Stratified by", "stratum": "Quintile", "window": "Window",
    "trend": "Trend", "top3_trusted_bookmakers": "Most trusted", "S": "$S$", "Z": "$Z$",
}

# Whole-cell display values, per column: raw CSV value -> (Greek, English).
# Several scripts store Greek labels or internal shorthands in their CSVs,
# which other scripts read back, so they are translated here, on display.
_CONFIG = {
    "PSC alone": ("μόνο PSC", "PSC alone"),
    "A: closing + Shin": ("closing-only + Shin", "closing-only + Shin"),
    "B: A minus single-season": ("closing-only + Shin, χωρίς μίας σεζόν",
                                 "closing-only + Shin, no single-season"),
    "closing only, basic": ("closing-only, αναλογική", "closing-only, proportional"),
    "baseline (full 26, basic)": ("πλήρες panel, αναλογική", "full panel, proportional"),
    "full 26, Shin": ("πλήρες panel, Shin", "full panel, Shin"),
}
_PANEL = {
    "full panel (24 experts)": ("πλήρες (24)", "full (24)"),
    "closing-only panel (12 experts)": ("closing (12)", "closing (12)"),
    "opening-only panel (12 experts)": ("opening (12)", "opening (12)"),
}
_VERDICT = {
    "tie (not significant)": ("ισοπαλία", "tie"), "ισοπαλία": ("ισοπαλία", "tie"),
    "αλγόριθμος": ("νίκη μείγματος", "mixture better"),
    "BFEC": ("νίκη BFEC", "BFEC better"),
}
_VERDICT.update({f"{k} better": (f"νίκη {v[0]}", f"{v[1]} better")
                 for k, v in _CONFIG.items()})
VALUES = {
    "config": _CONFIG, "vs": _CONFIG, "verdict": _VERDICT,
    "panel": {**_PANEL, "B'": ("B$'$", "B$'$")}, "dataset": {"B'": ("B$'$", "B$'$")},
    "stage": {"raw": ("raw", "raw"), "calibrated": ("βαθμ.", "cal.")},
    "market": {"full": ("πλήρες", "full")},
    "metric": {"log_loss": ("log-loss", "log-loss"), "brier": ("Brier", "Brier"),
               "accuracy": ("ακρίβεια", "accuracy"), "ece": ("ECE", "ECE"),
               "overround_pct": ("περιθώριο", "overround")},
    "normalization": {"Basic": ("αναλογική", "basic")},
    "panel_type": {"closing-trained": ("closing", "closing"),
                   "mixed-trained": ("πλήρες", "full"),
                   "opening-trained": ("opening", "opening")},
    "training_panel": {"πλήρες panel": ("πλήρες", "full"),
                       "ταιριασμένο": ("ταιριασμένο", "matched")},
    "window": {"10yr": ("10 σεζόν", "10 seasons"), "5yr": ("5 σεζόν", "5 seasons")},
    "stratifier": {"line movement": ("κίνηση γραμμής", "line movement"),
                   "Shin z anomaly": ("ανωμαλία $z$ του Shin", "Shin $z$ anomaly")},
    "comparison": {"warm-start vs per-season (cold-start reset)":
                   ("warm-start vs ανά σεζόν", "warm-start vs per-season")},
    "trend": {"decreasing": ("πτωτική", "decreasing"), "increasing": ("ανοδική", "increasing")},
    "type": {"αλγόριθμος": ("μείγμα", "mixture")},
    "family": {
        "Hedge x mult": ("Hedge, πολλαπλασιαστής", "Hedge, multiplier"),
        "OGD x mult": ("OGD, πολλαπλασιαστής", "OGD, multiplier"),
        "FTRL x mult": ("FTRL, πολλαπλασιαστής", "FTRL, multiplier"),
        "Hedge const eta": ("Hedge, σταθερό $\\eta$", "Hedge, constant $\\eta$"),
        "OGD const eta": ("OGD, σταθερό $\\eta$", "OGD, constant $\\eta$"),
        "FTRL const eta": ("FTRL, σταθερό $\\eta$", "FTRL, constant $\\eta$"),
        "OGD adaptive": ("OGD, AdaGrad", "OGD, AdaGrad"),
        "OGD honest M": ("OGD, φράγμα $M$", "OGD, loss bound $M$"),
        "shipped": ("σε ισχύ", "in force"),
    },
}

# Set by main(); everything below reads these rather than taking a language
# argument through every call site.
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
    if EN:
        # a few scripts store Greek labels as CSV values; translate for display
        # here rather than in the CSVs, which other scripts read back
        for a, b in (("πλήρες panel", "full panel"), ("παγωμένο", "frozen"),
                     ("αλγόριθμος", "algorithm"), ("ισοπαλία", "tie"),
                     ("ταιριασμένο", "matched")):
            s = s.replace(a, b)
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def _bankroll(x):
    """Final bankrolls span 0.0002 to 29,342: digits where they mean something."""
    if x >= 100:
        return f"{x:,.0f}"
    return f"{x:.2f}" if x >= 0.1 else f"{x:.4f}"


def _setting(x):
    """A step setting: plain for ordinary magnitudes, a power of ten otherwise."""
    if x == 0 or 1e-3 <= abs(x) < 1e4:
        return f"{x:g}"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"$10^{{{e}}}$" if abs(m - 1) < 1e-9 else f"${m:g}\\cdot10^{{{e}}}$"


def _p(p):
    return "$<$0.001" if p < 0.0005 else f"{p:.3f}"


# Per-column number formats that esc() cannot infer from the value alone.
_LOSS = ("raw_log_loss", "calibrated_log_loss", "log_loss", "raw_brier",
         "calibrated_brier", "brier", "accuracy", "calibrated_accuracy",
         "full_log_loss", "train_log_loss", "val_log_loss", "global_log_loss",
         "specialist_log_loss")
FORMAT = {c: "{:.5f}" for c in _LOSS}
FORMAT.update({
    "coverage_pct": "{:.1f}", "coverage_test_pct": "{:.1f}", "coverage_tail_pct": "{:.1f}",
    "rank": "{:.0f}", "n": "{:,.0f}", "n_bets": "{:,.0f}", "rounds": "{:,.0f}",
    "n_matches": "{:,.0f}", "S": "{:.0f}", "Z": "{:.2f}", "best_lr_scale": "{:g}",
    "alpha": "{:g}", "step_multiplier": "{:g}", "tail_log_loss": "{:.5f}",
    "tail_mean_diff": "{:+.5f}", "tail_p_value": _p,
    "hedge_weight_last_awake": "{:.3f}", "final_bankroll": _bankroll,
    "setting": _setting, "p_value": _p, "mean_diff": "{:+.5f}",
    "mean_log_growth_per_bet": "{:+.5f}", "stratum": lambda s: f"Q{int(s) + 1}",
})


def cell(col, v):
    """One formatted, LaTeX-safe cell of column `col`."""
    if isinstance(v, (str, bool, np.bool_)) or v is None or (isinstance(v, float) and np.isnan(v)):
        return esc(v)
    f = FORMAT.get(col)
    if f is None:
        s = esc(v)
    else:
        s = f(v) if callable(f) else f.format(v)
        if not EN and "$" not in s:
            s = s.replace(",", "\0").replace(".", ",").replace("\0", ".")
    # a typographic minus, not a hyphen
    if s.startswith("-"):
        s = "$-$" + s[1:]
    elif s.startswith("+"):
        s = "$+$" + s[1:]
    return s


def display(df):
    """Apply the whole-cell display values of VALUES, column by column."""
    df = df.copy()
    k = 1 if EN else 0
    for c in df.columns:
        if c in VALUES:
            m = VALUES[c]
            df[c] = [m[v][k] if isinstance(v, str) and v in m else v for v in df[c]]
    return df


# Columns carried by several CSVs that add width without adding information in
# a printed appendix: the verdict string already states what significance says,
# and the block length is an implementation detail of the bootstrap.
DROP_ALWAYS = ("significant_95", "block_length", "bet_rate_pct", "active_rounds",
               "common_rounds", "leakage_risk", "aside")


def _width(df):
    """Rough printed width in characters, used to pick a font size."""
    total = 0
    for c in df.columns:
        cells = [len(cell(c, v)) for v in df[c].values] or [0]
        total += max(max(cells), len(HEAD.get(c, c))) + 2
    return total


def longtable(df, caption, label, cols=None):
    """A CSV as a page-breaking longtable, sized to fit the text block.

    With `cols` the table shows exactly those columns, in that order. Without
    it, the widest columns are dropped until the table fits -- never the
    first one, which is what tells the reader what a row is."""
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    else:
        df = df[[c for c in df.columns if c not in DROP_ALWAYS]].copy()
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype(str).str.slice(0, 20)
        while _width(df) > 106 and len(df.columns) > 4:
            widest = max(df.columns[1:],
                         key=lambda c: max([len(cell(c, v)) for v in df[c].values] or [0]))
            df = df.drop(columns=[widest])
    df = display(df)
    w = _width(df)
    size = r"\small" if w <= 72 else (r"\footnotesize" if w <= 92 else r"\scriptsize")
    if w > 100:
        size += r"\setlength{\tabcolsep}{4pt}"
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
        out.append(" & ".join(cell(c, v) for c, v in zip(df.columns, r.values)) + r" \\")
    out.append(r"\end{longtable}")
    out.append("}")
    return "\n".join(out) + "\n\n"


def read(name):
    p = os.path.join(OUT_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def count(n):
    """An integer count in the edition's thousands separator."""
    s = f"{int(n):,}"
    return s if EN else s.replace(",", ".")


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

Σε κάθε πίνακα ζευγαρωτών ελέγχων, η \textbf{Μ. διαφορά} είναι η μέση ανά γύρο διαφορά απώλειας
της σειράς που κατονομάζεται πρώτη μείον της δεύτερης, οπότε αρνητική τιμή σημαίνει ότι η πρώτη
είναι καλύτερη· τα \textbf{CI κάτω} και \textbf{CI άνω} είναι τα όρια του 95\% διαστήματος
εμπιστοσύνης του moving block bootstrap, $n$ το πλήθος των κοινών γύρων και $p$ η τιμή $p$ του
ελέγχου.

\section{Κατάταξη και στατιστική σημαντικότητα}
"""

PREAMBLE_EN = r"""% GENERATED FILE -- do not edit by hand.
% Produced by scripts/make_appendix_tables.py --en from results/*.csv.

\chapter{Full result tables}
\label{app:tables}

This appendix gives in full the tables that the chapters of the main text present in abridged
form. It is generated automatically from the result files, so that it cannot drift away from
them.

In every table of paired tests, \textbf{Mean diff.} is the mean per-round difference in loss
between the first-named series and the second, so a negative value means that the first is
better; \textbf{CI low} and \textbf{CI high} bound its 95\% moving block bootstrap confidence
interval, $n$ is the number of rounds the two series share and $p$ is the $p$-value of the test.

\section{Ranking and statistical significance}
"""

# Lead-in of each section: which tables it holds and what each one completes.
LEAD = {
 "ranking": (
  r"""Ο Πίνακας~\ref{app:ranking} δίνει πλήρη την κατάταξη της \S\ref{sec:res-ranking}, με την
κάλυψη, το πλήθος γύρων και την ακρίβεια κάθε σειράς, και ο Πίνακας~\ref{app:allseries} την
επεκτείνει σε κάθε σειρά του panel, και σε όσες είναι κάτω από το κατώφλι κάλυψης. Ο
Πίνακας~\ref{app:sig} παραθέτει τους ζευγαρωτούς ελέγχους της ίδιας ενότητας.""",
  r"""Table~\ref{app:ranking} gives in full the ranking of \S\ref{sec:res-ranking}, with each
series' coverage, number of rounds and accuracy, and Table~\ref{app:allseries} extends it to
every series of the panel, including those below the coverage threshold.
Table~\ref{app:sig} lists the paired tests quoted in that section."""),
 "panels": (
  r"""Οι Πίνακες~\ref{app:opening_vs_cprop} και~\ref{app:opening_vs_cShin} δίνουν το πλήρες
πλέγμα της \S\ref{sec:res-panels} (Πίνακας~\ref{tab:panels}) μαζί με Brier και ακρίβεια, και
οι Πίνακες~\ref{app:vsuniprop} και~\ref{app:vsuniShin} συγκρίνουν κάθε αλγόριθμο με τον απλό
μέσο όρο του ίδιου panel, τη σύγκριση από την οποία η ενότητα αυτή βγάζει το κύριο
συμπέρασμά της.""",
  r"""Tables~\ref{app:opening_vs_cprop} and~\ref{app:opening_vs_cShin} give the full grid of
\S\ref{sec:res-panels} (Table~\ref{tab:panels}) with the Brier score and accuracy added, and
Tables~\ref{app:vsuniprop} and~\ref{app:vsuniShin} compare each algorithm with the plain
average of the same panel, the comparison from which that section draws its main
conclusion."""),
 "configs": (
  r"""Ο Πίνακας~\ref{app:cfg} δίνει κάθε διαμόρφωση της \S\ref{sec:res-config} raw και
βαθμονομημένη (ο Πίνακας~\ref{tab:configs} δείχνει το βαθμονομημένο μισό), και ο
Πίνακας~\ref{app:cfgsig} τους ζευγαρωτούς ελέγχους στους οποίους στηρίζονται τα συμπεράσματά
της.""",
  r"""Table~\ref{app:cfg} gives every configuration of \S\ref{sec:res-config} raw and
calibrated (Table~\ref{tab:configs} shows the calibrated half), and Table~\ref{app:cfgsig} the
paired tests on which the conclusions of that section rest."""),
 "nopinnacle": (
  r"""Οι Πίνακες~\ref{app:np} και~\ref{app:npsig} δίνουν πλήρη τα τρία panel της
\S\ref{sec:res-nopinnacle}, που συνοψίζει ο Πίνακας~\ref{tab:nopinnacle}.""",
  r"""Tables~\ref{app:np} and~\ref{app:npsig} give in full the three panels of
\S\ref{sec:res-nopinnacle}, which Table~\ref{tab:nopinnacle} summarises."""),
 "recent": (
  r"""Οι Πίνακες~\ref{app:rw} και~\ref{app:rwsig} συμπληρώνουν την \S\ref{sec:res-recent}. Οι
Πίνακες~\ref{app:rwlive} και~\ref{app:rwlivesig} προσθέτουν μια σύγκριση που δεν εμφανίζεται
στο κυρίως κείμενο: την αγορά που μπορεί πράγματι να χρησιμοποιήσει σήμερα ένας παίκτης, δηλαδή
τις σειρές που δίνουν ακόμη αποδόσεις στο τελευταίο τέταρτο του συνόλου των δεδομένων. Εκεί το
Hedge κερδίζει σημαντικά κάθε τιμή ανοίγματος και ισοπαλεί με κάθε τιμή κλεισίματος. Οι
Πίνακες~\ref{app:dep} και~\ref{app:depsig} συμπληρώνουν την \S\ref{sec:res-deploy}.""",
  r"""Tables~\ref{app:rw} and~\ref{app:rwsig} complete \S\ref{sec:res-recent}.
Tables~\ref{app:rwlive} and~\ref{app:rwlivesig} add a comparison not shown in the main text:
the market a bettor can actually use today, that is the series still quoting in the last
quarter of the full history. There Hedge significantly beats every opening price and ties with
every closing one. Tables~\ref{app:dep} and~\ref{app:depsig} complete
\S\ref{sec:res-deploy}."""),
 "common": (
  r"""Οι Πίνακες~\ref{app:common}--\ref{app:common-vb} συμπληρώνουν την \S\ref{sec:res-common}: τις
επιδόσεις όλων των σειρών στους κοινούς αγώνες, τους ζευγαρωτούς ελέγχους και το value betting
στα σύνολα B και B$'$.""",
  r"""Tables~\ref{app:common}--\ref{app:common-vb} complete \S\ref{sec:res-common}: the scores of
every series on the common matches, the paired tests, and the value betting on datasets B and
B$'$."""),
 "value": (
  r"""Ο Πίνακας~\ref{app:vball} δίνει τον χάρτη πλεονεκτήματος του Πίνακα~\ref{tab:vb-sig} με
raw και με βαθμονομημένες πιθανότητες, ο Πίνακας~\ref{app:vbsweep} την πλήρη σάρωση μεθόδου
de-vig, panel εκπαίδευσης και στόχου πίσω από την αντιστοίχιση φάσης της
\S\ref{sec:res-value}, και ο Πίνακας~\ref{app:vbrw} την ίδια άσκηση στις τελευταίες πέντε
σεζόν. Και οι τρεις ποντάρουν το σταθερό $\tfrac14$ του κλάσματος Kelly. \textbf{Στοιχ.} είναι
το πλήθος των στοιχημάτων, \textbf{Κεφάλαιο} το τελικό κεφάλαιο με αφετηρία το 1 και
\textbf{log-growth} η μέση λογαριθμική αύξηση του κεφαλαίου ανά στοίχημα, με το 95\% διάστημα
εμπιστοσύνης της και την τιμή $p$ του ελέγχου moving block bootstrap ότι διαφέρει από το
μηδέν· εδώ, αντίθετα από τους πίνακες απώλειας, μεγαλύτερο σημαίνει καλύτερο.""",
  r"""Table~\ref{app:vball} gives the edge map of Table~\ref{tab:vb-sig} with raw as well as
calibrated probabilities, Table~\ref{app:vbsweep} the full sweep of de-vig method, training
panel and target behind the phase matching of \S\ref{sec:res-value}, and
Table~\ref{app:vbrw} the same exercise over the last five seasons. All three stake the fixed
$\tfrac14$ of the Kelly fraction. \textbf{Bets} is the number of bets placed, \textbf{Bankroll}
the final bankroll starting from 1, and \textbf{log-growth} the mean logarithmic growth of
capital per bet, with its 95\% confidence interval and the $p$-value of the moving block
bootstrap test that it differs from zero; here, unlike in the loss tables, higher is
better."""),
 "extensions": (
  r"""Οι υπόλοιποι πίνακες τεκμηριώνουν τα αποτελέσματα της \S\ref{sec:res-negative} και δύο
αναλύσεις που το κυρίως κείμενο αναφέρει μόνο εν παρόδω. Το Fixed-Share βρίσκεται στους
Πίνακες~\ref{app:fs} και~\ref{app:fsw}, οι ανά πρωτάθλημα ειδικοί στον Πίνακα~\ref{app:ctx} και
η επίδοση ανά κατηγορία αποτελέσματος στον Πίνακα~\ref{app:cls}. Οι
Πίνακες~\ref{app:calseason} και~\ref{app:calwarm} ελέγχουν δύο παραλλαγές του σταδίου
βαθμονόμησης: με επανεκκίνηση σε κάθε σεζόν εξακολουθεί να κερδίζει τις raw πιθανότητες για 11
από τις 13 σειρές, αλλά δεν κερδίζει ποτέ την ενιαία βαθμονόμηση, και η εκκίνηση κάθε σεζόν από
τη θερμοκρασία της προηγούμενης δεν το αλλάζει αυτό. Ο Πίνακας~\ref{app:mk} δίνει τους
ελέγχους τάσης της \S\ref{sec:disc-material}, ο Πίνακας~\ref{app:lr} την πλήρη σάρωση βήματος
της \S\ref{sec:res-steps}, ο Πίνακας~\ref{app:pool} τις κατατάξεις στα δύο χρονικά παράθυρα
και ο Πίνακας~\ref{app:info} τους ζευγαρωτούς ελέγχους της \S\ref{sec:res-arrival}.""",
  r"""The remaining tables document the results of \S\ref{sec:res-negative} and two analyses
that the main text mentions only in passing. Fixed-Share is in Tables~\ref{app:fs}
and~\ref{app:fsw}, the per-league specialists in Table~\ref{app:ctx} and the performance by
outcome class in Table~\ref{app:cls}. Tables~\ref{app:calseason} and~\ref{app:calwarm} test
two variants of the calibration stage: restarted every season, it still beats the raw
probabilities for 11 of the 13 series but never beats the single global calibration, and
starting each season from the previous season's temperature does not change that.
Table~\ref{app:mk} gives the trend tests of \S\ref{sec:disc-material}, Table~\ref{app:lr}
the full step sweep of \S\ref{sec:res-steps}, Table~\ref{app:pool} the rankings over the two
time windows and Table~\ref{app:info} the paired tests of \S\ref{sec:res-arrival}."""),
}


def section(el, en, lead):
    return T(f"\\section{{{el}}}\n", f"\\section{{{en}}}\n") + T(*LEAD[lead]) + "\n\n"


def main():
    parts = []
    add = parts.append

    add(T(PREAMBLE_EL, PREAMBLE_EN))
    add(T(*LEAD["ranking"]) + "\n\n")

    t = read("final_ranking_table.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Κατάταξη στις δέκα σεζόν των σειρών με κάλυψη τουλάχιστον 60\%, κατά
βαθμονομημένο log-loss. \textbf{Κάλυψη} είναι το ποσοστό των 76.584 αγώνων για τους οποίους η
σειρά έδωσε αποδόσεις και \textbf{Γύροι} το πλήθος τους. Τα log-loss και Brier δίνονται raw και
βαθμονομημένα, μικρότερο καλύτερο· \textbf{Acc.} είναι το ποσοστό των αγώνων στους οποίους
συνέβη το πιθανότερο αποτέλεσμα, που η βαθμονόμηση δεν το αλλάζει.""",
            r"""Ranking over the ten seasons of the series with coverage of at least 60\%, ordered
by calibrated log-loss. \textbf{Coverage} is the share of the 76,584 matches for which a series
quoted a price and \textbf{Rounds} their number. Log-loss and Brier are given raw and
calibrated, lower better; \textbf{Acc.} is the share of matches whose most probable outcome
occurred, which calibration does not change."""), "app:ranking",
            cols=["name", "coverage_pct", "rounds", "raw_log_loss", "calibrated_log_loss",
                  "raw_brier", "calibrated_brier", "calibrated_accuracy"]))

    t = read("results_table.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Όλες οι σειρές του panel χωρίς κατώφλι κάλυψης, raw επιδόσεις στους αγώνες που
κάλυψε η καθεμία, κατά log-loss. \textbf{Σεζόν} είναι το πλήθος των σεζόν στις οποίες έδωσε
αποδόσεις· μια σειρά ενεργή σε μία μόνο σεζόν βαθμολογείται μόνο σε αυτήν, οπότε η θέση της
μπερδεύει την ποιότητά της με τη δυσκολία της σεζόν. \textbf{Βάρος Hedge} είναι το βάρος που
της έδωσε το Hedge στον τελευταίο γύρο που ήταν ξύπνια.""",
            r"""Every series of the panel with no coverage threshold, raw scores over the matches
each series quoted, ordered by log-loss. \textbf{Seasons} is the number of seasons in which a
series quoted; one active in a single season is scored on that season alone, so its position
mixes its quality with that season's difficulty. \textbf{Hedge weight} is the weight Hedge gave
it at its own last awake round."""), "app:allseries",
            cols=["name", "coverage_pct", "n_seasons_active", "rounds", "log_loss", "brier",
                  "accuracy", "hedge_weight_last_awake"]))

    t = read("significance_test_table.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Οι ζευγαρωτοί έλεγχοι της \S\ref{sec:res-ranking}, στους γύρους που μοιράζονται
οι δύο σειρές. Σε σύγκριση της μορφής \emph{raw vs calibrated}, θετική \textbf{Μ. διαφορά}
σημαίνει ότι η βαθμονόμηση βοηθά.""",
            r"""The paired tests of \S\ref{sec:res-ranking}, over the rounds the two compared
series share. In a comparison of the form \emph{raw vs calibrated}, a positive
\textbf{Mean diff.} means that calibration helps."""), "app:sig",
            cols=["comparison", "n", "mean_diff", "ci_low", "ci_high", "p_value"]))

    add(section("Panel και μέθοδος de-vig", "Panel and de-vig method", "panels"))
    for f, cap_el, cap_en in (
            ("opening_vs_closing_table.csv",
             r"""Κάθε αλγόριθμος σε καθένα από τα τρία panel, με αναλογική κανονικοποίηση: raw
επιδόσεις στους γύρους που καλύπτει το panel, μικρότερα log-loss και Brier καλύτερα.
\textbf{Panel} είναι το opening-only, το πλήρες ή το closing-only, και \textbf{Experts} το
πλήθος των σειρών bookmaker που περιέχει.""",
             r"""Every algorithm on each of the three panels, with proportional normalisation:
raw scores over the rounds the panel covers, lower log-loss and Brier better. \textbf{Panel} is
opening-only, full or closing-only, and \textbf{Experts} is the number of bookmaker series in
it."""),
            ("opening_vs_closing_table_shin.csv",
             r"Όπως ο Πίνακας~\ref{app:opening_vs_cprop}, με τη μέθοδο de-vig του Shin.",
             r"As Table~\ref{app:opening_vs_cprop}, with Shin's de-vig method.")):
        t = read(f)
        if t is not None:
            add(longtable(t, T(cap_el, cap_en),
                          "app:" + f[:12] + ("Shin" if "shin" in f else "prop"),
                          cols=["market", "algorithm", "n_bookmakers", "rounds", "log_loss",
                                "brier", "accuracy"]))
    for f, cap_el, cap_en in (
            ("opening_vs_closing_vs_uniform.csv",
             r"""Κάθε αλγόριθμος έναντι του απλού μέσου όρου του ίδιου panel, αναλογική
κανονικοποίηση, για τρεις μετρικές. \textbf{Μ. διαφορά} είναι αλγόριθμος μείον μέσος όρος· για
log-loss και Brier αρνητική τιμή ευνοεί τον αλγόριθμο, για την ακρίβεια, όπου μεγαλύτερο είναι
καλύτερο, θετική. \textbf{Έκβαση} είναι το αποτέλεσμα στο επίπεδο 95\%.""",
             r"""Each algorithm against the plain average of the same panel, proportional
normalisation, for three metrics. \textbf{Mean diff.} is algorithm minus average; for log-loss
and Brier a negative value favours the algorithm, for accuracy, where higher is better, a
positive one does. \textbf{Verdict} gives the outcome at the 95\% level."""),
            ("opening_vs_closing_vs_uniform_shin.csv",
             r"Όπως ο Πίνακας~\ref{app:vsuniprop}, με τη μέθοδο de-vig του Shin.",
             r"As Table~\ref{app:vsuniprop}, with Shin's de-vig method.")):
        t = read(f)
        if t is not None:
            add(longtable(t, T(cap_el, cap_en),
                          "app:vsuni" + ("Shin" if "shin" in f else "prop"),
                          cols=["market", "algorithm", "metric", "n", "mean_diff", "ci_low",
                                "ci_high", "p_value", "verdict"]))

    add(section("Διαμορφώσεις", "Configurations", "configs"))
    t = read("best_configuration.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Log-loss του Hedge σε κάθε διαμόρφωση, στους ίδιους 71.818 αγώνες: \textbf{πλήρες}
σε όλους, \textbf{train} στο χρονολογικό πρόθεμα 75\% και \textbf{ουρά} στο υπόλοιπο 25\%,
μικρότερο καλύτερο· \textbf{Στάδιο} δείχνει αν οι πιθανότητες είναι raw ή βαθμονομημένες.
\emph{Μόνο PSC} είναι ο bookmaker αναφοράς, βαθμολογημένος στους ίδιους αγώνες.""",
            r"""Log-loss of Hedge in each configuration, on the same 71,818 matches: \textbf{full}
over all of them, \textbf{train} over their chronological 75\% prefix and \textbf{tail} over
the remaining 25\%, lower better; \textbf{Stage} says whether the probabilities are raw or
calibrated. \emph{PSC alone} is the benchmark bookmaker, scored on the same matches."""),
            "app:cfg",
            cols=["config", "stage", "full_log_loss", "train_log_loss", "val_log_loss"]))
    t = read("best_configuration_significance.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Κάθε διαμόρφωση έναντι του PSC μόνου και έναντι του πλήρους panel με αναλογική
κανονικοποίηση, ζευγαρωτά στους ίδιους 71.818 αγώνες. \textbf{Μ. διαφορά} είναι το log-loss
της διαμόρφωσης μείον της σειράς της στήλης \textbf{Έναντι}, οπότε αρνητική τιμή σημαίνει ότι
η διαμόρφωση είναι καλύτερη.""",
            r"""Each configuration against PSC alone and against the full panel with
proportional normalisation, paired over the same 71,818 matches. \textbf{Mean diff.} is the
configuration's log-loss minus that of the series in \textbf{Against}, so a negative value
means the configuration is better."""), "app:cfgsig",
            cols=["config", "stage", "vs", "mean_diff", "ci_low", "ci_high", "p_value"]))

    add(section("Χωρίς την Pinnacle", "Without Pinnacle", "nopinnacle"))
    t = read("no_pinnacle_ranking.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Κάθε σειρά κάθε panel που ξαναπροσαρμόστηκε χωρίς τις δύο στήλες της Pinnacle,
κατά βαθμονομημένο log-loss στους αγώνες που κοινούς σε όλο το panel (\textbf{\#} η θέση μέσα
στο panel, σε παρένθεση το πλήθος των experts που απομένουν). Το log-loss δίνεται raw και
βαθμονομημένο σε όλους τους κοινούς αγώνες, και βαθμονομημένο στο χρονολογικό πρόθεμα 75\%
(\textbf{train}) και στο υπόλοιπο 25\% (\textbf{ουρά}).""",
            r"""Every series of each panel refitted without the two Pinnacle columns, ranked by
calibrated log-loss on the matches common to the whole panel (\textbf{\#} is the rank within
the panel, and the number of experts left is in parentheses). Log-loss is given raw and
calibrated over all common matches, and calibrated over their chronological 75\% prefix
(\textbf{train}) and the remaining 25\% (\textbf{tail})."""), "app:np",
            cols=["panel", "rank", "name", "coverage_pct", "raw_log_loss",
                  "calibrated_log_loss", "train_log_loss", "val_log_loss"]))
    t = read("no_pinnacle_significance.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Το Hedge έναντι κάθε bookmaker κάθε panel χωρίς την Pinnacle, ζευγαρωτά στους κοινούς
αγώνες του panel, βαθμονομημένο log-loss. Αρνητική \textbf{Μ. διαφορά} σημαίνει ότι το Hedge είναι
καλύτερο.""",
            r"""Hedge against each bookmaker of each panel without Pinnacle, paired over the
panel's common matches, calibrated log-loss. A negative \textbf{Mean diff.} means that Hedge is
better."""), "app:npsig",
            cols=["panel", "bookmaker", "n", "mean_diff", "ci_low", "ci_high", "p_value",
                  "verdict"]))

    add(section("Πρόσφατο παράθυρο και έλεγχος ανάπτυξης",
                "Recent window and deployment test", "recent"))
    t = read("recent_window_ranking.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Κατάταξη στις τελευταίες πέντε σεζόν (38.749 αγώνες), πλήρες panel, κατά
βαθμονομημένο log-loss: ο Πίνακας~\ref{tab:recent} πλήρης. Η \textbf{Κάλυψη} υπολογίζεται μέσα
στο παράθυρο· log-loss και Brier δίνονται raw και βαθμονομημένα, μικρότερο καλύτερο, καθένα
στους αγώνες που κάλυψε η σειρά.""",
            r"""Ranking over the last five seasons (38,749 matches), full panel, ordered by
calibrated log-loss: Table~\ref{tab:recent} in full. \textbf{Coverage} is computed inside the
window; log-loss and Brier are given raw and calibrated, lower better, each over the matches
the series quoted."""), "app:rw",
            cols=["name", "phase", "coverage_pct", "raw_log_loss", "calibrated_log_loss",
                  "raw_brier", "calibrated_brier"]))
    t = read("recent_window_significance.csv")
    if t is not None:
        add(longtable(t, T(
            r"Οι ζευγαρωτοί έλεγχοι της \S\ref{sec:res-recent}, στις τελευταίες πέντε σεζόν.",
            r"The paired tests of \S\ref{sec:res-recent}, over the last five seasons."),
            "app:rwsig",
            cols=["comparison", "n", "mean_diff", "ci_low", "ci_high", "p_value"]))
    t = read("recent_window_live_market.csv")
    if t is not None:
        # one CSV, two tables: the ranking rows, then Hedge's tests against the books
        n_live = count(t.loc[t["bookmaker"].notna(), "n"].iloc[0])
        add(longtable(t[t["rank"].notna()], T(
            r"""Η αγορά του σήμερα: οι σειρές που περνούν το κατώφλι κάλυψης στο τελευταίο
τέταρτο του συνόλου των δεδομένων (19.146 αγώνες), κατά βαθμονομημένο log-loss στους @N@ από
αυτούς τους αγώνες που κάλυψαν όλες. Τα μείγματα προσαρμόζονται σε όλο το ιστορικό σε μία
αιτιακή διέλευση, και η \textbf{Κάλυψη} υπολογίζεται μόνο στο τελευταίο τέταρτο.""",
            r"""The market of today: the series that clear the coverage threshold in the last
quarter of the full history (19,146 matches), ranked by calibrated log-loss on the @N@ of
those matches that all of them quoted. The mixtures are fitted on the full history in one
causal run, and \textbf{Coverage} is computed on the last quarter alone.""").replace("@N@", n_live),
            "app:rwlive",
            cols=["rank", "name", "phase", "coverage_tail_pct", "log_loss"]))
        add(longtable(t[t["bookmaker"].notna()], T(
            r"""Το Hedge έναντι κάθε bookmaker του Πίνακα~\ref{app:rwlive}, ζευγαρωτά στους ίδιους
@N@ αγώνες· αρνητική \textbf{Μ. διαφορά} σημαίνει ότι το Hedge είναι καλύτερο.""",
            r"""Hedge against each bookmaker of Table~\ref{app:rwlive}, paired over the same @N@
matches; a negative \textbf{Mean diff.} means that Hedge is better.""").replace("@N@", n_live),
            "app:rwlivesig",
            cols=["bookmaker", "phase", "n", "mean_diff", "ci_low", "ci_high", "p_value",
                  "verdict"]))
    t = read("deployment_test.csv")
    s = read("deployment_test_significance.csv")
    if t is not None and s is not None:
        n_dep = count(s.loc[~s["aside"].astype(bool), "n"].iloc[0])
        add(longtable(t, T(
            r"""Έλεγχος ανάπτυξης: εκπαίδευση έως 31/12/2024 και βαθμονομημένο log-loss στους
@N@ αγώνες του 2025--26 που κάλυψαν όλες οι σειρές· ο Πίνακας~\ref{tab:deploy-table} πλήρης.
\textbf{Κάλυψη} είναι το ποσοστό των αγώνων του παραθύρου που κάλυψε η σειρά, και
\emph{online} και \emph{frozen} οι δύο αναγνώσεις του μείγματος που ορίζει η
\S\ref{sec:res-deploy}.""",
            r"""Deployment test: trained up to 31/12/2024 and scored by calibrated log-loss on
the @N@ matches of 2025--26 that every listed series quoted; Table~\ref{tab:deploy-table} in
full. \textbf{Coverage} is the share of the test window's matches a series quoted, and
\emph{online} and \emph{frozen} are the two readings of the mixture defined in
\S\ref{sec:res-deploy}.""").replace("@N@", n_dep), "app:dep",
            cols=["rank", "name", "type", "coverage_test_pct", "log_loss"]))
    t = read("deployment_test_significance.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Κάθε παραλλαγή του μείγματος έναντι κάθε bookmaker που δίνει ακόμη αποδόσεις το
2025--26, ζευγαρωτά, βαθμονομημένο log-loss. Αρνητική \textbf{Μ. διαφορά} σημαίνει ότι το μείγμα
είναι καλύτερο. Η Pinnacle (PS, PSC), της οποίας η κάλυψη πέφτει κάτω από το κατώφλι σε αυτό
το παράθυρο, συγκρίνεται στους δικούς της γύρους, γι' αυτό και το μικρότερο $n$.""",
            r"""Each variant of the mixture against each bookmaker still quoting in 2025--26,
paired, calibrated log-loss. A negative \textbf{Mean diff.} means that the mixture is better.
Pinnacle (PS, PSC), whose coverage falls below the threshold in this window, is compared over
its own rounds, hence its smaller $n$."""), "app:depsig",
            cols=["variant", "bookmaker", "n", "mean_diff", "ci_low", "ci_high", "p_value",
                  "verdict"]))

    add(section("Κοινοί αγώνες", "Common matches", "common"))
    t = read("common_matches_table.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Βαθμονομημένο log-loss κάθε σειράς στους αγώνες που κάλυψαν όλες οι σειρές του
συνόλου, σε όλους τους αγώνες και στην ουρά 25\%, μικρότερο καλύτερο. \textbf{Βήμα} είναι ο
πολλαπλασιαστής βήματος που επιλέγει κάθε αλγόριθμος στο χρονολογικό πρόθεμα 75\%.""",
            r"""Calibrated log-loss of every series on the matches that every series of the
dataset quoted, over all of them and over the 25\% tail, lower better. \textbf{Step} is the
step multiplier each algorithm selects on the chronological 75\% prefix."""), "app:common",
            cols=["dataset", "series", "step_multiplier", "calibrated_log_loss",
                  "tail_log_loss"]))
    t = read("common_matches_significance.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Οι ζευγαρωτοί έλεγχοι της \S\ref{sec:res-common}, σε όλους τους αγώνες του
συνόλου και στην ουρά 25\%· αρνητική διαφορά σημαίνει ότι η σειρά που κατονομάζεται πρώτη είναι
καλύτερη.""",
            r"""The paired tests of \S\ref{sec:res-common}, over all matches of the dataset and
over its 25\% tail; a negative difference means that the first-named series is better."""),
            "app:commonsig",
            cols=["dataset", "comparison", "mean_diff", "ci_low", "ci_high", "p_value",
                  "tail_mean_diff", "tail_p_value"]))
    t = read("common_matches_value_betting.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Value betting στα σύνολα B και B$'$ (χωρίς Pinnacle), με το σταθερό $\tfrac14$
του κλάσματος Kelly απέναντι στις πραγματικές αποδόσεις και μείγμα Hedge της φάσης του στόχου,
χωρίς τον ίδιο τον στόχο· στήλες όπως στον Πίνακα~\ref{app:vbsweep}.""",
            r"""Value betting on datasets B and B$'$ (without Pinnacle), with the fixed $\tfrac14$
of the Kelly fraction against the real odds and a Hedge mixture of the target's phase without the
target itself; columns as in Table~\ref{app:vbsweep}."""), "app:common-vb",
            cols=["panel", "bookmaker", "phase", "n_bets", "final_bankroll",
                  "mean_log_growth_per_bet", "ci_low", "ci_high", "p_value"]))

    add(section("Value betting", "Value betting", "value"))
    t = read("value_betting_all_bookmakers_table.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Value betting έναντι καθενός από τους έντεκα στόχους, με raw και με βαθμονομημένες
πιθανότητες· το μείγμα εκπαιδεύεται στο panel της φάσης του στόχου, χωρίς τον ίδιο τον
στόχο.""",
            r"""Value betting against each of the eleven targets, with raw and with calibrated
probabilities; the mixture is trained on the panel of the target's own phase, with the target
itself removed."""), "app:vball",
            cols=["bookmaker", "stage", "n_bets", "final_bankroll", "mean_log_growth_per_bet",
                  "ci_low", "ci_high", "p_value"]))
    t = read("value_betting_training_panel_sweep.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Value betting έναντι κάθε στόχου, για κάθε συνδυασμό μεθόδου de-vig, panel
εκπαίδευσης (\textbf{Panel εκπ.}: opening-only, closing-only ή πλήρες) και σταδίου. Οι γραμμές
που ποντάρουν σε τιμή ανοίγματος με το πλήρες panel χρησιμοποιούν τιμές κλεισίματος που δεν
υπάρχουν ακόμη τη στιγμή του στοιχήματος, και δίνονται μόνο για πληρότητα.""",
            r"""Value betting against each target, for every combination of de-vig method,
training panel (\textbf{Train panel}: opening-only, closing-only or full) and stage. The rows
that bet on an opening price with the full panel use closing prices that do not yet exist when
the bet is placed, and are shown only for completeness."""), "app:vbsweep",
            cols=["normalization", "panel_type", "bookmaker", "stage", "n_bets",
                  "final_bankroll", "mean_log_growth_per_bet", "ci_low", "ci_high",
                  "p_value"]))
    t = read("recent_window_value_betting.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Value betting στις τελευταίες πέντε σεζόν, με το μείγμα εκπαιδευμένο είτε στο
πλήρες panel είτε στο panel που ταιριάζει στη φάση του στόχου (\textbf{Panel εκπ.}).""",
            r"""Value betting over the last five seasons, with the mixture trained either on the
full panel or on the panel matched to the target's phase (\textbf{Train panel})."""),
            "app:vbrw",
            cols=["bookmaker", "training_panel", "stage", "n_bets", "final_bankroll",
                  "mean_log_growth_per_bet", "ci_low", "ci_high", "p_value"]))

    add(section("Επεκτάσεις και αρνητικά αποτελέσματα",
                "Extensions and negative results", "extensions"))
    t = read("fixed_share_table.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Fixed-Share (\S\ref{sec:fixed-share}) ανά ρυθμό διαρροής $\alpha$, με το βήμα του
Hedge σε ισχύ: raw επιδόσεις σε όλους τους 76.584 γύρους, μικρότερα log-loss και Brier
καλύτερα. Το $\alpha=0$ είναι το απλό Hedge.""",
            r"""Fixed-Share (\S\ref{sec:fixed-share}) by leak rate $\alpha$, with the Hedge step
in force: raw scores over all 76,584 rounds, lower log-loss and Brier better. $\alpha=0$ is
plain Hedge."""), "app:fs",
            cols=["alpha", "log_loss", "brier", "accuracy"]))
    t = read("fixed_share_weights.csv")
    if t is not None:
        alphas = [c for c in t.columns if c != "bookmaker"]
        t = t.rename(columns={a: f"$\\alpha={float(a):g}$" for a in alphas})
        for a in alphas:
            FORMAT[f"$\\alpha={float(a):g}$"] = "{:.3f}"
        add(longtable(t, T(
            r"""Βάρος Hedge των έντεκα σειρών που δίνουν αποδόσεις μόνο στις τελευταίες σεζόν,
στον τελευταίο γύρο που ήταν ξύπνια η καθεμία, ανά $\alpha$. Χωρίς διαρροή ($\alpha=0$) οι
τιμές ανοίγματος κρατούν σχεδόν μηδενικό βάρος (0,001 έως 0,012) και οι τιμές κλεισίματος 0,02
έως 0,14· κάθε $\alpha>0$ φέρνει όλες κοντά σε ίσο μερίδιο μεταξύ των experts που ήταν ξύπνιοι
μαζί τους (0,06 έως 0,11).""",
            r"""Hedge weight of the eleven series that quote only in the last seasons, read at
each one's own last awake round, by $\alpha$. Without a leak ($\alpha=0$) the opening
series hold almost no weight (0.001 to 0.012) and the closing ones 0.02 to 0.14; any $\alpha>0$
pulls all of them towards an equal share among the experts awake with them (0.06 to 0.11)."""), "app:fsw"))
    t = read("contextual_experts_table.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Ανά πρωτάθλημα ειδικοί: το Hedge εκπαιδευμένο σε ένα μόνο πρωτάθλημα
(\textbf{Ειδικός}) έναντι του ενιαίου Hedge εκπαιδευμένου σε όλα (\textbf{Ενιαίο}), με log-loss
στους αγώνες του πρωταθλήματος, μικρότερο καλύτερο. \textbf{Νίκη ειδ.} δείχνει αν ο ειδικός
είναι καλύτερος, και \textbf{Πιο έμπιστοι} οι τρεις σειρές με το μεγαλύτερο βάρος στον
ειδικό.""",
            r"""Per-league specialists: Hedge trained on one league alone (\textbf{Specialist})
against the single Hedge trained on all of them (\textbf{Unified}), both scored by log-loss on that
league's matches, lower better. \textbf{Spec. wins} says whether the specialist is better, and
\textbf{Most trusted} lists the three series to which the specialist gave the largest
weight."""), "app:ctx",
            cols=["league_name", "n_matches", "global_log_loss", "specialist_log_loss",
                  "specialist_wins", "top3_trusted_bookmakers"]))
    t = read("outcome_class_accuracy_table.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Επίδοση ανά κατηγορία αποτελέσματος: για κάθε σειρά και κάθε πραγματικό
αποτέλεσμα, $n$ είναι το πλήθος των αγώνων με αυτό το αποτέλεσμα, \textbf{Acc.} το ποσοστό τους
στο οποίο το πιθανότερο αποτέλεσμα της σειράς ήταν το σωστό, και log-loss και Brier
υπολογίζονται μόνο σε αυτούς τους αγώνες.""",
            r"""Performance by outcome class: for each series and each actual outcome, $n$ is
the number of matches with that outcome, \textbf{Acc.} the share of them in which the series'
most probable outcome was the right one, and log-loss and Brier are computed on those matches
alone."""), "app:cls",
            cols=["name", "outcome", "n", "accuracy", "log_loss", "brier"]))
    t = read("calibration_per_season_significance.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Βαθμονόμηση με επανεκκίνηση στην αρχή κάθε σεζόν, έναντι των raw πιθανοτήτων και
έναντι της ενιαίας βαθμονόμησης της \S\ref{sec:calib-theory}, ζευγαρωτά στους γύρους κάθε
σειράς· αρνητική \textbf{Μ. διαφορά} σημαίνει ότι η παραλλαγή ανά σεζόν είναι καλύτερη.""",
            r"""Calibration restarted at the start of every season, against the raw
probabilities and against the single global calibration of \S\ref{sec:calib-theory}, paired
over each series' rounds; a negative \textbf{Mean diff.} means that the per-season variant is
better."""), "app:calseason",
            cols=["series", "comparison", "n", "mean_diff", "ci_low", "ci_high", "p_value"]))
    t = read("calibration_warmstart_significance.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Βαθμονόμηση ανά σεζόν που ξεκινά κάθε σεζόν από την τελική θερμοκρασία της
προηγούμενης (warm start), έναντι των raw πιθανοτήτων, της ενιαίας βαθμονόμησης και της
παραλλαγής ανά σεζόν που ξεκινά από την αρχή· αρνητική \textbf{Μ. διαφορά} σημαίνει ότι το warm
start είναι καλύτερο.""",
            r"""Per-season calibration that starts each season from the previous season's final
temperature (warm start), against the raw probabilities, the global calibration and the
per-season variant restarted from scratch; a negative \textbf{Mean diff.} means that the warm
start is better."""), "app:calwarm",
            cols=["series", "comparison", "n", "mean_diff", "ci_low", "ci_high", "p_value"]))
    t = read("temporal_trends_mk_test.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Έλεγχοι Mann--Kendall για μονότονη τάση στις δέκα σεζόν, ανά σειρά και μετρική.
$S$ είναι η στατιστική Mann--Kendall, $Z$ η κανονική της προσέγγιση και $p$ η αμφίπλευρη τιμή
$p$· η \textbf{Τάση} δίνει το πρόσημο του $S$, που σημαίνει κάτι μόνο όπου το $p$ είναι μικρό.
ECE είναι το αναμενόμενο σφάλμα βαθμονόμησης (expected calibration error) και
\emph{περιθώριο} το περιθώριο του bookmaker.""",
            r"""Mann--Kendall tests for a monotone trend over the ten seasons, per series and
metric. $S$ is the Mann--Kendall statistic, $Z$ its normal approximation and $p$ the two-sided
$p$-value; \textbf{Trend} gives the sign of $S$, which means something only where $p$ is
small. ECE is the expected calibration error and \emph{overround} the bookmaker's
margin."""), "app:mk",
            cols=["series", "metric", "S", "Z", "p_value", "trend"]))
    t = read("learning_rate_study.csv")
    if t is not None:
        # The train figure is exactly recoverable from full and tail, since
        # log-loss is a plain mean over rounds; it is the one selection uses.
        n_all = 76584
        n_train = int(round(n_all * 0.75))
        t = t.copy()
        t["train_log_loss"] = ((t["full_log_loss"] * n_all
                                - t["val_log_loss"] * (n_all - n_train)) / n_train)
        t["setting"] = [nm if pd.isna(s) else float(s)
                        for s, nm in zip(t["setting"], t["name"])]
        if EN:
            t["setting"] = [s.replace("(shipped)", "(in force)") if isinstance(s, str)
                            else s for s in t["setting"]]
        else:
            t["setting"] = [s.replace("(shipped)", "(σε ισχύ)") if isinstance(s, str)
                            else s for s in t["setting"]]
        add(longtable(t, T(
            r"""Η πλήρης σάρωση βήματος της \S\ref{sec:res-steps}: για κάθε οικογένεια σχημάτων
βήματος, κάθε ρύθμιση που δοκιμάστηκε, με το log-loss σε όλο το δείγμα (\textbf{πλήρες}), στο
χρονολογικό πρόθεμα 75\% στο οποίο γίνεται η επιλογή (\textbf{train}) και στην ουρά 25\% που
κρατήθηκε έξω (\textbf{ουρά}). Οι γραμμές \emph{πολλαπλασιαστή} κλιμακώνουν το φθίνον βήμα των
εγχειριδίων, οι γραμμές \emph{σταθερού $\eta$} το αντικαθιστούν με σταθερό βήμα, και οι
υπόλοιπες είναι προσαρμοστικές (AdaGrad) ή παραλλαγές του OGD με άλλο φράγμα απώλειας $M$, και
οι ρυθμίσεις σε ισχύ.""",
            r"""The full step sweep of \S\ref{sec:res-steps}: for each family of step schedules,
every setting tried, with the log-loss on the full sample (\textbf{full}), on the chronological
75\% prefix on which the selection is made (\textbf{train}) and on the held-out 25\% tail
(\textbf{tail}). \emph{Multiplier} rows scale the textbook decaying schedule, \emph{constant
$\eta$} rows replace it with a constant step, and the remaining rows are adaptive (AdaGrad) or
loss-bound $M$ variants of OGD, and the settings in force."""), "app:lr",
            cols=["family", "setting", "full_log_loss", "train_log_loss", "val_log_loss"]))
    t = read("pooling_window_ranking_table.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Κατάταξη σε όλες τις δέκα σεζόν και στις τελευταίες πέντε, για τις σειρές που
περνούν το κατώφλι κάλυψης σε κάθε παράθυρο· log-loss raw και βαθμονομημένο, μικρότερο
καλύτερο, καθένα στους αγώνες που κάλυψε η σειρά.""",
            r"""Ranking over the full ten seasons and over the last five, for the series that
clear the coverage threshold in each window; log-loss raw and calibrated, lower better, each
over the matches the series quoted."""), "app:pool",
            cols=["window", "name", "coverage_pct", "rounds", "raw_log_loss",
                  "calibrated_log_loss"]))
    t = read("information_arrival_significance.csv")
    if t is not None:
        add(longtable(t, T(
            r"""Οι ζευγαρωτοί έλεγχοι πίσω από τον Πίνακα~\ref{tab:arrival} και το
Σχήμα~\ref{fig:arrival}: σε κάθε πεμπτημόριο κάθε διαστρωμάτωσης, το Hedge του closing-only panel
έναντι του Hedge του opening-only panel, raw log-loss. Αρνητική \textbf{Μ. διαφορά} σημαίνει ότι
το closing panel είναι καλύτερο.""",
            r"""The paired tests behind Table~\ref{tab:arrival} and Figure~\ref{fig:arrival}: in
each quintile of each stratification, Hedge fitted on the closing-only panel against Hedge fitted on
the opening-only panel, raw log-loss. A negative \textbf{Mean diff.} means that the closing
panel is better."""), "app:info",
            cols=["stratifier", "stratum", "n", "mean_diff", "ci_low", "ci_high", "p_value"]))

    out_tex = OUT_TEX_EN if EN else OUT_TEX
    with open(out_tex, "w", encoding="utf-8", newline="") as fh:
        fh.write("".join(parts))

    n_tables = sum(p.count(r"\begin{longtable}") for p in parts)
    print(f"Saved: {out_tex}  ({n_tables} {T('πίνακες', 'tables')})")


if __name__ == "__main__":
    EN = "--en" in sys.argv[1:]
    HEAD = HEAD_EN if EN else HEAD_EL
    main()
