#!/usr/bin/env python3
"""
analyze_search_stats.py
-----------------------
Reads newline-delimited JSON from search_stats.json (one object per search call),
aggregates across all searches, and produces a multi-panel matplotlib figure.

Usage:
    python analyze_search_stats.py [search_stats.json] [--out report.png]

Dependencies:
    pip install matplotlib pandas
"""

import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#4C9BE8"
C_GREEN  = "#56C96B"
C_ORANGE = "#F5A623"
C_RED    = "#E85C5C"
C_PURPLE = "#9B59B6"
C_GREY   = "#8E9AAB"

BAR_COLORS = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE, C_GREY,
              "#1ABC9C", "#E67E22", "#2ECC71", "#E74C3C"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_large(n: float) -> str:
    """Human-readable large number."""
    if n >= 1e9:  return f"{n/1e9:.2f}B"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.1f}K"
    return str(int(n))

def pct(num: float, denom: float) -> str:
    if denom == 0: return "—"
    return f"{100*num/denom:.1f}%"

def hbar(ax, labels, values, colors=None, title="", xlabel="count", pct_of=None):
    """Horizontal bar chart, optional % annotation."""
    if colors is None:
        colors = BAR_COLORS[:len(labels)]
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: fmt_large(x)))
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        label = fmt_large(val)
        if pct_of is not None:
            label += f"  ({pct(val, pct_of)})"
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=7.5, color="#333")
    ax.spines[["top","right"]].set_visible(False)

def pie_chart(ax, labels, values, title=""):
    """Pie chart, suppressing zero slices."""
    pairs = [(l, v) for l, v in zip(labels, values) if v > 0]
    if not pairs:
        ax.set_visible(False)
        return
    ls, vs = zip(*pairs)
    colors = BAR_COLORS[:len(ls)]
    wedges, _, autotexts = ax.pie(
        vs, labels=None, autopct="%1.1f%%", startangle=90,
        colors=colors, pctdistance=0.80,
        wedgeprops=dict(edgecolor="white", linewidth=1.2))
    ax.legend(wedges, ls, loc="lower center", bbox_to_anchor=(0.5, -0.20),
              fontsize=7.5, frameon=False, ncol=2)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for at in autotexts:
        at.set_fontsize(7)

def ratio_bar(ax, data: dict, title=""):
    """Bar of ratio values (0–100 scale = %)."""
    labels = list(data.keys())
    vals   = list(data.values())
    colors = [C_GREEN if v >= 70 else (C_ORANGE if v >= 40 else C_RED) for v in vals]
    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, edgecolor="none", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("percent (%)", fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(min(bar.get_width() + 1, 98), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.axvline(80, color="#ccc", linewidth=0.8, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

# ── Load data ─────────────────────────────────────────────────────────────────

def load(path: str) -> pd.Series:
    rows = []
    with open(path) as f:
        # Read the whole file and split by lines to handle stray newlines
        content = f.read().strip()
        
        # If your engine is accidentally writing {obj1}{obj2} 
        # instead of {obj1}\n{obj2}, this fix will handle both:
        content = content.replace("}{", "}\n{") 
        
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {e}")
                continue
                
    if not rows:
        sys.exit(f"No data found in {path}")
    
    df = pd.DataFrame(rows)
    return df.sum()   # aggregate across all searches

# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(s: pd.Series, out_path: str):
    def g(key, default=0):
        return float(s.get(key, default))

    fig = plt.figure(figsize=(22, 26), facecolor="#F7F9FC")
    fig.suptitle("Search Statistics Analysis", fontsize=16, fontweight="bold", y=0.995)

    gs = fig.add_gridspec(5, 3, hspace=0.55, wspace=0.42,
                          left=0.07, right=0.97, top=0.97, bottom=0.03)

    # ── 0. Node breakdown ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    total_nodes = g("search_nodes") + g("qsearch_nodes")
    hbar(ax,
         ["PV search", "NonPV search", "QSearch"],
         [g("search_nodes_pv"), g("search_nodes_nonpv"), g("qsearch_nodes")],
         colors=[C_BLUE, C_ORANGE, C_GREEN],
         title="Node distribution",
         xlabel="nodes",
         pct_of=total_nodes)

    # ── 1. TT quality ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    tt_reads  = g("tt_reads")
    tt_hits   = g("tt_hits")
    tt_cut    = g("tt_cutoffs_taken")
    tt_block  = g("tt_cutoffs_blocked_50mr")
    hbar(ax,
         ["Reads", "Hits", "Cutoffs taken", "Blocked (50mr)"],
         [tt_reads, tt_hits, tt_cut, tt_block],
         colors=[C_BLUE, C_GREEN, C_ORANGE, C_RED],
         title="Transposition Table",
         xlabel="events")

    # ── 2. TT efficiency ratios ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ratio_bar(ax, {
        "TT hit rate  (hits/reads)":        100*tt_hits / max(tt_reads, 1),
        "Cutoff rate  (cutoffs/attempts)":  100*tt_cut  / max(g("tt_cutoff_attempts"), 1),
        "Blocker rate (blocked/attempts)":  100*tt_block/ max(g("tt_cutoff_attempts"), 1),
    }, title="TT efficiency ratios")

    # ── 3. Pre-move pruning (node savings) ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    hbar(ax,
         ["TT cutoff", "RFP", "NMP direct", "NMP verified", "Razoring", "ProbCut"],
         [g("tt_cutoffs_taken"), g("rfp_hits"), g("nmp_cutoffs_direct"),
          g("nmp_cutoffs_verified"), g("razoring_hits"), g("probcut_cutoffs")],
         title="Pre-move pruning (whole-node escapes)",
         xlabel="returns",
         pct_of=g("search_nodes"))

    # ── 4. Per-move pruning ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    hbar(ax,
         ["LMP", "FP", "BNFP", "SEE quiet", "SEE noisy"],
         [g("lmp_hits"), g("fp_hits"), g("bnfp_hits"),
          g("see_prune_quiet"), g("see_prune_noisy")],
         title="Per-move pruning",
         xlabel="moves pruned")

    # ── 5. NMP funnel ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    nmp_att = g("nmp_attempts")
    hbar(ax,
         ["Attempts", "Direct cutoff", "Verification searches", "Verified cutoff"],
         [nmp_att, g("nmp_cutoffs_direct"), g("nmp_verifications"), g("nmp_cutoffs_verified")],
         colors=[C_BLUE, C_GREEN, C_ORANGE, C_GREEN],
         title="Null Move Pruning funnel",
         pct_of=nmp_att)

    # ── 6. Singular Extensions breakdown ──────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    se_cand = g("se_candidates")
    hbar(ax,
         ["Candidates", "Single +1", "Double +2", "Triple +3",
          "Multi-cut", "Negative -2", "TT move cleared"],
         [se_cand,
          g("se_single_extension"), g("se_double_extension"), g("se_triple_extension"),
          g("se_multicut"), g("se_negative_extension"), g("se_tt_move_cleared")],
         colors=[C_GREY, C_GREEN, C_BLUE, C_PURPLE, C_RED, C_ORANGE, C_GREY],
         title="Singular Extensions",
         pct_of=se_cand)

    # ── 7. SE outcome pie ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    pie_chart(ax,
              ["Single", "Double", "Triple", "Multi-cut", "Negative", "TT clr", "No ext"],
              [g("se_single_extension"), g("se_double_extension"), g("se_triple_extension"),
               g("se_multicut"), g("se_negative_extension"), g("se_tt_move_cleared"),
               se_cand - g("se_single_extension") - g("se_double_extension")
                       - g("se_triple_extension") - g("se_multicut")
                       - g("se_negative_extension") - g("se_tt_move_cleared")],
              title="SE outcome distribution")

    # ── 8. Move ordering quality ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    total_cuts = g("beta_cutoffs_total")
    hbar(ax,
         ["Move 1", "Move 2", "Move 3-5", "Move 6+"],
         [g("beta_cutoff_move_1"), g("beta_cutoff_move_2"),
          g("beta_cutoff_move_3_to_5"), g("beta_cutoff_move_6_plus")],
         colors=[C_GREEN, C_BLUE, C_ORANGE, C_RED],
         title=f"Move ordering quality  (total β-cuts: {fmt_large(total_cuts)})",
         pct_of=total_cuts)

    # ── 9. Move ordering ratio bar ────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    ratio_bar(ax, {
        "1st-move cutoff rate  (ideal ≥ 80%)":
            100 * g("beta_cutoff_move_1") / max(total_cuts, 1),
        "1st or 2nd move rate  (ideal ≥ 90%)":
            100 * (g("beta_cutoff_move_1") + g("beta_cutoff_move_2")) / max(total_cuts, 1),
        "QS 1st-move rate":
            100 * g("qs_beta_cutoff_move_1")
                / max(g("qs_beta_cutoff_move_1") + g("qs_beta_cutoff_move_2_plus"), 1),
    }, title="Move ordering efficiency")

    # ── 10. LMR efficiency ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    lmr_app    = g("lmr_applied")
    lmr_res    = g("lmr_research_needed")
    lmr_ext    = g("lmr_depth_extended")
    hbar(ax,
         ["LMR applied", "Re-search triggered", "Depth extended after re-search"],
         [lmr_app, lmr_res, lmr_ext],
         colors=[C_BLUE, C_ORANGE, C_RED],
         title="Late Move Reductions",
         pct_of=lmr_app)

    # ── 11. Aspiration windows ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 2])
    total_asp = g("aspiration_fail_low") + g("aspiration_fail_high")
    hbar(ax,
         ["Fail low (widen β)", "Fail high (widen α)"],
         [g("aspiration_fail_low"), g("aspiration_fail_high")],
         colors=[C_RED, C_ORANGE],
         title=f"Aspiration window failures  (total: {fmt_large(total_asp)})")

    # ── 12. QSearch breakdown ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[4, 0])
    qs_nodes = g("qsearch_nodes")
    hbar(ax,
         ["QS nodes", "TT cutoffs", "Stand-pat cutoffs", "LMP", "SEE prune"],
         [qs_nodes, g("qs_tt_cutoffs"), g("qs_stand_pat_cutoffs"),
          g("qs_lmp_hits"), g("qs_see_prune_hits")],
         colors=[C_GREY, C_GREEN, C_BLUE, C_ORANGE, C_RED],
         title="QSearch activity",
         pct_of=qs_nodes)

    # ── 13. Hindsight depth modifications ────────────────────────────────────
    ax = fig.add_subplot(gs[4, 1])
    hbar(ax,
         ["Depth +1 (hindsight inc.)", "Depth -1 (hindsight dec.)"],
         [g("hindsight_depth_increase"), g("hindsight_depth_decrease")],
         colors=[C_GREEN, C_RED],
         title="Hindsight depth adjustments",
         pct_of=g("search_nodes"))

    # ── 14. ProbCut efficiency ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[4, 2])
    pc_tried = g("probcut_move_tried")
    hbar(ax,
         ["Moves tried", "Cutoffs"],
         [pc_tried, g("probcut_cutoffs")],
         colors=[C_BLUE, C_GREEN],
         title=f"ProbCut  (cutoff/{'' if pc_tried==0 else fmt_large(pc_tried)} tried)",
         pct_of=max(pc_tried, 1))

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Report saved → {out_path}")

    # ── Print text summary ────────────────────────────────────────────────────
    print("\n════════════════════════════════════════════════════════")
    print("  SEARCH STATS SUMMARY")
    print("════════════════════════════════════════════════════════")

    total = g("search_nodes") + g("qsearch_nodes")
    print(f"\n  Nodes total          : {fmt_large(total)}")
    print(f"  ├─ Search (PV)       : {fmt_large(g('search_nodes_pv'))}  ({pct(g('search_nodes_pv'), total)})")
    print(f"  ├─ Search (NonPV)    : {fmt_large(g('search_nodes_nonpv'))}  ({pct(g('search_nodes_nonpv'), total)})")
    print(f"  └─ QSearch           : {fmt_large(g('qsearch_nodes'))}  ({pct(g('qsearch_nodes'), total)})")

    print(f"\n  TT hit rate          : {pct(g('tt_hits'), g('tt_reads'))}")
    print(f"  TT cutoff rate       : {pct(g('tt_cutoffs_taken'), g('tt_cutoff_attempts'))}")

    print(f"\n  Move ordering (main) : {pct(g('beta_cutoff_move_1'), g('beta_cutoffs_total'))} first-move cutoffs  "
          f"(ideal ≥ 80%)")
    print(f"  Move ordering (QS)   : {pct(g('qs_beta_cutoff_move_1'), g('qs_beta_cutoff_move_1')+g('qs_beta_cutoff_move_2_plus'))} first-move cutoffs")

    print(f"\n  LMR re-search rate   : {pct(g('lmr_research_needed'), g('lmr_applied'))}")

    print(f"\n  NMP cutoff rate      : {pct(g('nmp_cutoffs_direct')+g('nmp_cutoffs_verified'), g('nmp_attempts'))}")
    print(f"  RFP hits / node      : {pct(g('rfp_hits'), g('search_nodes'))}")
    print(f"  SE extension rate    : {pct(g('se_single_extension')+g('se_double_extension')+g('se_triple_extension'), g('se_candidates'))}")
    print(f"  SE multi-cut rate    : {pct(g('se_multicut'), g('se_candidates'))}")
    print("════════════════════════════════════════════════════════\n")


# ── Interpretation guide (printed to stdout) ──────────────────────────────────

GUIDE = """
INTERPRETATION GUIDE
════════════════════════════════════════════════════════════════════════════
MOVE ORDERING (most impactful)
  • First-move β-cutoff rate < 75%  → move ordering is poor; improve killer/history
  • High "Move 6+" share            → TT move missing or history heuristics weak

TT QUALITY
  • Hit rate < 50%                  → TT is too small for the position set
  • Cutoff blocked (50mr) is high   → Position set has many endgames; expected
  • Cutoff rate << hit rate         → Many hits are stale depth (entries too shallow)

LMR
  • Re-search rate > 30%            → Reductions are too aggressive; reduce LMR depth
  • Re-search rate < 5%             → Reductions may be too conservative (leaving speed on table)

NMP
  • NMP cutoff rate < 40%           → zugzwang detection not needed; NMP conditions too loose
  • Many verifications               → depth >= 16 positions; normal for long searches

SINGULAR EXTENSIONS
  • Many double/triple extensions   → Possibly extending too eagerly; check extension caps
  • High multi-cut rate             → Good sign: many false singular candidates correctly pruned
  • High negative extension rate    → cut-node recognition is working

PROBCUT
  • Low cutoff/tried ratio          → probcut_beta margin too tight; few moves exceed it

QSEARCH
  • Stand-pat > 80% of QS nodes    → Position set is tactical (normal)
  • SEE pruning > LMP               → Noisy positions; expected
════════════════════════════════════════════════════════════════════════════
"""

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze search_stats.json")
    parser.add_argument("input", nargs="?", default="search_stats.json",
                        help="Path to newline-delimited JSON stats file")
    parser.add_argument("--out", default="search_report.png",
                        help="Output image path (default: search_report.png)")
    parser.add_argument("--guide", action="store_true",
                        help="Print interpretation guide and exit")
    args = parser.parse_args()

    if args.guide:
        print(GUIDE)
        sys.exit(0)

    s = load(args.input)
    plot(s, args.out)
    print(GUIDE)
