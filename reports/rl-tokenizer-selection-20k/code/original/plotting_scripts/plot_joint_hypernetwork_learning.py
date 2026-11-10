"""Plot the fixed held-out tokenizer evaluations from the SmolLM2 joint run.

Run from the workspace root:
    python plotting_scripts/plot_joint_hypernetwork_learning.py
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, StrMethodFormatter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "zett/output_joint_smollm2_32k_zett_main/events.jsonl"
OUT = ROOT / "plots/joint_hypernetwork_learning"
BLUE = "#2F8FD0"
PURPLE = "#8657BC"
GRAY = "#73808E"
TEXT = "#333333"


def style():
    font_dir = Path.home() / ".local/share/fonts/manrope"
    if not font_dir.is_dir():
        raise FileNotFoundError(f"Manrope font not found: {font_dir}")
    for font in font_manager.findSystemFonts([str(font_dir)]):
        font_manager.fontManager.addfont(font)
    plt.rcParams.update({
        "font.family": "Manrope", "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": "#555B65", "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def read_evals():
    rows = []
    for line in SOURCE.read_text().splitlines():
        event = json.loads(line)
        if event.get("stage") not in ("initial_eval", "eval"):
            continue
        step = event.get("step", 0)
        results = event["results"]
        if len(results) != 2 or any(r["examples"] != 92 for r in results):
            raise ValueError(f"Unexpected held-out evaluation at step {step}")
        total_bytes = sum(r["bytes"] for r in results)
        generated = sum(r["generated_bpb"] * r["bytes"] for r in results) / total_bytes
        canonical = sum(r["canonical_bpb"] * r["bytes"] for r in results) / total_bytes
        rows.append({
            "step": step, "generated_bpb": generated,
            "canonical_bpb": canonical, "gap_bpb": generated - canonical,
            "tokenizer_1_gap_bpb": results[0]["generated_bpb"] - results[0]["canonical_bpb"],
            "tokenizer_2_gap_bpb": results[1]["generated_bpb"] - results[1]["canonical_bpb"],
            "tokenizer_1_generated_bpb": results[0]["generated_bpb"],
            "tokenizer_2_generated_bpb": results[1]["generated_bpb"],
            "tokenizer_1_canonical_bpb": results[0]["canonical_bpb"],
            "tokenizer_2_canonical_bpb": results[1]["canonical_bpb"],
            "tokenizer_1_bytes": results[0]["bytes"],
            "tokenizer_2_bytes": results[1]["bytes"],
        })
    rows.sort(key=lambda row: row["step"])
    if [r["step"] for r in rows] != [0, 100] + list(range(1000, 10001, 1000)):
        raise ValueError("Expected initial, step 100, and 1k-10k evaluations")
    return rows


def main():
    rows = read_evals()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUT.with_suffix(".json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "measurement": "BPB on two fixed unseen 32k Unigram tokenizers; 92 snippets each, "
                       "23 held-out text categories with four snippets per category",
        "aggregation": "Bytes-weighted average of both tokenizers' BPB values. "
                       "Each tokenizer's canonical comparison uses the same suffix bytes.",
        "training": "The hypernetwork and LM backbone both change during this run. "
                    "Learning rates increase through update 10000, when the run pauses.",
        "change_at_update_3000": "Minimum training-text length changed from 64 to 32 characters.",
        "scope": "These curves do not track 48k BPE tokenizers or the external 10 MB evaluation.",
    }, indent=2) + "\n")

    style()
    steps = [r["step"] for r in rows]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.4, 6.3), sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )
    fig.subplots_adjust(left=0.11, right=0.965, top=0.84, bottom=0.17, hspace=0.16)

    top.plot(steps, [r["generated_bpb"] for r in rows], color=BLUE,
             lw=2.5, marker="o", ms=3.8, label="Generated input (mean of 2)")
    top.plot(steps, [r["canonical_bpb"] for r in rows], color=GRAY,
             lw=2.0, ls="--", marker="o", ms=3.1, label="Canonical input (mean of 2)")
    top.set_ylabel("Held-out BPB  ↓")
    top.set_ylim(0.97, 1.51)
    top.legend(loc="upper right", frameon=False)

    bottom.plot(steps, [r["tokenizer_1_gap_bpb"] for r in rows],
                color=PURPLE, lw=1.0, alpha=0.48, marker="o", ms=2.5,
                label="Tokenizer 1")
    bottom.plot(steps, [r["tokenizer_2_gap_bpb"] for r in rows],
                color=BLUE, lw=1.0, alpha=0.48, marker="o", ms=2.5,
                label="Tokenizer 2")
    bottom.plot(steps, [r["gap_bpb"] for r in rows], color=TEXT,
                lw=2.4, marker="o", ms=3.7, label="Mean gap")
    bottom.axhline(0, color=GRAY, lw=0.85)
    bottom.set_ylim(-0.014, 0.43)
    bottom.set_ylabel("Generated − canonical BPB  ↓")
    bottom.set_xlabel("Joint-training updates after identity warmup")
    bottom.legend(loc="upper right", frameon=False, ncol=3)

    for ax in (top, bottom):
        ax.set_xlim(-130, 10500)
        ax.xaxis.set_major_locator(MultipleLocator(2000))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#DCE1E5", lw=0.55)
        ax.axvline(3000, color="#AAB2BB", lw=0.9, ls=":")
        ax.tick_params(length=3, width=0.6)

    top.annotate("1.443", (0, rows[0]["generated_bpb"]),
                 xytext=(550, 1.46), fontsize=8, color=BLUE,
                 arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.7})
    top.annotate("1.034", (10000, rows[-1]["generated_bpb"]),
                 xytext=(7550, 1.075), fontsize=8, color=BLUE,
                 arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.7})
    top.annotate("1.018", (10000, rows[-1]["canonical_bpb"]),
                 xytext=(7550, 0.985), fontsize=8, color=GRAY,
                 arrowprops={"arrowstyle": "-", "color": GRAY, "lw": 0.7})
    bottom.annotate("+0.380", (0, rows[0]["gap_bpb"]),
                    xytext=(610, 0.39), fontsize=8, color=TEXT,
                    arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 0.7})
    bottom.annotate("+0.016", (10000, rows[-1]["gap_bpb"]),
                    xytext=(7600, 0.072), fontsize=8, color=TEXT,
                    arrowprops={"arrowstyle": "-", "color": TEXT, "lw": 0.7})
    bottom.text(3080, 0.285, "Training text minimum\n64 → 32 chars",
                fontsize=7.5, color="#697480", va="top")
    fig.text(0.11, 0.935, "Transfer kept improving through the 10k ramp",
             fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.11, 0.887,
             "Two fixed unseen 32k tokenizers · 92 snippets each · both LM and hypernetwork trained",
             fontsize=8.2, ha="left", va="top", color="#56616D")
    fig.text(0.11, 0.035,
             "Update 10,000 is the end of the learning-rate ramp, not the end of the 190k-update plan.",
             fontsize=7.5, ha="left", color="#56616D")
    for suffix in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{suffix}"), dpi=220)
    plt.close(fig)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
