import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# PATH
# =========================

BASE_DIR = Path(r"C:\Users\alema\Desktop\cpu\baseline_qwen_en_cpu")
CSV_DIR = BASE_DIR / "results_server_en_qwen" / "results_server_en_qwen"
OUT_DIR = Path(r"C:\Users\alema\Desktop\cpu\baseline_qwen_en_cpu\grafici_baseline_cpu")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# CONFIG
# =========================

MODELS = [
    ("qwen2.5-0.5b", "Qwen2.5-0.5B"),
    ("qwen2.5-1.5b", "Qwen2.5-1.5B"),
    ("qwen2.5-3b", "Qwen2.5-3B"),
    ("qwen2.5-7b", "Qwen2.5-7B"),
]

MODES = ["cold", "warm"]

COLORS = {
    "cold": "#3498DB",  # blu
    "warm": "#E74C3C",  # rosso
}


# =========================
# DATA LOADING
# =========================

def find_ttft_column(df):
    candidates = [
        "ttft_s",
        "ttft",
        "TTFT",
        "time_to_first_token",
        "time_to_first_token_s",
        "first_token_latency_s",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Colonna TTFT non trovata. Colonne disponibili: "
        + ", ".join(df.columns)
    )


def load_temperature_data(temp):
    dfs = []
    ttft_col_found = None

    for model_file, model_label in MODELS:
        for mode in MODES:
            filename = f"server_results_{model_file}_temp_{temp}_{mode}.csv"
            path = CSV_DIR / filename

            if not path.exists():
                print(f"[ATTENZIONE] File mancante: {path}")
                continue

            df = pd.read_csv(path)

            ttft_col = find_ttft_column(df)
            ttft_col_found = ttft_col

            df[ttft_col] = pd.to_numeric(df[ttft_col], errors="coerce")

            df["model_label"] = model_label
            df["mode"] = mode
            df["temperature"] = float(temp)
            df["source_file"] = filename

            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"Nessun CSV trovato per temperatura {temp}. "
            f"Cartella cercata: {CSV_DIR}"
        )

    out = pd.concat(dfs, ignore_index=True)

    return out, ttft_col_found


def summarize_ttft(df, ttft_col):
    summary = (
        df.groupby(["model_label", "mode"], observed=False)[ttft_col]
        .agg(
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )

    model_order = [label for _, label in MODELS]

    summary["model_label"] = pd.Categorical(
        summary["model_label"],
        categories=model_order,
        ordered=True,
    )

    summary["mode"] = pd.Categorical(
        summary["mode"],
        categories=MODES,
        ordered=True,
    )

    summary = summary.sort_values(["model_label", "mode"]).reset_index(drop=True)

    return summary


# =========================
# PLOT
# =========================

def plot_ttft(df, ttft_col, temp):
    fig, ax = plt.subplots(figsize=(8.8, 5.0))

    group_spacing = 0.90
    box_offset = 0.15
    box_width = 0.25

    model_labels = [label for _, label in MODELS]
    x_centers = np.arange(len(model_labels)) * group_spacing

    positions = []
    data = []
    box_colors = []

    for i, model in enumerate(model_labels):
        for mode, offset in zip(MODES, [-box_offset, box_offset]):
            subset = df[
                (df["model_label"] == model) &
                (df["mode"] == mode)
            ][ttft_col].dropna()

            positions.append(x_centers[i] + offset)
            data.append(subset)
            box_colors.append(COLORS[mode])

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=True,
        showmeans=True,
        meanline=True,
        medianprops=dict(color="none", linewidth=0),
        meanprops=dict(color="black", linewidth=0.5, linestyle="-"),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3.5, markeredgewidth=0.8),
    )

    for i, (patch, color) in enumerate(zip(bp["boxes"], box_colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

        # whiskers: linee verticali
        bp["whiskers"][2 * i].set_color(color)
        bp["whiskers"][2 * i + 1].set_color(color)
        bp["whiskers"][2 * i].set_linewidth(1.2)
        bp["whiskers"][2 * i + 1].set_linewidth(1.2)

        # caps: "piedini" sopra e sotto
        bp["caps"][2 * i].set_color(color)
        bp["caps"][2 * i + 1].set_color(color)
        bp["caps"][2 * i].set_linewidth(1.2)
        bp["caps"][2 * i + 1].set_linewidth(1.2)

    for flier, color in zip(bp["fliers"], box_colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
        flier.set_alpha(0.90)

    ax.set_ylabel("TTFT (s)", fontsize=16, fontweight="bold")

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        model_labels,
        fontsize=11,
        fontweight="bold",
    )

    ax.tick_params(axis="y", labelsize=12)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS["cold"],
            edgecolor=COLORS["cold"],
            alpha=0.65,
            label="cold",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS["warm"],
            edgecolor=COLORS["warm"],
            alpha=0.65,
            label="warm",
        ),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        fontsize=10,
        borderaxespad=0.0,
    )

    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor("white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_y = df[ttft_col].dropna().max()
    if pd.notna(max_y) and max_y > 0:
        ax.set_ylim(0, max_y * 1.18)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    safe_temp = temp.replace(".", "_")

    output_png = OUT_DIR / f"cpu_boxplot_ttft_temp_{safe_temp}.png"
    output_pdf = OUT_DIR / f"cpu_boxplot_ttft_temp_{safe_temp}.pdf"
    concat_csv = OUT_DIR / f"cpu_concat_ttft_temp_{safe_temp}.csv"
    summary_csv = OUT_DIR / f"cpu_summary_ttft_temp_{safe_temp}.csv"

    df.to_csv(concat_csv, index=False)

    summary = summarize_ttft(df, ttft_col)
    summary.to_csv(summary_csv, index=False)

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"CSV concatenato salvato: {concat_csv}")
    print(f"CSV riepilogo salvato: {summary_csv}")
    print(f"Grafico salvato: {output_png}")
    print(f"Grafico salvato: {output_pdf}")

    print("\nStatistiche TTFT:")
    print(summary.round(4))


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--temp",
        required=True,
        choices=["0.0", "0.5", "1.0"],
        help="Temperatura da analizzare: 0.0, 0.5 oppure 1.0",
    )

    args = parser.parse_args()

    df, ttft_col = load_temperature_data(args.temp)

    print(f"Colonna TTFT usata: {ttft_col}")
    print("Righe per modello/modalità:")
    print(df[["model_label", "mode"]].value_counts().sort_index())

    plot_ttft(df, ttft_col, args.temp)


if __name__ == "__main__":
    main()