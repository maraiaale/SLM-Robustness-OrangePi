from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter


# =========================
# PATH
# =========================

BASELINE_DIR = Path(r"C:\Users\alema\Desktop\npu_pinning\baseline_qwen_en_npu")

STRESS_DIR = Path(
    r"C:\Users\alema\Desktop\npu_pinning\results_stress_npu_warm_allmodels_temp_0_0_0_5_1_0"
)

OUT_DIR = Path(r"C:\Users\alema\Desktop\npu_pinning\grafici_stress")

PNG_DIR = OUT_DIR / "png"
PDF_DIR = OUT_DIR / "pdf"
CSV_DIR = OUT_DIR / "csv"

for d in [OUT_DIR, PNG_DIR, PDF_DIR, CSV_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================
# CONFIG
# =========================

MODELS = [
    ("qwen2.5-0.5b", "Qwen2.5-0.5B"),
    ("qwen2.5-1.5b", "Qwen2.5-1.5B"),
    ("qwen2.5-3b", "Qwen2.5-3B"),
    ("qwen2.5-7b", "Qwen2.5-7B"),
]

TEMPS = ["0.0", "0.5", "1.0"]

STRESSORS = [
    ("no_stress", "no stress"),
    ("cpu_w5", "cpu"),
    ("cache_w5", "cache"),
    ("memcpy_w5", "memcpy"),
    ("open_w5", "open"),
    ("itimer_w5", "itimer"),
]

MODEL_COLORS = {
    "Qwen2.5-0.5B": "#2ECC71",  # verde
    "Qwen2.5-1.5B": "#3498DB",  # blu
    "Qwen2.5-3B": "#E74C3C",    # rosso
    "Qwen2.5-7B": "#9B59B6",    # viola
}

METRICS = {
    "tgs": {
        "label": "TGS (token/s)",
        "output": "tgs",
        "candidates": [
            "tgs_tokens_s",
            "tgs",
            "tgs_s",
            "TGS",
            "TGS_s",
            "tokens_per_second",
            "tok_s",
            "token_s",
            "generation_speed",
            "generation_speed_tok_s",
        ],
        "log_scale": False,
        "symlog_scale": False,
        "kind": "box",
    },
    "ttft": {
        "label": "TTFT (s)",
        "output": "ttft",
        "candidates": [
            "ttft_s",
            "ttft",
            "TTFT",
            "time_to_first_token",
            "time_to_first_token_s",
            "first_token_latency_s",
        ],
        "log_scale": True,
        "symlog_scale": False,
        "kind": "box",
    },
    "total_latency": {
        "label": "Total latency (s)",
        "output": "total_latency",
        "candidates": [
            "total_latency_s",
            "latency_s",
            "total_latency",
            "total_time_s",
            "elapsed_s",
            "duration_s",
        ],
        "log_scale": True,
        "symlog_scale": False,
        "kind": "box",
    },
    "rss": {
        "label": "RSS peak (MB)",
        "output": "rss_peak",
        "candidates": [
            "rss_peak_mb",
            "rss_mb",
            "memory_mb",
            "ram_mb",
            "RSS",
            "RSS_MB",
            "resident_set_size_mb",
        ],
        "log_scale": False,
        "symlog_scale": True,
        "kind": "box",
    },
    "cpu_usage": {
        "label": "CPU media host (% processo, multi-core)",
        "output": "cpu_host_usage",
        "candidates": [
            "cpu_avg_percent",
            "cpu_percent",
            "cpu_avg",
            "cpu_usage",
            "cpu_usage_percent",
            "CPU",
            "CPU_percent",
        ],
        "log_scale": False,
        "symlog_scale": True,
        "kind": "box",
    },
    "board_temp": {
        "label": "Temperatura media board (°C)",
        "output": "board_temp",
        "candidates": [
            "temp_avg_c",
            "temperature_avg_c",
            "board_temp_c",
            "temp_c",
            "avg_temp_c",
            "temperature_c",
        ],
        "log_scale": False,
        "symlog_scale": False,
        "kind": "box",
    },
    "pass1": {
        "label": "Pass@1 Rate (%)",
        "output": "pass1_rate",
        "candidates": [
            "pass1",
        ],
        "log_scale": False,
        "symlog_scale": False,
        "kind": "bar_rate",
    },
}


# =========================
# HELPERS
# =========================

def safe_temp(temp: str) -> str:
    return temp.replace(".", "_")


def clean_log_ticks(ax, values):
    """
    Rende leggibili i tick logaritmici.
    Invece di mostrare 10^0, 2x10^0, ecc.,
    mostra valori numerici: 1, 2, 3, 5, 10, 20, 50...
    """
    values = pd.to_numeric(values, errors="coerce").dropna()
    values = values[values > 0]

    if values.empty:
        return

    ymin = max(values.min() * 0.75, 0.3)
    ymax = values.max() * 1.25

    tick_candidates = np.array([
        0.3, 0.5,
        1, 2, 3, 5,
        10, 20, 30, 50,
        100, 200, 300, 500,
        1000
    ])

    ticks = tick_candidates[
        (tick_candidates >= ymin) &
        (tick_candidates <= ymax)
    ]

    if len(ticks) == 0:
        return

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(ticks)

    def fmt(y, _):
        if y >= 10:
            return f"{int(y)}"
        return f"{y:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())


def clean_symlog_ticks(ax, values, metric_key):
    """
    Scala symlog per metriche con valori piccoli schiacciati
    da poche condizioni molto grandi.
    Serve per mantenere un solo grafico, ma rendere leggibili
    sia le condizioni basse sia quelle alte.
    """
    values = pd.to_numeric(values, errors="coerce").dropna()

    if values.empty:
        return

    max_value = values.max()
    min_value = values.min()

    if metric_key == "rss":
        # RSS: valori bassi circa 140 MB, open/itimer molto più alti.
        # Con linthresh=160 la zona intorno alla baseline resta leggibile,
        # mentre i valori alti vengono compressi.
        ax.set_yscale("symlog", linthresh=160, linscale=1.0)

        ymin = max(min_value - 10, 120)
        ymax = max_value + 30

        ticks = [120, 130, 140, 150, 160, 200, 250, 300, 350, 400, 450]

        ax.set_ylim(ymin, ymax)
        ax.set_yticks([t for t in ticks if ymin <= t <= ymax])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}"))

    elif metric_key == "cpu_usage":
        # CPU host usage: molte condizioni sono circa zero,
        # open/itimer possono essere molto più alte.
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.0)

        ymin = -0.05
        ymax = max_value * 1.25 if max_value > 0 else 1

        ticks = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 40]

        ax.set_ylim(ymin, ymax)
        ax.set_yticks([t for t in ticks if ymin <= t <= ymax])

        def fmt(y, _):
            if abs(y) < 1e-12:
                return "0"
            if y < 1:
                return f"{y:g}"
            return f"{int(y)}"

        ax.yaxis.set_major_formatter(FuncFormatter(fmt))

    ax.yaxis.set_minor_formatter(NullFormatter())


def find_first_existing_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_baseline_csv(model_file: str, temp: str):
    expected = f"npu_results_{model_file}_temp_{temp}_warm.csv"

    direct = BASELINE_DIR / expected
    if direct.exists():
        return direct

    matches = list(BASELINE_DIR.rglob(expected))
    if matches:
        return matches[0]

    loose_matches = list(BASELINE_DIR.rglob(f"*{model_file}*temp_{temp}_warm.csv"))
    if loose_matches:
        return loose_matches[0]

    return None


def find_stress_csv(stressor: str, model_file: str, temp: str):
    expected = f"npu_stress_{model_file}_temp_{temp}_warm.csv"

    direct = STRESS_DIR / stressor / model_file / expected
    if direct.exists():
        return direct

    matches = list((STRESS_DIR / stressor).rglob(expected))
    if matches:
        return matches[0]

    loose_matches = list((STRESS_DIR / stressor).rglob(f"*{model_file}*temp_{temp}_warm.csv"))
    if loose_matches:
        return loose_matches[0]

    return None


def load_all_data():
    rows = []
    missing = []

    for temp in TEMPS:
        for model_file, model_label in MODELS:

            # Baseline / no stress
            baseline_path = find_baseline_csv(model_file, temp)

            if baseline_path is None:
                missing.append({
                    "type": "baseline",
                    "stressor": "no_stress",
                    "model": model_file,
                    "temperature": temp,
                    "expected": f"npu_results_{model_file}_temp_{temp}_warm.csv",
                })
            else:
                try:
                    df = pd.read_csv(baseline_path)
                    df["condition"] = "no_stress"
                    df["condition_label"] = "no stress"
                    df["model_file"] = model_file
                    df["model_label"] = model_label
                    df["temperature"] = temp
                    df["source_file"] = str(baseline_path)
                    rows.append(df)
                except Exception as e:
                    missing.append({
                        "type": "baseline_read_error",
                        "stressor": "no_stress",
                        "model": model_file,
                        "temperature": temp,
                        "expected": str(baseline_path),
                        "error": str(e),
                    })

            # Stress
            for stressor, stressor_label in STRESSORS:
                if stressor == "no_stress":
                    continue

                stress_path = find_stress_csv(stressor, model_file, temp)

                if stress_path is None:
                    missing.append({
                        "type": "stress",
                        "stressor": stressor,
                        "model": model_file,
                        "temperature": temp,
                        "expected": f"{stressor}\\{model_file}\\npu_stress_{model_file}_temp_{temp}_warm.csv",
                    })
                    continue

                try:
                    df = pd.read_csv(stress_path)
                    df["condition"] = stressor
                    df["condition_label"] = stressor_label
                    df["model_file"] = model_file
                    df["model_label"] = model_label
                    df["temperature"] = temp
                    df["source_file"] = str(stress_path)
                    rows.append(df)
                except Exception as e:
                    missing.append({
                        "type": "stress_read_error",
                        "stressor": stressor,
                        "model": model_file,
                        "temperature": temp,
                        "expected": str(stress_path),
                        "error": str(e),
                    })

    if not rows:
        raise FileNotFoundError(
            "Nessun CSV trovato. Controlla BASELINE_DIR e STRESS_DIR nello script."
        )

    all_df = pd.concat(rows, ignore_index=True)

    missing_df = pd.DataFrame(missing)
    missing_path = CSV_DIR / "npu_stress_missing_files.csv"
    missing_df.to_csv(missing_path, index=False)

    print(f"\nDati caricati: {len(all_df)} righe")
    print(f"File mancanti o errori: {len(missing)}")
    print(f"Log file mancanti: {missing_path}")

    return all_df, missing_df


def build_metric_long_df(all_df: pd.DataFrame):
    metric_frames = []

    for metric_key, cfg in METRICS.items():
        col = find_first_existing_column(all_df, cfg["candidates"])

        if col is None:
            print(f"[ATTENZIONE] Nessuna colonna trovata per metrica: {metric_key}")
            continue

        tmp = all_df[
            [
                "condition",
                "condition_label",
                "model_file",
                "model_label",
                "temperature",
                "source_file",
                col,
            ]
        ].copy()

        tmp = tmp.rename(columns={col: "value"})
        tmp["metric"] = metric_key
        tmp["metric_label"] = cfg["label"]
        tmp["metric_column"] = col
        tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")

        metric_frames.append(tmp)

        print(f"Metrica {metric_key}: colonna usata = {col}")

    if not metric_frames:
        raise ValueError("Nessuna metrica disponibile nei CSV.")

    long_df = pd.concat(metric_frames, ignore_index=True)
    long_path = CSV_DIR / "npu_stress_all_metrics_long.csv"
    long_df.to_csv(long_path, index=False)

    print(f"CSV lungo salvato: {long_path}")

    return long_df


def make_summary(long_df: pd.DataFrame):
    summary = (
        long_df
        .groupby(
            ["metric", "temperature", "condition", "condition_label", "model_label"],
            dropna=False
        )["value"]
        .agg(
            runs="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max"
        )
        .reset_index()
    )

    summary_path = CSV_DIR / "npu_stress_summary_stats.csv"
    summary.to_csv(summary_path, index=False)

    print(f"CSV summary salvato: {summary_path}")

    return summary


def apply_order(df: pd.DataFrame):
    condition_order = [label for _, label in STRESSORS]
    model_order = [label for _, label in MODELS]

    df = df.copy()

    df["condition_label"] = pd.Categorical(
        df["condition_label"],
        categories=condition_order,
        ordered=True,
    )

    df["model_label"] = pd.Categorical(
        df["model_label"],
        categories=model_order,
        ordered=True,
    )

    return df.sort_values(["condition_label", "model_label"])


# =========================
# PLOTS
# =========================

def plot_box_metric(
    metric_df: pd.DataFrame,
    metric_key: str,
    cfg: dict,
    temp: str,
):
    metric_df = apply_order(metric_df)

    condition_labels = [label for _, label in STRESSORS]
    model_labels = [label for _, label in MODELS]

    fig, ax = plt.subplots(figsize=(18, 7.5))

    group_spacing = 1.75
    x_centers = np.arange(len(condition_labels)) * group_spacing

    offsets = np.linspace(-0.36, 0.36, len(model_labels))
    box_width = 0.19

    data = []
    positions = []
    colors = []

    for i, condition in enumerate(condition_labels):
        for j, model in enumerate(model_labels):
            subset = metric_df[
                (metric_df["condition_label"] == condition) &
                (metric_df["model_label"] == model)
            ]["value"].dropna()

            if cfg["log_scale"]:
                subset = subset[subset > 0]

            data.append(subset)
            positions.append(x_centers[i] + offsets[j])
            colors.append(MODEL_COLORS[model])

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
        flierprops=dict(marker="o", markersize=3.2, markeredgewidth=0.6),
    )

    for i, (patch, color) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(0.90)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

        # whiskers: linee verticali del boxplot
        bp["whiskers"][2 * i].set_color(color)
        bp["whiskers"][2 * i + 1].set_color(color)
        bp["whiskers"][2 * i].set_linewidth(1.2)
        bp["whiskers"][2 * i + 1].set_linewidth(1.2)

        # caps: i "piedini" sopra e sotto
        bp["caps"][2 * i].set_color(color)
        bp["caps"][2 * i + 1].set_color(color)
        bp["caps"][2 * i].set_linewidth(1.2)
        bp["caps"][2 * i + 1].set_linewidth(1.2)

    for flier, color in zip(bp["fliers"], colors):
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)
        flier.set_alpha(0.90)

    if cfg["log_scale"]:
        ax.set_yscale("log")
        clean_log_ticks(ax, metric_df["value"])

    if cfg.get("symlog_scale", False):
        clean_symlog_ticks(ax, metric_df["value"], metric_key)

    ax.set_ylabel(cfg["label"], fontsize=16, fontweight="bold")
    #ax.set_xlabel("Condizione di stress", fontsize=14)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(condition_labels, rotation=35, ha="right", fontsize=14, fontweight="bold")

    ax.tick_params(axis="y", labelsize=12)

    # Separatore verticale tra gruppi
    for i in range(1, len(condition_labels)):
        sep_x = (x_centers[i - 1] + x_centers[i]) / 2
        ax.axvline(
            sep_x,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.55,
        )

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=MODEL_COLORS[label],
            edgecolor=MODEL_COLORS[label],
            alpha=0.90,
            label=label,
        )
        for label in model_labels
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        fontsize=11,
        borderaxespad=0.0,
    )

    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_name = f"npu_stress_{cfg['output']}_temp_{safe_temp(temp)}"

    png_path = PNG_DIR / f"{out_name}.png"
    pdf_path = PDF_DIR / f"{out_name}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Salvato: {pdf_path}")


def plot_pass1_rate(metric_df: pd.DataFrame, cfg: dict, temp: str):
    metric_df = apply_order(metric_df)

    condition_labels = [label for _, label in STRESSORS]
    model_labels = [label for _, label in MODELS]

    # Pass@1 rate = mean(pass1) * 100
    rate_df = (
        metric_df
        .groupby(["condition_label", "model_label"])["value"]
        .mean()
        .reset_index()
    )

    rate_df["pass_rate_percent"] = rate_df["value"] * 100

    fig, ax = plt.subplots(figsize=(18, 7.0))

    group_spacing = 1.75
    x_centers = np.arange(len(condition_labels)) * group_spacing

    offsets = np.linspace(-0.36, 0.36, len(model_labels))
    bar_width = 0.19

    for j, model in enumerate(model_labels):
        values = []

        for condition in condition_labels:
            row = rate_df[
                (rate_df["condition_label"] == condition) &
                (rate_df["model_label"] == model)
            ]

            if row.empty:
                values.append(np.nan)
            else:
                values.append(float(row["pass_rate_percent"].iloc[0]))

        bars = ax.bar(
            x_centers + offsets[j],
            values,
            width=bar_width,
            color=MODEL_COLORS[model],
            alpha=0.90,
            edgecolor="black",
            linewidth=0.8,
            label=model,
        )

        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_ylabel("Pass@1 Rate (%)", fontsize=16, fontweight="bold")
    #ax.set_xlabel("Condizione di stress", fontsize=14)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(condition_labels, rotation=35, ha="right", fontsize=14, fontweight="bold")

    ax.tick_params(axis="y", labelsize=12)

    ax.set_ylim(0, 110)

    for i in range(1, len(condition_labels)):
        sep_x = (x_centers[i - 1] + x_centers[i]) / 2
        ax.axvline(
            sep_x,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.55,
        )

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        fontsize=11,
        borderaxespad=0.0,
    )

    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_name = f"npu_stress_{cfg['output']}_temp_{safe_temp(temp)}"

    png_path = PNG_DIR / f"{out_name}.png"
    pdf_path = PDF_DIR / f"{out_name}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Salvato: {pdf_path}")


def generate_plots(long_df: pd.DataFrame):
    for temp in TEMPS:
        for metric_key, cfg in METRICS.items():
            metric_df = long_df[
                (long_df["temperature"] == temp) &
                (long_df["metric"] == metric_key)
            ].copy()

            if metric_df.empty:
                print(f"[SKIP] Nessun dato per {metric_key} temp {temp}")
                continue

            if cfg["kind"] == "bar_rate":
                plot_pass1_rate(metric_df, cfg, temp)
            else:
                plot_box_metric(metric_df, metric_key, cfg, temp)


def main():
    print("=== NPU stress plot generator ===")
    print(f"Baseline dir: {BASELINE_DIR}")
    print(f"Stress dir:   {STRESS_DIR}")
    print(f"Output dir:   {OUT_DIR}")

    all_df, missing_df = load_all_data()

    raw_path = CSV_DIR / "npu_stress_raw_loaded.csv"
    all_df.to_csv(raw_path, index=False)
    print(f"CSV raw salvato: {raw_path}")

    long_df = build_metric_long_df(all_df)
    summary = make_summary(long_df)

    generate_plots(long_df)

    print("\nFatto.")
    print(f"Grafici PNG: {PNG_DIR}")
    print(f"Grafici PDF: {PDF_DIR}")
    print(f"CSV:         {CSV_DIR}")


if __name__ == "__main__":
    main()