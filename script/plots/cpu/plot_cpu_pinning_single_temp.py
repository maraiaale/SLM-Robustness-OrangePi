from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter


# =========================
# PATH
# =========================

BASELINE_DIR = Path(
    r"C:\Users\alema\Desktop\cpu\baseline_qwen_en_cpu"
)

PINNING_DIR = Path(
    r"C:\Users\alema\Desktop\cpu\results_cpu_pinning\results_cpu_pinning"
)

OUT_DIR = Path(
    r"C:\Users\alema\Desktop\cpu\grafici_cpu_pinning"
)

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

CONDITIONS = [
    ("no_stress", "no stress"),
    ("cpu_w5", "cpu"),
    ("cache_w5", "cache"),
    ("memcpy_w5", "memcpy"),
    ("open_w5", "open"),
    ("itimer_w5", "itimer"),
]

MODEL_COLORS = {
    "Qwen2.5-0.5B": "#2ECC71",
    "Qwen2.5-1.5B": "#3498DB",
    "Qwen2.5-3B": "#E74C3C",
    "Qwen2.5-7B": "#9B59B6",
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
        "scale": "linear",
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
        "scale": "log",
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
        "scale": "log",
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
        "scale": "linear",
        "kind": "box",
    },
    "cpu_usage": {
        "label": "CPU media (% processo, multi-core)",
        "output": "cpu_usage",
        "candidates": [
            "cpu_avg_percent",
            "cpu_percent",
            "cpu_avg",
            "cpu_usage",
            "cpu_usage_percent",
            "CPU",
            "CPU_percent",
        ],
        "scale": "linear",
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
        "scale": "linear",
        "kind": "box",
    },
    "pass1": {
        "label": "Pass@1 Rate (%)",
        "output": "pass1_rate",
        "candidates": [
            "pass1",
        ],
        "scale": "linear",
        "kind": "bar_rate",
    },
}


# =========================
# HELPERS
# =========================

def safe_temp(temp: str) -> str:
    return temp.replace(".", "_")


def find_first_existing_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def is_bad_candidate(path: Path):
    name = path.name.lower()
    full = str(path).lower()

    bad_words = [
        "summary",
        "missing",
        "all_metrics",
        "raw_loaded",
        "long",
        "stats",
        "grafici",
        "grafico",
        "concat",
    ]

    return any(w in name or w in full for w in bad_words)


def has_any_metric_column(csv_path: Path):
    try:
        df = pd.read_csv(csv_path, nrows=3)
    except Exception:
        return False

    cols = set(df.columns)

    for cfg in METRICS.values():
        for c in cfg["candidates"]:
            if c in cols:
                return True

    return False


def find_baseline_csv(model_file: str, temp: str):
    """
    Baseline CPU: cerca ricorsivamente i CSV warm.
    Di solito il prefisso è server_results_...
    """
    patterns = [
        f"server_results_{model_file}_temp_{temp}_warm.csv",
        f"*{model_file}*temp_{temp}*warm*.csv",
        f"*{model_file}*temp_{safe_temp(temp)}*warm*.csv",
        f"*{model_file}*{temp}*.csv",
        f"*{model_file}*{safe_temp(temp)}*.csv",
    ]

    candidates = []

    for pattern in patterns:
        candidates.extend(BASELINE_DIR.rglob(pattern))

    candidates = sorted(set(candidates))
    candidates = [p for p in candidates if p.is_file()]
    candidates = [p for p in candidates if not is_bad_candidate(p)]
    candidates = [p for p in candidates if has_any_metric_column(p)]

    if not candidates:
        return None

    warm_candidates = [p for p in candidates if "warm" in p.name.lower()]
    if warm_candidates:
        return sorted(warm_candidates)[0]

    return sorted(candidates)[0]


def find_pinning_csv(stressor: str, model_file: str, temp: str):
    """
    Pinning CPU: struttura:
    results_cpu_pinning/results_cpu_pinning/<stressor>/<model>/server_stress_<model>_temp_<temp>_warm.csv
    """
    expected = f"server_stress_{model_file}_temp_{temp}_warm.csv"

    direct = PINNING_DIR / stressor / model_file / expected
    if direct.exists():
        return direct

    stressor_root = PINNING_DIR / stressor

    if stressor_root.exists():
        matches = list(stressor_root.rglob(expected))
        if matches:
            return sorted(matches)[0]

        loose_matches = list(stressor_root.rglob(f"*{model_file}*temp_{temp}*warm*.csv"))
        loose_matches += list(stressor_root.rglob(f"*{model_file}*temp_{safe_temp(temp)}*warm*.csv"))
        loose_matches += list(stressor_root.rglob(f"*{model_file}*{temp}*.csv"))
        loose_matches += list(stressor_root.rglob(f"*{model_file}*{safe_temp(temp)}*.csv"))

        loose_matches = sorted(set(loose_matches))
        loose_matches = [p for p in loose_matches if p.is_file()]
        loose_matches = [p for p in loose_matches if not is_bad_candidate(p)]
        loose_matches = [p for p in loose_matches if has_any_metric_column(p)]

        if loose_matches:
            return loose_matches[0]

    return None


def load_all_data():
    rows = []
    missing = []

    for temp in TEMPS:
        for model_file, model_label in MODELS:

            # baseline / no stress
            baseline_path = find_baseline_csv(model_file, temp)

            if baseline_path is None:
                missing.append({
                    "type": "baseline",
                    "condition": "no_stress",
                    "model": model_file,
                    "temperature": temp,
                    "expected": f"server_results_{model_file}_temp_{temp}_warm.csv",
                    "error": "not found",
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
                        "condition": "no_stress",
                        "model": model_file,
                        "temperature": temp,
                        "expected": str(baseline_path),
                        "error": str(e),
                    })

            # pinning conditions
            for condition, condition_label in CONDITIONS:
                if condition == "no_stress":
                    continue

                pinning_path = find_pinning_csv(condition, model_file, temp)

                if pinning_path is None:
                    missing.append({
                        "type": "pinning",
                        "condition": condition,
                        "model": model_file,
                        "temperature": temp,
                        "expected": f"{condition}\\{model_file}\\server_stress_{model_file}_temp_{temp}_warm.csv",
                        "error": "not found",
                    })
                    continue

                try:
                    df = pd.read_csv(pinning_path)
                    df["condition"] = condition
                    df["condition_label"] = condition_label
                    df["model_file"] = model_file
                    df["model_label"] = model_label
                    df["temperature"] = temp
                    df["source_file"] = str(pinning_path)
                    rows.append(df)
                except Exception as e:
                    missing.append({
                        "type": "pinning_read_error",
                        "condition": condition,
                        "model": model_file,
                        "temperature": temp,
                        "expected": str(pinning_path),
                        "error": str(e),
                    })

    if not rows:
        raise FileNotFoundError(
            "Nessun CSV trovato. Controlla BASELINE_DIR e PINNING_DIR."
        )

    all_df = pd.concat(rows, ignore_index=True)

    missing_df = pd.DataFrame(missing)
    missing_path = CSV_DIR / "cpu_pinning_missing_files.csv"
    missing_df.to_csv(missing_path, index=False)

    print(f"\nDati caricati: {len(all_df)} righe")
    print(f"File mancanti/errori: {len(missing)}")
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

    long_path = CSV_DIR / "cpu_pinning_all_metrics_long.csv"
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
            max="max",
        )
        .reset_index()
    )

    summary_path = CSV_DIR / "cpu_pinning_summary_stats.csv"
    summary.to_csv(summary_path, index=False)

    print(f"CSV summary salvato: {summary_path}")

    return summary


def apply_order(df: pd.DataFrame):
    condition_order = [label for _, label in CONDITIONS]
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


def set_scale(ax, scale_type, values):
    values = pd.to_numeric(values, errors="coerce").dropna()

    if values.empty:
        return

    if scale_type == "log":
        values = values[values > 0]

        if values.empty:
            return

        ax.set_yscale("log")

        # Prima era 0.3: tagliava via quasi tutti i TTFT piccoli.
        ymin = max(values.min() * 0.75, 0.04)
        ymax = values.max() * 1.35

        tick_candidates = np.array([
            0.04, 0.05, 0.07,
            0.1, 0.2, 0.3, 0.5,
            1, 2, 3, 5,
            10, 20, 30, 50,
            100, 200, 300, 500,
            1000,
        ])

        ticks = tick_candidates[
            (tick_candidates >= ymin) &
            (tick_candidates <= ymax)
        ]

        ax.set_ylim(ymin, ymax)

        if len(ticks) > 0:
            ax.set_yticks(ticks)

        def fmt(y, _):
            if y >= 10:
                return f"{int(y)}"
            return f"{y:g}"

        ax.yaxis.set_major_formatter(FuncFormatter(fmt))
        ax.yaxis.set_minor_formatter(NullFormatter())

    else:
        ymin = values.min()
        ymax = values.max()

        pad = (ymax - ymin) * 0.12 if ymax > ymin else max(
            1,
            ymax * 0.2 if ymax > 0 else 1,
        )

        lower = max(0, ymin - pad) if ymin >= 0 else ymin - pad
        upper = ymax + pad

        ax.set_ylim(lower, upper)
# =========================
# PLOTS
# =========================

def plot_box_metric(metric_df: pd.DataFrame, metric_key: str, cfg: dict, temp: str):
    metric_df = apply_order(metric_df)

    condition_labels = [label for _, label in CONDITIONS]
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

            if cfg["scale"] == "log":
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

    set_scale(ax, cfg["scale"], metric_df["value"])

    ax.set_ylabel(cfg["label"], fontsize=16, fontweight="bold")
    #ax.set_xlabel("Condizione", fontsize=14)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(condition_labels, rotation=35, ha="right", fontsize=14,  fontweight="bold")

    ax.tick_params(axis="y", labelsize=12)

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

    out_name = f"cpu_pinning_{cfg['output']}_temp_{safe_temp(temp)}"

    png_path = PNG_DIR / f"{out_name}.png"
    pdf_path = PDF_DIR / f"{out_name}.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Salvato: {pdf_path}")


def plot_pass1_rate(metric_df: pd.DataFrame, cfg: dict, temp: str):
    metric_df = apply_order(metric_df)

    condition_labels = [label for _, label in CONDITIONS]
    model_labels = [label for _, label in MODELS]

    rate_df = (
        metric_df
        .groupby(["condition_label", "model_label"], observed=False)["value"]
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
    #ax.set_xlabel("Condizione", fontsize=14)

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

    out_name = f"cpu_pinning_{cfg['output']}_temp_{safe_temp(temp)}"

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
    print("=== CPU pinning plot generator - singole temperature ===")
    print(f"Baseline dir: {BASELINE_DIR}")
    print(f"Pinning dir:  {PINNING_DIR}")
    print(f"Output dir:   {OUT_DIR}")

    all_df, missing_df = load_all_data()

    raw_path = CSV_DIR / "cpu_pinning_raw_loaded.csv"
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