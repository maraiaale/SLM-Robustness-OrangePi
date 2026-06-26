#!/bin/bash

set -u
set -o pipefail

BASE_DIR="$HOME/benchmark_slm"
BENCH_SCRIPT="$BASE_DIR/run_benchmark_server_en.sh"

cd "$BASE_DIR" || exit 1

if [ ! -x "$BENCH_SCRIPT" ]; then
    echo "ERRORE: benchmark non eseguibile: $BENCH_SCRIPT"
    echo "Esegui: chmod +x $BENCH_SCRIPT"
    exit 1
fi

ACTIVE_MODELS=$(
    grep -E '^[[:space:]]*run_all_conditions ' "$BENCH_SCRIPT" |
    awk -F'"' '{print $2}'
)

if [ -z "$ACTIVE_MODELS" ]; then
    echo "ERRORE: nessun modello attivo trovato in $BENCH_SCRIPT"
    grep -n 'run_all_conditions' "$BENCH_SCRIPT" || true
    exit 1
fi

OUT_BASE="$BASE_DIR/results_stress_final"
mkdir -p "$OUT_BASE"

STRESS_TESTS=(
  "cpu_w5|--cpu 5"
  "cache_w5|--cache 5"
  "memcpy_w5|--memcpy 5"
  "open_w5|--open 5"
  "itimer_w5|--itimer 5"
)

STRESS_PID=""
STRESS_PGID=""

cleanup_stress() {
    if [ -n "${STRESS_PGID:-}" ]; then
        echo "Fermo stress-ng PGID $STRESS_PGID..."

        kill -INT -- -"$STRESS_PGID" 2>/dev/null || true

        for _ in $(seq 1 60); do
            if [ -z "${STRESS_PID:-}" ] || ! kill -0 "$STRESS_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        if [ -n "${STRESS_PID:-}" ] && kill -0 "$STRESS_PID" 2>/dev/null; then
            echo "stress-ng non si è chiuso con SIGINT. Provo SIGTERM..."
            kill -TERM -- -"$STRESS_PGID" 2>/dev/null || true

            for _ in $(seq 1 30); do
                if ! kill -0 "$STRESS_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
        fi

        if [ -n "${STRESS_PID:-}" ] && kill -0 "$STRESS_PID" 2>/dev/null; then
            echo "stress-ng non si è chiuso con SIGTERM. Uso SIGKILL..."
            kill -KILL -- -"$STRESS_PGID" 2>/dev/null || true
        fi
    fi

    STRESS_PID=""
    STRESS_PGID=""
}

cleanup_all() {
    cleanup_stress

    echo "Controllo eventuali processi residui..."
    pgrep -a stress-ng || true
    pgrep -a llama-server || true
}

organize_results() {
    local tmp_dir="$1"
    local stressor_name="$2"
    local model
    local model_dir

    for model in $ACTIVE_MODELS; do
        model_dir="$OUT_BASE/$model/$stressor_name"
        mkdir -p "$model_dir"

        echo "Organizzo risultati: $model -> $model_dir"

        find "$tmp_dir" -maxdepth 1 -type f -name "*${model}*" -exec mv -t "$model_dir" {} +

        if [ -d "$tmp_dir/stress_logs" ]; then
            mkdir -p "$model_dir/stress_logs"
            cp -a "$tmp_dir/stress_logs"/. "$model_dir/stress_logs"/ 2>/dev/null || true
        fi
    done
}

trap 'cleanup_all; exit 130' INT TERM
trap 'cleanup_all' EXIT

echo "======================================"
echo "Stress benchmark finale"
echo "Modelli attivi:"
echo "$ACTIVE_MODELS"
echo "Output base: $OUT_BASE"
echo "======================================"

echo
echo "Controllo processi già attivi:"
pgrep -a stress-ng || true
pgrep -a llama-server || true
echo

for TEST in "${STRESS_TESTS[@]}"; do
    NAME="${TEST%%|*}"
    ARGS="${TEST#*|}"

    COND_DIR="$OUT_BASE/_tmp_${NAME}_$$"
    STRESS_LOG_DIR="$COND_DIR/stress_logs"
    STRESS_LOG="$STRESS_LOG_DIR/stress_${NAME}.log"

    rm -rf "$COND_DIR"
    mkdir -p "$STRESS_LOG_DIR"

    echo
    echo "######################################"
    echo "STRESSOR: $NAME"
    echo "MODELLI ATTIVI:"
    echo "$ACTIVE_MODELS"
    echo "Comando: stress-ng $ARGS --metrics-brief"
    echo "Risultati temporanei: $COND_DIR"
    echo "Log stress-ng: $STRESS_LOG"
    echo "######################################"

    setsid stress-ng $ARGS --metrics-brief > "$STRESS_LOG" 2>&1 &

    STRESS_PID=$!
    sleep 1
    STRESS_PGID=$(ps -o pgid= -p "$STRESS_PID" | tr -d ' ')

    echo "stress-ng PID:  $STRESS_PID"
    echo "stress-ng PGID: $STRESS_PGID"

    RESULTS_DIR="$COND_DIR" "$BENCH_SCRIPT"
    BENCH_STATUS=$?

    echo
    echo "Benchmark terminato per stressor $NAME"
    echo "Fermo stress-ng..."

    cleanup_stress

    if [ "$BENCH_STATUS" -ne 0 ]; then
        echo "ERRORE: benchmark fallito per stressor $NAME"
        echo "Directory temporanea conservata: $COND_DIR"
        exit "$BENCH_STATUS"
    fi

    echo
    echo "Ultime righe log stress-ng per $NAME:"
    tail -n 30 "$STRESS_LOG" || true

    organize_results "$COND_DIR" "$NAME"

    rm -rf "$COND_DIR"

    echo
    echo "Condizione completata: $NAME"

    sleep 10
done

trap - EXIT

echo
echo "======================================"
echo "Stress benchmark finale completato"
echo "Modelli:"
echo "$ACTIVE_MODELS"
echo "Risultati in: $OUT_BASE"
echo "======================================"