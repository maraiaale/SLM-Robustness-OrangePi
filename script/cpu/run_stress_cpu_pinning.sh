#!/bin/bash

set -u
set -o pipefail

BASE_DIR="${BASE_DIR:-$HOME/benchmark_slm}"
BENCHMARK_PY="${BENCHMARK_PY:-$BASE_DIR/benchmark_server.py}"

# Nota sperimentale: durante open_w5 stress-ng può saturare temporaneamente gli inode
# del filesystem principale (/). Per evitare che questo interferisca con la scrittura
# dei CSV/log del benchmark, i risultati vengono prodotti di default in /dev/shm
# e copiati in FINAL_RESULTS_BASE solo al termine di tutti gli stressor.
FINAL_RESULTS_BASE="${FINAL_RESULTS_BASE:-$BASE_DIR/results_cpu_pinning}"
RESULTS_BASE="${RESULTS_BASE:-/dev/shm/results_cpu_pinning}"
mkdir -p "$RESULTS_BASE"

LLAMA_SERVER="${LLAMA_SERVER:-/home/orangepi/llama.cpp/build/bin/llama-server}"
PPL_BIN="${PPL_BIN:-/home/orangepi/llama.cpp/build/bin/llama-perplexity}"
PPL_DATASET="${PPL_DATASET:-$BASE_DIR/ppl_wikipedia_dataset_en.txt}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SERVER_URL="http://${HOST}:${PORT}"

N_RUNS="${N_RUNS:-30}"
N_TOKENS="${N_TOKENS:-192}"
CTX="${CTX:-2048}"
THREADS="${THREADS:-$(nproc)}"

MODELS_LIST="${MODELS_LIST:-qwen2.5-0.5b qwen2.5-1.5b qwen2.5-3b qwen2.5-7b}"
TEMPERATURES_LIST="${TEMPERATURES_LIST:-0.0 0.5 1.0}"

PINNING="${PINNING:-1}"
SERVER_CPUS="${SERVER_CPUS:-4-7}"
STRESS_CPUS="${STRESS_CPUS:-0-3}"

PROMPT="Explain what Linux is.Answer only in English. In a single paragraph, describe its open-source nature, security, stability, portability, and some examples of its use. Write about 70 words, without using bullet points, and end with a complete sentence."
EXPECTED="operating system"

QWEN05="${QWEN05:-/home/orangepi/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
QWEN15="${QWEN15:-/home/orangepi/models/qwen2.5-1.5b.gguf}"
QWEN3="${QWEN3:-/home/orangepi/models/qwen2.5-3b.gguf}"
QWEN7="${QWEN7:-/home/orangepi/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}"
QWEN7_PART2="${QWEN7_PART2:-/home/orangepi/models/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf}"

STRESS_TESTS=(
  "cpu_w5|--cpu 5"
  "cache_w5|--cache 5"
  "memcpy_w5|--memcpy 5"
  "open_w5|--open 5"
  "itimer_w5|--itimer 5"
)

SERVER_PID=""
STRESS_PID=""
STRESS_PGID=""

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERRORE: file non trovato: $1"
        exit 1
    fi
}

check_requirements() {
    check_file "$LLAMA_SERVER"
    check_file "$PPL_BIN"
    check_file "$BENCHMARK_PY"
    check_file "$PPL_DATASET"
    check_file "$QWEN05"
    check_file "$QWEN15"
    check_file "$QWEN3"
    check_file "$QWEN7"
    check_file "$QWEN7_PART2"

    if ! command -v stress-ng >/dev/null 2>&1; then
        echo "ERRORE: stress-ng non trovato. Installa con: sudo apt install stress-ng"
        exit 1
    fi

    if ! command -v taskset >/dev/null 2>&1; then
        echo "ERRORE: taskset non trovato. Di solito è nel pacchetto util-linux."
        exit 1
    fi
}

model_path_for() {
    case "$1" in
        qwen2.5-0.5b) echo "$QWEN05" ;;
        qwen2.5-1.5b) echo "$QWEN15" ;;
        qwen2.5-3b)   echo "$QWEN3" ;;
        qwen2.5-7b)   echo "$QWEN7" ;;
        *)
            echo "ERRORE: modello non riconosciuto: $1" >&2
            return 1
            ;;
    esac
}

cleanup_server() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Spengo llama-server PID $SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true

        for _ in $(seq 1 30); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 0.3
        done

        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "llama-server non chiuso, SIGKILL."
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi

        wait "$SERVER_PID" 2>/dev/null || true
    fi

    SERVER_PID=""
}

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
    cleanup_server
    cleanup_stress

    echo "Controllo eventuali processi residui..."
    pgrep -a stress-ng || true
    pgrep -a llama-server || true
}

start_server() {
    local model_name="$1"
    local model_path="$2"
    local server_log="$3"

    pkill -f "llama-server" 2>/dev/null || true
    sleep 2

    echo "======================================"
    echo "Avvio llama-server CPU per: $model_name"
    echo "Modello: $model_path"
    echo "Log server: $server_log"
    echo "SERVER_CPUS: $SERVER_CPUS"
    echo "======================================"

    taskset -c "$SERVER_CPUS" "$LLAMA_SERVER" \
        -m "$model_path" \
        --host "$HOST" \
        --port "$PORT" \
        -c "$CTX" \
        -t "$THREADS" \
        > "$server_log" 2>&1 &

    SERVER_PID=$!

    echo "llama-server PID: $SERVER_PID"
    echo "Attendo che il server sia pronto..."

    local ready=0

    for _ in $(seq 1 180); do
        if curl -fsS "$SERVER_URL/health" >/dev/null 2>&1; then
            ready=1
            break
        fi

        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            break
        fi

        sleep 1
    done

    if [ "$ready" -ne 1 ]; then
        echo "ERRORE: llama-server non è diventato pronto."
        tail -n 120 "$server_log" || true
        cleanup_server
        return 1
    fi

    echo "llama-server pronto."
    return 0
}

run_warm_condition() {
    local model="$1"
    local temp="$2"
    local cond_dir="$3"
    local model_path

    model_path=$(model_path_for "$model") || return 1

    local final_csv_path="$cond_dir/server_stress_${model}_temp_${temp}_warm.csv"
    local TMP_RESULTS_DIR="/dev/shm/cpu_pinning_tmp_${model}_temp_${temp}_$$"
    local csv_path="$TMP_RESULTS_DIR/server_stress_${model}_temp_${temp}_warm.csv"
    local server_log="$TMP_RESULTS_DIR/server_${model}_temp_${temp}_warm.log"

    mkdir -p "$cond_dir"
    rm -rf "$TMP_RESULTS_DIR"
    mkdir -p "$TMP_RESULTS_DIR"

    if [ -f "$final_csv_path" ]; then
        lines=$(wc -l < "$final_csv_path")
        if [ "$lines" -ge $((N_RUNS + 1)) ]; then
            echo "SKIP warm già completo: $model temp=$temp ($((lines - 1))/$N_RUNS)"
            rm -rf "$TMP_RESULTS_DIR"
            return 0
        elif [ "$lines" -gt 1 ]; then
            echo "CSV warm parziale trovato, lo ricreo: $final_csv_path"
            rm -f "$final_csv_path"
        fi
    fi

    echo
    echo "======================================"
    echo "CPU WARM STRESS PINNING"
    echo "MODELLO: $model"
    echo "TEMP: $temp"
    echo "RUN: $N_RUNS"
    echo "CSV: $csv_path"
    echo "======================================"

    start_server "$model" "$model_path" "$server_log" || {
        rm -rf "$TMP_RESULTS_DIR"
        return 1
    }

    RESULTS_DIR="$TMP_RESULTS_DIR" python3 "$BENCHMARK_PY" \
        --model-name "$model" \
        --model-path "$model_path" \
        --server-url "$SERVER_URL" \
        --server-pid "$SERVER_PID" \
        --ppl-bin "$PPL_BIN" \
        --ppl-dataset "$PPL_DATASET" \
        --ppl-value "" \
        --runs "$N_RUNS" \
        --run-start 1 \
        --tokens "$N_TOKENS" \
        --ctx "$CTX" \
        --threads "$THREADS" \
        --prompt "$PROMPT" \
        --expected "$EXPECTED" \
        --temperature "$temp" \
        --mode warm \
        --warmup-request \
        --csv-path "$csv_path"

    local status=$?

    cleanup_server

    if [ "$status" -eq 0 ]; then
        cp -a "$TMP_RESULTS_DIR"/. "$cond_dir"/

        # Nel CSV sostituiamo i path temporanei con quelli finali.
        if [ -f "$final_csv_path" ]; then
            sed -i "s#${TMP_RESULTS_DIR}#${cond_dir}#g" "$final_csv_path"
        fi

        rm -rf "$TMP_RESULTS_DIR"
    else
        echo "ERRORE: benchmark fallito. NON cancello i file temporanei."
        echo "Directory temporanea conservata: $TMP_RESULTS_DIR"
        return "$status"
    fi

    sleep 3

    return "$status"
}

trap 'cleanup_all; exit 130' INT TERM
trap 'cleanup_all' EXIT

cd "$BASE_DIR" || exit 1
check_requirements

echo "======================================"
echo "Stress CPU llama.cpp - SOLO WARM con pinning"
echo "Risultati base: $RESULTS_BASE"
echo "Modelli: $MODELS_LIST"
echo "Temperature: $TEMPERATURES_LIST"
echo "Run per condizione: $N_RUNS"
echo "PINNING: $PINNING"
echo "SERVER_CPUS: $SERVER_CPUS"
echo "STRESS_CPUS: $STRESS_CPUS"
echo "THREADS llama-server: $THREADS"
echo "Server URL: $SERVER_URL"
echo "======================================"

echo
echo "Controllo processi già attivi:"
pgrep -a stress-ng || true
pgrep -a llama-server || true
echo

for TEST in "${STRESS_TESTS[@]}"; do
    NAME="${TEST%%|*}"
    ARGS="${TEST#*|}"

    STRESS_DIR="$RESULTS_BASE/$NAME"
    STRESS_LOG_DIR="$STRESS_DIR/stress_logs"
    STRESS_LOG="$STRESS_LOG_DIR/stress_${NAME}.log"

    mkdir -p "$STRESS_LOG_DIR"

    echo
    echo "######################################"
    echo "STRESSOR: $NAME"
    echo "Comando: taskset -c $STRESS_CPUS stress-ng $ARGS --metrics-brief"
    echo "Directory: $STRESS_DIR"
    echo "######################################"

    TMPDIR=/tmp setsid taskset -c "$STRESS_CPUS" stress-ng --temp-path /tmp $ARGS --metrics-brief > "$STRESS_LOG" 2>&1 &

    STRESS_PID=$!
    sleep 1
    STRESS_PGID=$(ps -o pgid= -p "$STRESS_PID" | tr -d ' ')

    echo "stress-ng PID:  $STRESS_PID"
    echo "stress-ng PGID: $STRESS_PGID"

    for model in $MODELS_LIST; do
        for temp in $TEMPERATURES_LIST; do
            COND_DIR="$STRESS_DIR/$model"

            run_warm_condition "$model" "$temp" "$COND_DIR"
            STATUS=$?

            if [ "$STATUS" -ne 0 ]; then
                echo "ERRORE: benchmark CPU warm fallito."
                echo "Stressor: $NAME"
                echo "Modello: $model"
                echo "Temperatura: $temp"
                echo "Log stress: $STRESS_LOG"
                cleanup_stress
                exit "$STATUS"
            fi
        done
    done

    echo
    echo "Benchmark completato per stressor $NAME."
    echo "Fermo stress-ng..."

    cleanup_stress

    echo
    echo "Ultime righe log stress-ng per $NAME:"
    tail -n 30 "$STRESS_LOG" || true

    echo
    echo "Condizione stress completata: $NAME"

    sleep 10
done

trap - EXIT

echo
echo "======================================"
echo "Copia risultati finali"
echo "Da: $RESULTS_BASE"
echo "A:  $FINAL_RESULTS_BASE"
echo "======================================"

if [ "$RESULTS_BASE" != "$FINAL_RESULTS_BASE" ]; then
    mkdir -p "$FINAL_RESULTS_BASE"
    cp -a "$RESULTS_BASE"/. "$FINAL_RESULTS_BASE"/
    echo "Copia completata in: $FINAL_RESULTS_BASE"
else
    echo "RESULTS_BASE coincide con FINAL_RESULTS_BASE: nessuna copia necessaria."
fi

echo
echo "======================================"
echo "Stress CPU llama.cpp SOLO WARM con pinning completato"
echo "Risultati in: $RESULTS_BASE"
echo "======================================"
