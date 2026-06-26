# ==============================================================================
# NOTE METODOLOGICA - CPU PINNING E GESTIONE OUTPUT
# ==============================================================================
#
# Questo script esegue la campagna di stress NPU in modalità warm applicando
# CPU affinity tramite taskset:
#
#   - RKLLama/server: CPU 4-7
#   - stress-ng:      CPU 0-3
#
# Sulla Orange Pi 5 Plus/RK3588 i core 4-7 sono i core ad alte prestazioni,
# mentre i core 0-3 sono i core a frequenza inferiore. Il pinning viene usato
# per ridurre la competizione diretta tra il processo di inferenza e i processi
# di stress-ng.
#
# Per lo stressor open_w5, che sollecita intensivamente le operazioni di
# apertura/chiusura file, può essere utile salvare temporaneamente i risultati
# su tmpfs (/dev/shm), senza modificare lo stressor:
#
#   RESULTS_BASE="/dev/shm/results_stress_npu_pinning" ./run_stress_npu_pinning.sh
#
# Al termine, i risultati possono essere consolidati nella directory finale:
#
#   cp -a /dev/shm/results_stress_npu_pinning/open_w5 \
#         ~/benchmark_slm_npu/results_stress_npu_pinning/
#
# Se anche altri stressor sono stati completati su /dev/shm, copiarli allo stesso
# modo nella cartella finale. L'uso di /dev/shm riguarda solo la destinazione
# temporanea dei file di output e non modifica i parametri del benchmark né la
# configurazione dello stressor.
#
# ==============================================================================


#!/bin/bash

set -u
set -o pipefail

BASE_DIR="$HOME/benchmark_slm_npu"
RKLLAMA_DIR="${RKLLAMA_DIR:-$HOME/rkllama}"
MODELS_ROOT="${MODELS_ROOT:-$HOME/rkllama/models}"
BENCHMARK_PY="${BENCHMARK_PY:-$BASE_DIR/benchmark_server_npu.py}"

RESULTS_BASE="${RESULTS_BASE:-$BASE_DIR/results_stress_npu_pinning}"
mkdir -p "$RESULTS_BASE"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SERVER_URL="http://${HOST}:${PORT}"

N_RUNS="${N_RUNS:-30}"
N_TOKENS="${N_TOKENS:-192}"
CTX="${CTX:-2048}"
THREADS="${THREADS:-$(nproc)}"

#MODELS_LIST="${MODELS_LIST:-qwen2.5-0.5b qwen2.5-1.5b qwen2.5-3b qwen2.5-7b}"
MODELS_LIST="${MODELS_LIST:-qwen2.5-0.5b qwen2.5-1.5b qwen2.5-3b qwen2.5-7b}"


# Temperature della baseline NPU finale effettivamente usate.
TEMPERATURES_LIST="${TEMPERATURES_LIST:-0.0 0.5 1.0}"

PINNING="${PINNING:-1}"
SERVER_CPUS="${SERVER_CPUS:-4-7}"
STRESS_CPUS="${STRESS_CPUS:-0-3}"

PROMPT="Explain what Linux is.Answer only in English. In a single paragraph, describe its open-source nature, security, stability, portability, and some examples of its use. Write about 70 words, without using bullet points, and end with a complete sentence."
EXPECTED="operating system"

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

check_requirements() {
    if [ ! -f "$BENCHMARK_PY" ]; then
        echo "ERRORE: benchmark Python non trovato: $BENCHMARK_PY"
        exit 1
    fi

    if [ ! -d "$RKLLAMA_DIR" ]; then
        echo "ERRORE: directory RKLLama non trovata: $RKLLAMA_DIR"
        exit 1
    fi

    if ! command -v stress-ng >/dev/null 2>&1; then
        echo "ERRORE: stress-ng non trovato."
        echo "Installa con: sudo apt install stress-ng"
        exit 1
    fi
}

prepare_models() {
    mkdir -p "$MODELS_ROOT/qwen2.5-0.5b"
    mkdir -p "$MODELS_ROOT/qwen2.5-1.5b"
    mkdir -p "$MODELS_ROOT/qwen2.5-3b"
    mkdir -p "$MODELS_ROOT/qwen2.5-7b"

    ln -sf "$HOME/models/qwen2.5-0.5b-instruct-rk3588-w8a8.rkllm" \
        "$MODELS_ROOT/qwen2.5-0.5b/qwen2.5-0.5b-instruct-rk3588-w8a8.rkllm"

    ln -sf "$HOME/models/qwen2.5-1.5b-instruct-rk3588-w8a8.rkllm" \
        "$MODELS_ROOT/qwen2.5-1.5b/qwen2.5-1.5b-instruct-rk3588-w8a8.rkllm"

    ln -sf "$HOME/models/qwen2.5-3b-instruct-rk3588.rkllm" \
        "$MODELS_ROOT/qwen2.5-3b/qwen2.5-3b-instruct-rk3588.rkllm"

    ln -sf "$HOME/models/qwen2.5-7b-instruct-rk3588.rkllm" \
        "$MODELS_ROOT/qwen2.5-7b/qwen2.5-7b-instruct-rk3588.rkllm"

    cat > "$MODELS_ROOT/qwen2.5-0.5b/Modelfile" <<'EOF'
FROM="qwen2.5-0.5b-instruct-rk3588-w8a8.rkllm"
HUGGINGFACE_PATH="Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM="You are a technical assistant. Always answer in English, correctly, simply and concisely."
PARAMETER top_p 0.9
EOF

    cat > "$MODELS_ROOT/qwen2.5-1.5b/Modelfile" <<'EOF'
FROM="qwen2.5-1.5b-instruct-rk3588-w8a8.rkllm"
HUGGINGFACE_PATH="Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM="You are a technical assistant. Always answer in English, correctly, simply and concisely."
PARAMETER top_p 0.9
EOF

    cat > "$MODELS_ROOT/qwen2.5-3b/Modelfile" <<'EOF'
FROM="qwen2.5-3b-instruct-rk3588.rkllm"
HUGGINGFACE_PATH="Qwen/Qwen2.5-3B-Instruct"
SYSTEM="You are a technical assistant. Always answer in English, correctly, simply and concisely."
PARAMETER top_p 0.9
EOF

    cat > "$MODELS_ROOT/qwen2.5-7b/Modelfile" <<'EOF'
FROM="qwen2.5-7b-instruct-rk3588.rkllm"
HUGGINGFACE_PATH="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a technical assistant. Always answer in English, correctly, simply and concisely."
PARAMETER top_p 0.9
EOF
}

cleanup_server() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Spengo RKLLama PID $SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true

        for _ in $(seq 1 30); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 0.3
        done

        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "RKLLama non chiuso, SIGKILL."
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
    pgrep -a rkllama || true
    pgrep -a python3 | grep -E "rkllama|benchmark_server_npu" || true
}

start_server() {
    local server_log="$1"

    pkill -f "rkllama.server.server" 2>/dev/null || true
    sleep 2

    echo "Avvio RKLLama..."
    cd "$RKLLAMA_DIR" || exit 1

    PYTHONPATH="$RKLLAMA_DIR/src" taskset -c "$SERVER_CPUS" python3 -m rkllama.server.server \
        --models "$MODELS_ROOT" \
        --processor rk3588 \
        --port "$PORT" \
        > "$server_log" 2>&1 &

    SERVER_PID=$!

    echo "RKLLama PID: $SERVER_PID"
    echo "Attendo server..."

    local ready=0

    for _ in $(seq 1 180); do
        if curl -fsS "$SERVER_URL/v1/models" >/dev/null 2>&1; then
            ready=1
            break
        fi

        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            break
        fi

        sleep 1
    done

    if [ "$ready" -ne 1 ]; then
        echo "ERRORE: RKLLama non pronto."
        tail -n 120 "$server_log" || true
        cleanup_server
        return 1
    fi

    echo "RKLLama pronto."
    return 0
}

run_warm_condition() {
    local model="$1"
    local temp="$2"
    local cond_dir="$3"

    local csv_path="$cond_dir/npu_stress_${model}_temp_${temp}_warm.csv"
    local server_log="/tmp/rkllama_server_${model}_temp_${temp}_warm.log"

    mkdir -p "$cond_dir"

    if [ -f "$csv_path" ]; then
        lines=$(wc -l < "$csv_path")
        if [ "$lines" -ge $((N_RUNS + 1)) ]; then
            echo "SKIP warm già completo: $model temp=$temp ($((lines - 1))/$N_RUNS)"
            return 0
        elif [ "$lines" -gt 1 ]; then
            echo "CSV warm parziale trovato, lo ricreo: $csv_path"
            rm -f "$csv_path"
        fi
    fi

    echo
    echo "======================================"
    echo "NPU WARM STRESS"
    echo "MODELLO: $model"
    echo "TEMP: $temp"
    echo "RUN: $N_RUNS"
    echo "CSV: $csv_path"
    echo "======================================"

    start_server "$server_log" || return 1

    python3 "$BENCHMARK_PY" \
        --model-name "$model" \
        --model-path "$MODELS_ROOT/$model" \
        --server-url "$SERVER_URL" \
        --server-pid "$SERVER_PID" \
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
    sleep 3

    return "$status"
}

trap 'cleanup_all; exit 130' INT TERM
trap 'cleanup_all' EXIT

check_requirements
prepare_models

echo "======================================"
echo "Stress NPU RKLLama - SOLO WARM"
echo "Risultati base: $RESULTS_BASE"
echo "Modelli: $MODELS_LIST"
echo "Temperature: $TEMPERATURES_LIST"
echo "Run per condizione: $N_RUNS"
echo "PINNING: $PINNING"
echo "SERVER_CPUS: $SERVER_CPUS"
echo "STRESS_CPUS: $STRESS_CPUS"
echo "Server URL: $SERVER_URL"
echo "======================================"

echo
echo "Controllo processi già attivi:"
pgrep -a stress-ng || true
pgrep -a rkllama || true
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
    echo "Comando: stress-ng $ARGS --metrics-brief"
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
                echo "ERRORE: benchmark NPU warm fallito."
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
echo "Stress NPU RKLLama SOLO WARM completato"
echo "Risultati in: $RESULTS_BASE"
echo "======================================"
