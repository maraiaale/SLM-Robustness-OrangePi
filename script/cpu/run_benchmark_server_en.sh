#!/bin/bash

set -u
set -o pipefail

#RESULTS_DIR="$HOME/benchmark_slm/results_server" #per baseline 
RESULTS_DIR="${RESULTS_DIR:-$HOME/benchmark_slm/results_server}"
export RESULTS_DIR
mkdir -p "$RESULTS_DIR"

LLAMA_SERVER="/home/orangepi/llama.cpp/build/bin/llama-server"
PPL_BIN="/home/orangepi/llama.cpp/build/bin/llama-perplexity"
BENCHMARK_PY="$HOME/benchmark_slm/benchmark_server.py"

# Per la prima prova usiamo 3 run.
# Dopo la validazione passeremo a 30.
N_RUNS=30
N_TOKENS=192
CTX=2048
THREADS=$(nproc)
#TEMPERATURES=(0.0 0.3 0.5 0.7 1.0)
TEMPERATURES=(0.0 0.5 1.0)
TEMP=""

HOST="127.0.0.1"
PORT=8080
SERVER_URL="http://${HOST}:${PORT}"

# Prompt unico in inglese per tutti i modelli.
PROMPT="Explain what Linux is.Answer only in English. In a single paragraph, describe its open-source nature, security, stability, portability, and some examples of its use. Write about 70 words, without using bullet points, and end with a complete sentence."
EXPECTED="operating system"

PPL_DATASET="$HOME/benchmark_slm/ppl_wikipedia_dataset_en.txt"

QWEN05="/home/orangepi/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
QWEN15="/home/orangepi/models/qwen2.5-1.5b.gguf"
QWEN3="/home/orangepi/models/qwen2.5-3b.gguf"
# Il 7B Q4_K_M è splittato in due file; llama.cpp carica il secondo pezzo automaticamente
# se si passa il primo file e il secondo resta nella stessa cartella.
QWEN7="/home/orangepi/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
QWEN7_PART2="/home/orangepi/models/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"


SERVER_PID=""

echo "======================================"
echo "Avvio benchmark SERVER SLM"
echo "Risultati in: $RESULTS_DIR"
echo "Run cold per modello: $N_RUNS"
echo "Run warm per modello: $N_RUNS"
echo "Temperature modello: ${TEMPERATURES[*]}"
echo "======================================"

check_file() {
    if [ ! -f "$1" ]; then
        echo "ERRORE: file non trovato: $1"
        exit 1
    fi
}

check_file "$LLAMA_SERVER"
check_file "$PPL_BIN"
check_file "$BENCHMARK_PY"
check_file "$PPL_DATASET"
check_file "$QWEN05"
check_file "$QWEN15"
check_file "$QWEN3"
check_file "$QWEN7"
check_file "$QWEN7_PART2"



calculate_ppl() {
    local model_name="$1"
    local model_path="$2"
    local ppl_log

    ppl_log="$RESULTS_DIR/ppl_${model_name}_server.log"

    echo "======================================"
    echo "Calcolo Perplexity per: $model_name"
    echo "Modello: $model_path"
    echo "Log PPL: $ppl_log"
    echo "======================================"

    "$PPL_BIN" \
        -m "$model_path" \
        -f "$PPL_DATASET" \
        -c "$CTX" \
        -t "$THREADS" \
        > "$ppl_log" 2>&1

    local status=$?

    if [ "$status" -ne 0 ]; then
        echo "ERRORE: calcolo PPL fallito per $model_name"
        echo "Ultime righe log PPL:"
        tail -n 80 "$ppl_log"
        exit "$status"
    fi

    PPL_VALUE=$(
        grep -E "Final estimate: PPL =|PPL =" "$ppl_log" |
        tail -n 1 |
        sed -E 's/.*PPL = ([0-9.]+).*/\1/'
    )

    if [ -z "$PPL_VALUE" ]; then
        echo "ERRORE: PPL non estratta per $model_name"
        echo "Ultime righe log PPL:"
        tail -n 80 "$ppl_log"
        exit 1
    fi

    echo "PPL $model_name = $PPL_VALUE"
}


start_server() {
    local model_name="$1"
    local model_path="$2"
    local server_log="$3"

    echo "======================================"
    echo "Avvio llama-server per: $model_name"
    echo "Modello: $model_path"
    echo "Log server: $server_log"
    echo "======================================"

    "$LLAMA_SERVER" \
        -m "$model_path" \
        --host "$HOST" \
        --port "$PORT" \
        -c "$CTX" \
        -t "$THREADS" \
        > "$server_log" 2>&1 &

    SERVER_PID=$!

    echo "llama-server PID: $SERVER_PID"
    echo "Attendo che il server sia realmente pronto..."

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

        if [ -f "$server_log" ]; then
            echo "Ultime righe log:"
            tail -n 100 "$server_log"
        fi

        stop_server
        return 1
    fi

    echo "llama-server pronto."
    return 0
}


stop_server() {
    if [ -n "${SERVER_PID:-}" ] &&
       kill -0 "$SERVER_PID" 2>/dev/null; then

        echo "Spengo llama-server PID $SERVER_PID"

        kill "$SERVER_PID" 2>/dev/null || true

        for _ in $(seq 1 20); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi

            sleep 0.2
        done

        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Il server non si è chiuso: invio SIGKILL."
            kill -9 "$SERVER_PID" 2>/dev/null || true
        fi

        wait "$SERVER_PID" 2>/dev/null || true
    fi

    SERVER_PID=""
}


run_model_cold() {
    local model_name="$1"
    local model_path="$2"
    local ppl_value="$3"

    local csv_path
    csv_path="$RESULTS_DIR/server_results_${model_name}_temp_${TEMP}_cold.csv"

    # Il file viene ricreato all'inizio dell'esperimento cold.
    rm -f "$csv_path"

    echo
    echo "======================================"
    echo "Benchmark COLD: $model_name"
    echo "Temperatura modello: $TEMP"
    echo "Run misurati: $N_RUNS"
    echo "======================================"

    local run_number

    for run_number in $(seq 1 "$N_RUNS"); do
        echo
        echo "----- COLD RUN $run_number/$N_RUNS -----"

        local server_log
        server_log="$RESULTS_DIR/server_${model_name}_temp_${TEMP}_cold_${run_number}.log"

        start_server \
            "$model_name" \
            "$model_path" \
            "$server_log" || exit 1

        trap 'stop_server; exit 130' INT TERM
        trap 'stop_server' EXIT

        python3 "$BENCHMARK_PY" \
            --model-name "$model_name" \
            --model-path "$model_path" \
            --server-url "$SERVER_URL" \
            --server-pid "$SERVER_PID" \
            --ppl-bin "$PPL_BIN" \
            --ppl-dataset "$PPL_DATASET" \
            --ppl-value "$ppl_value" \
            --runs 1 \
            --run-start "$run_number" \
            --tokens "$N_TOKENS" \
            --ctx "$CTX" \
            --threads "$THREADS" \
            --prompt "$PROMPT" \
            --expected "$EXPECTED" \
            --temperature "$TEMP" \
            --mode cold \
            --append-csv \
            --csv-path "$csv_path"

        local status=$?

        stop_server
        trap - INT TERM EXIT

        if [ "$status" -ne 0 ]; then
            echo "ERRORE nel cold run $run_number di $model_name"
            echo "Controlla il log: $server_log"
            exit "$status"
        fi

        # Piccola pausa tra processi server consecutivi.
        sleep 2
    done

    echo "CSV cold creato: $csv_path"
}


run_model_warm() {
    local model_name="$1"
    local model_path="$2"
    local ppl_value="$3"

    local csv_path
    local server_log

    csv_path="$RESULTS_DIR/server_results_${model_name}_temp_${TEMP}_warm.csv"
    server_log="$RESULTS_DIR/server_${model_name}_temp_${TEMP}_warm.log"

    # Il file warm viene ricreato all'inizio.
    rm -f "$csv_path"

    echo
    echo "======================================"
    echo "Benchmark WARM: $model_name"
    echo "Temperatura modello: $TEMP"
    echo "Run misurati: $N_RUNS"
    echo "È prevista una richiesta warm-up non registrata."
    echo "======================================"

    start_server \
        "$model_name" \
        "$model_path" \
        "$server_log" || exit 1

    trap 'stop_server; exit 130' INT TERM
    trap 'stop_server' EXIT

    python3 "$BENCHMARK_PY" \
        --model-name "$model_name" \
        --model-path "$model_path" \
        --server-url "$SERVER_URL" \
        --server-pid "$SERVER_PID" \
        --ppl-bin "$PPL_BIN" \
        --ppl-dataset "$PPL_DATASET" \
        --ppl-value "$ppl_value" \
        --runs "$N_RUNS" \
        --run-start 1 \
        --tokens "$N_TOKENS" \
        --ctx "$CTX" \
        --threads "$THREADS" \
        --prompt "$PROMPT" \
        --expected "$EXPECTED" \
        --temperature "$TEMP" \
        --mode warm \
        --warmup-request \
        --csv-path "$csv_path"

    local status=$?

    stop_server
    trap - INT TERM EXIT

    if [ "$status" -ne 0 ]; then
        echo "ERRORE nel benchmark warm di $model_name"
        echo "Controlla il log: $server_log"
        exit "$status"
    fi

    echo "CSV warm creato: $csv_path"
}


run_all_conditions() {
    local model_name="$1"
    local model_path="$2"

    # La PPL viene calcolata una sola volta per modello:
    # non dipende dalla temperatura di generazione.
    calculate_ppl "$model_name" "$model_path"

    local model_ppl="$PPL_VALUE"
    local temp_value

    for temp_value in "${TEMPERATURES[@]}"; do
        TEMP="$temp_value"

        echo
        echo "######################################"
        echo "MODELLO: $model_name"
        echo "TEMPERATURA: $TEMP"
        echo "######################################"
#commento righe 362-365 per exe gli stress ed usare solo la modalità warm 
        run_model_cold \
            "$model_name" \
            "$model_path" \
            "$model_ppl"

        run_model_warm \
            "$model_name" \
            "$model_path" \
            "$model_ppl"
    done
}

run_all_conditions "qwen2.5-0.5b" "$QWEN05"
run_all_conditions "qwen2.5-1.5b" "$QWEN15"
run_all_conditions "qwen2.5-3b" "$QWEN3"
run_all_conditions "qwen2.5-7b" "$QWEN7"


echo
echo "======================================"
echo "Benchmark SERVER cold/warm completato"
echo "CSV generati in: $RESULTS_DIR"
echo "======================================"