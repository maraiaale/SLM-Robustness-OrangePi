
#!/bin/bash

set -u
set -o pipefail

RESULTS_DIR="${RESULTS_DIR:-$HOME/benchmark_slm_npu/results_npu_rkllama_final_30runs}"
export RESULTS_DIR
mkdir -p "$RESULTS_DIR"

RKLLAMA_DIR="${RKLLAMA_DIR:-$HOME/rkllama}"
MODELS_ROOT="${MODELS_ROOT:-$HOME/rkllama/models}"
BENCHMARK_PY="${BENCHMARK_PY:-$HOME/benchmark_slm_npu/benchmark_server_npu.py}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SERVER_URL="http://${HOST}:${PORT}"

N_RUNS="${N_RUNS:-30}"
N_TOKENS="${N_TOKENS:-192}"
CTX="${CTX:-2048}"
THREADS="${THREADS:-$(nproc)}"

#  Per tutte: TEMPERATURES_LIST="0.0 0.3 0.5 0.7 1.0"
TEMPERATURES_LIST="${TEMPERATURES_LIST:-0.0 0.5 1.0 }"

PROMPT="Explain what Linux is.Answer only in English. In a single paragraph, describe its open-source nature, security, stability, portability, and some examples of its use. Write about 70 words, without using bullet points, and end with a complete sentence."
EXPECTED="operating system"

SERVER_PID=""

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

start_server() {
    local log_file="$1"

    pkill -f "flask_server.py" 2>/dev/null || true
    pkill -f "rkllama.server.server" 2>/dev/null || true
    sleep 2

    echo "Avvio RKLLama..."
    cd "$RKLLAMA_DIR" || exit 1

    PYTHONPATH="$RKLLAMA_DIR/src" python3 -m rkllama.server.server \
        --models "$MODELS_ROOT" \
        --processor rk3588 \
        --port "$PORT" \
        --debug > "$log_file" 2>&1 &

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
        tail -n 120 "$log_file" || true
        exit 1
    fi

    echo "RKLLama pronto."
}

stop_server() {
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

run_one_model() {
    local model_name="$1"
    local temp="$2"
    local mode="$3"

    local csv_path
    csv_path="$RESULTS_DIR/npu_results_${model_name}_temp_${temp}_${mode}.csv"

    echo
    echo "======================================"
    echo "MODELLO: $model_name | TEMP: $temp | MODE: $mode"
    echo "CSV: $csv_path"
    echo "======================================"

    local warmup_arg=""
    if [ "$mode" = "warm" ]; then
        warmup_arg="--warmup-request"
    fi

    python3 "$BENCHMARK_PY" \
        --model-name "$model_name" \
        --model-path "$MODELS_ROOT/$model_name" \
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
        --mode "$mode" \
        $warmup_arg \
        --csv-path "$csv_path"
}

prepare_models

echo "======================================"
echo "Benchmark NPU con RKLLama"
echo "Risultati in: $RESULTS_DIR"
echo "Temperature: $TEMPERATURES_LIST"
echo "Run per condizione: $N_RUNS"
echo "Server URL: $SERVER_URL"
echo "======================================"

trap 'stop_server; exit 130' INT TERM
trap 'stop_server' EXIT

csv_done() {
    local csv="$1"
    if [ -f "$csv" ]; then
        local lines
        lines=$(wc -l < "$csv")
        if [ "$lines" -ge $((N_RUNS + 1)) ]; then
            return 0
        fi
    fi
    return 1
}

completed_runs() {
    local csv="$1"
    if [ -f "$csv" ]; then
        local lines
        lines=$(wc -l < "$csv")
        local runs=$((lines - 1))
        if [ "$runs" -lt 0 ]; then
            runs=0
        fi
        echo "$runs"
    else
        echo 0
    fi
}

for temp in $TEMPERATURES_LIST; do
    for model in qwen2.5-0.5b qwen2.5-1.5b qwen2.5-3b qwen2.5-7b; do

        cold_csv="$RESULTS_DIR/npu_results_${model}_temp_${temp}_cold.csv"

        if csv_done "$cold_csv"; then
            echo "SKIP cold completo: $model temp=$temp"
        else
            done_runs=$(completed_runs "$cold_csv")
            start_run=$((done_runs + 1))

            echo
            echo "######################################"
            echo "RESUME COLD: $model | temp=$temp | da run $start_run a $N_RUNS"
            echo "CSV: $cold_csv"
            echo "######################################"

            for run_number in $(seq "$start_run" "$N_RUNS"); do
                server_log="$RESULTS_DIR/rkllama_server_${model}_temp_${temp}_cold_${run_number}.log"

                start_server "$server_log"

                python3 "$BENCHMARK_PY" \
                    --model-name "$model" \
                    --model-path "$MODELS_ROOT/$model" \
                    --server-url "$SERVER_URL" \
                    --server-pid "$SERVER_PID" \
                    --ppl-value "" \
                    --runs 1 \
                    --run-start "$run_number" \
                    --tokens "$N_TOKENS" \
                    --ctx "$CTX" \
                    --threads "$THREADS" \
                    --prompt "$PROMPT" \
                    --expected "$EXPECTED" \
                    --temperature "$temp" \
                    --mode cold \
                    --append-csv \
                    --csv-path "$cold_csv"

                status=$?

                stop_server
                sleep 3

                if [ "$status" -ne 0 ]; then
                    echo "ERRORE: cold fallito per $model temp=$temp run=$run_number"
                    echo "Log server: $server_log"
                    tail -n 120 "$server_log" || true
                    exit "$status"
                fi
            done
        fi

        warm_csv="$RESULTS_DIR/npu_results_${model}_temp_${temp}_warm.csv"

        if csv_done "$warm_csv"; then
            echo "SKIP warm completo: $model temp=$temp"
        else
            # Warm deve essere coerente: se parziale, lo rifacciamo da zero per quella condizione.
            if [ -f "$warm_csv" ]; then
                lines=$(wc -l < "$warm_csv")
                if [ "$lines" -gt 1 ]; then
                    echo "Warm parziale trovato, lo ricreo: $warm_csv"
                    rm -f "$warm_csv"
                fi
            fi

            echo
            echo "######################################"
            echo "RESUME WARM: $model | temp=$temp | run=$N_RUNS"
            echo "CSV: $warm_csv"
            echo "######################################"

            server_log="$RESULTS_DIR/rkllama_server_${model}_temp_${temp}_warm.log"

            start_server "$server_log"

            run_one_model "$model" "$temp" warm
            status=$?

            stop_server
            sleep 3

            if [ "$status" -ne 0 ]; then
                echo "ERRORE: warm fallito per $model temp=$temp"
                echo "Log server: $server_log"
                tail -n 120 "$server_log" || true
                exit "$status"
            fi
        fi
    done
done

trap - INT TERM EXIT

echo
echo "======================================"
echo "Benchmark RKLLama NPU completato"
echo "CSV in: $RESULTS_DIR"
echo "======================================"
