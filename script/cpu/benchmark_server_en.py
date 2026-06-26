#!/usr/bin/env python3

import argparse
import csv
import threading
import json
import os
import re
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path


def clean_ansi_codes(text):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def get_temp():
    temp_file = Path("/sys/class/thermal/thermal_zone0/temp")
    if temp_file.exists():
        try:
            return float(temp_file.read_text().strip()) / 1000.0
        except Exception:
            return None
    return None


def read_rss_kb(pid):
    status_file = Path(f"/proc/{pid}/status")
    if not status_file.exists():
        return None

    try:
        for line in status_file.read_text(errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except Exception:
        return None

    return None


def read_process_cpu_ticks(pid):
    """
    Restituisce i tick CPU user + system consumati dal processo.

    /proc/<pid>/stat:
      campo 14 = utime
      campo 15 = stime
    """
    stat_path = Path(f"/proc/{pid}/stat")

    try:
        content = stat_path.read_text()
        closing_paren = content.rfind(")")

        if closing_paren == -1:
            return None

        fields = content[closing_paren + 2:].split()

        # Dopo aver rimosso pid e comm:
        # fields[11] corrisponde a utime (campo 14)
        # fields[12] corrisponde a stime (campo 15)
        utime = int(fields[11])
        stime = int(fields[12])

        return utime + stime

    except Exception:
        return None

def monitor_process(pid, stop_event, stats, interval_s=0.2):
    """
    Campiona RAM, CPU e temperatura durante l'inferenza
    senza avviare processi esterni come `ps`.
    """

    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    cpu_count = os.cpu_count() or 1

    previous_ticks = read_process_cpu_ticks(pid)
    previous_time = time.monotonic()

    while not stop_event.wait(interval_s):
        now = time.monotonic()

        rss_kb = read_rss_kb(pid)
        if rss_kb is not None:
            stats["rss_peak_kb"] = max(
                stats["rss_peak_kb"],
                rss_kb,
            )

        current_ticks = read_process_cpu_ticks(pid)

        if previous_ticks is not None and current_ticks is not None:
            elapsed_s = now - previous_time
            delta_ticks = current_ticks - previous_ticks

            if elapsed_s > 0 and delta_ticks >= 0:
                cpu_percent = (
                    delta_ticks / clock_ticks
                ) / elapsed_s * 100.0

                # Come `ps %cpu`, può superare 100%:
                # 400% significa circa quattro core pieni.
                max_percent = cpu_count * 100.0
                cpu_percent = min(cpu_percent, max_percent)

                stats["cpu_values"].append(cpu_percent)

        temp_now = get_temp()
        if temp_now is not None:
            stats["temp_values"].append(temp_now)

        stats["sample_count"] += 1

        previous_ticks = current_ticks
        previous_time = now

    # Ultima lettura RSS e temperatura.
    rss_kb = read_rss_kb(pid)
    if rss_kb is not None:
        stats["rss_peak_kb"] = max(
            stats["rss_peak_kb"],
            rss_kb,
        )

    temp_now = get_temp()
    if temp_now is not None:
        stats["temp_values"].append(temp_now)

def extract_ppl_from_text(text):
    text = clean_ansi_codes(text)

    patterns = [
        r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bPPL\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bperplexity\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bperplexity:\s*([0-9]+(?:\.[0-9]+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1]

    return ""


def calculate_perplexity(ppl_bin, model_path, ppl_dataset, ctx, threads, model_name, results_dir, timeout_s):
    log_path = Path(results_dir) / f"ppl_{model_name}_server.log"

    if not Path(ppl_bin).is_file():
        log_path.write_text(f"ERRORE: llama-perplexity non trovato: {ppl_bin}\n")
        return ""

    if not os.access(ppl_bin, os.X_OK):
        log_path.write_text(f"ERRORE: llama-perplexity non eseguibile: {ppl_bin}\n")
        return ""

    if not Path(model_path).is_file():
        log_path.write_text(f"ERRORE: modello non trovato: {model_path}\n")
        return ""

    if not Path(ppl_dataset).is_file():
        log_path.write_text(f"ERRORE: dataset PPL non trovato: {ppl_dataset}\n")
        return ""

    if Path(ppl_dataset).stat().st_size == 0:
        log_path.write_text(f"ERRORE: dataset PPL vuoto: {ppl_dataset}\n")
        return ""

    print(f"Calcolo Perplexity per {model_name}...")

    cmd = [
        ppl_bin,
        "-m", model_path,
        "-f", ppl_dataset,
        "-c", str(ctx),
        "-t", str(threads),
    ]

    try:
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )

        output = clean_ansi_codes(result.stdout)
        log_path.write_text(output, errors="replace")

        ppl = extract_ppl_from_text(output)

        if ppl:
            print(f"PPL {model_name} = {ppl}")
            return ppl

        print(f"Attenzione: PPL non estratta. Controlla log: {log_path}")
        return ""

    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")

        log_path.write_text(
            "ERRORE: timeout durante il calcolo PPL.\n\n"
            + clean_ansi_codes(partial),
            errors="replace",
        )
        print(f"Attenzione: timeout PPL per {model_name}. Controlla log: {log_path}")
        return ""

    except Exception as exc:
        log_path.write_text(f"ERRORE durante PPL: {exc}\n", errors="replace")
        print(f"Attenzione: errore PPL per {model_name}. Controlla log: {log_path}")
        return ""


def tidy_sentence(text):
    if not text:
        return text

    text = re.sub(r"\s+", " ", text).strip()

    numbered_items = list(re.finditer(r"(?:^|\s)\d+[\.\)]\s+", text))
    if numbered_items:
        last_item = numbered_items[-1]
        tail = text[last_item.start():].strip()

        if not re.search(r"[.!?…]$", tail):
            text = text[:last_item.start()].strip()

    text = re.sub(r"\s+\d+[\.\)]\s*$", "", text).strip()

    matches = list(re.finditer(r"[.!?…]", text))
    if matches:
        idx = matches[-1].start()
        text = text[:idx + 1].strip()

    text = re.sub(r"\s+\d+[\.\)]\s*$", "", text).strip()

    return text


def wait_server_ready(server_url, timeout_s):
    health_url = server_url.rstrip("/") + "/health"
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return True
        except Exception:
            pass

        time.sleep(1)

    return False


def parse_sse_line(line):
    """
    Parse una riga SSE tipo:
    data: {...}
    oppure:
    data: [DONE]
    """
    line = line.strip()

    if not line:
        return None

    if not line.startswith("data:"):
        return None

    payload = line[len("data:"):].strip()

    if payload == "[DONE]":
        return {"done": True}

    try:
        return json.loads(payload)
    except Exception:
        return None


def extract_text_from_stream_event(event):
    """
    Supporta sia endpoint OpenAI-like:
      choices[0].delta.content
      choices[0].text

    sia endpoint llama.cpp /completion:
      content
    """
    if not isinstance(event, dict):
        return ""

    if "content" in event and isinstance(event["content"], str):
        return event["content"]

    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]

        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content

        text = choice.get("text")
        if isinstance(text, str):
            return text

    return ""


def extract_timings_from_event(event):
    """
    Alcune versioni/endpoint di llama-server possono restituire timings.
    Se ci sono, li usiamo.
    """
    if not isinstance(event, dict):
        return {}

    timings = event.get("timings")
    if isinstance(timings, dict):
        return timings

    return {}


def stream_completion(server_url, prompt, tokens, temperature, timeout_s):
    """
    Versione streaming chat OpenAI-compatible.

    TTFT reale:
      request_time = prima della richiesta HTTP
      first_token_time = quando arriva il primo chunk testuale dello stream

    TGS:
      - se il server restituisce timings.predicted_per_second, usiamo quello;
      - altrimenti fallback: numero di chunk testuali / tempo da primo chunk a fine stream.
    """
    url = server_url.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": "local-model",
        "messages": [
            {
                "role": "system",
                "content": "You are a technical assistant. Always answer in English, correctly, simply and concisely."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": tokens,
        "temperature": temperature,
        "stream": True,
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    request_time = time.time()
    first_token_time = None
    end_time = None

    text_parts = []
    chunk_count = 0
    last_timings = {}

    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        for raw_line in response:
            now = time.time()

            try:
                line = raw_line.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            if not line:
                continue

            if not line.startswith("data:"):
                continue

            payload_line = line[len("data:"):].strip()

            if payload_line == "[DONE]":
                end_time = now
                break

            try:
                event = json.loads(payload_line)
            except Exception:
                continue

            timings = event.get("timings")
            if isinstance(timings, dict):
                last_timings = timings

            piece = ""

            choices = event.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]

                delta = choice.get("delta")
                if isinstance(delta, dict):
                    content = delta.get("content")
                    if isinstance(content, str):
                        piece = content

                if not piece:
                    text = choice.get("text")
                    if isinstance(text, str):
                        piece = text

            if piece:
                if first_token_time is None:
                    first_token_time = now

                text_parts.append(piece)
                chunk_count += 1

        if end_time is None:
            end_time = time.time()

    generated_text = "".join(text_parts)

    if first_token_time is not None:
        ttft_s = first_token_time - request_time
    else:
        ttft_s = ""

    total_latency_s = end_time - request_time

    predicted_per_second = ""
    predicted_n = ""

    if last_timings:
        predicted_per_second = last_timings.get("predicted_per_second", "")
        predicted_n = last_timings.get("predicted_n", "")

    if predicted_per_second != "":
        tgs_tokens_s = predicted_per_second
    else:
        if first_token_time is not None:
            generation_time_s = max(end_time - first_token_time, 1e-9)
            tgs_tokens_s = chunk_count / generation_time_s
        else:
            tgs_tokens_s = ""

    if predicted_n == "":
        predicted_n = chunk_count

    return {
        "ttft_s": ttft_s,
        "tgs_tokens_s": tgs_tokens_s,
        "total_latency_s": total_latency_s,
        "generated_text": generated_text,
        "chunk_count": chunk_count,
        "predicted_n": predicted_n,
    }


def run_single_benchmark(args, run_number, ppl_value, writer):
    print(f"Esecuzione run {run_number}")

    results_dir = Path(os.environ.get("RESULTS_DIR", "."))
    temperature_label = str(args.temperature).replace(".", "_")

    log_path = results_dir / (
        f"server_run_{args.model_name}_"
        f"{args.mode}_temp_{temperature_label}_"
        f"{run_number}.log"
    )

    final_path = results_dir / (
        f"server_run_{args.model_name}_"
        f"{args.mode}_temp_{temperature_label}_"
        f"{run_number}_final.txt"
    )

    monitor_stats = {
        "rss_peak_kb": 0,
        "cpu_values": [],
        "temp_values": [],
        "sample_count": 0,
    }

    monitor_stop_event = threading.Event()

    monitor_thread = threading.Thread(
        target=monitor_process,
        args=(
            args.server_pid,
            monitor_stop_event,
            monitor_stats,
        ),
        kwargs={
            "interval_s": 0.2,
        },
        daemon=True,
    )

    monitor_thread.start()

    try:
        result = stream_completion(
            server_url=args.server_url,
            prompt=args.prompt,
            tokens=args.tokens,
            temperature=args.temperature,
            timeout_s=args.timeout,
        )
    finally:
        monitor_stop_event.set()
        monitor_thread.join(timeout=3)

    generated_text = result["generated_text"]
    generated_text_tidy = tidy_sentence(generated_text).strip()

    response_norm = generated_text_tidy.lower()
    expected_norm = args.expected.lower()

    pass1 = 1 if expected_norm in response_norm else 0

    log_path.write_text(
        generated_text + "\n",
        errors="replace",
    )

    final_path.write_text(
        generated_text_tidy + "\n",
        errors="replace",
    )

    print("\n===== Output finale =====\n")
    print(generated_text_tidy + "\n")
    print("===== Fine output =====\n")

    rss_peak_kb = monitor_stats["rss_peak_kb"]
    cpu_values = monitor_stats["cpu_values"]
    temp_values = monitor_stats["temp_values"]

    rss_peak_mb = (
        rss_peak_kb / 1024.0
        if rss_peak_kb
        else ""
    )

    cpu_avg = (
        sum(cpu_values) / len(cpu_values)
        if cpu_values
        else ""
    )

    temp_avg = (
        sum(temp_values) / len(temp_values)
        if temp_values
        else ""
    )

    writer.writerow([
        args.model_name,
        args.mode,
        args.temperature,
        run_number,
        result["ttft_s"],
        result["tgs_tokens_s"],
        result["total_latency_s"],
        rss_peak_mb,
        cpu_avg,
        temp_avg,
        monitor_stats["sample_count"],
        pass1,
        ppl_value,
        result["chunk_count"],
        result["predicted_n"],
        str(log_path),
        str(final_path),
    ])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--server-pid", type=int, required=True)

    parser.add_argument("--ppl-bin", required=True)
    parser.add_argument("--ppl-dataset", required=True)

    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--expected", required=True)

    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--ppl-timeout", type=int, default=900)
    parser.add_argument("--server-ready-timeout", type=int, default=180)
    parser.add_argument("--ppl-value", default="")

    parser.add_argument(
        "--mode",
        choices=["cold", "warm"],
        required=True,
    )

    parser.add_argument(
        "--run-start",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--append-csv",
        action="store_true",
    )

    parser.add_argument(
        "--csv-path",
        default="",
    )

    parser.add_argument(
        "--warmup-request",
        action="store_true",
    )

    args = parser.parse_args()

    results_dir = Path(os.environ.get("RESULTS_DIR", "."))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("======================================")
    print(f"Benchmark server modello: {args.model_name}")
    print(f"Modalità: {args.mode}")
    print(f"Temperatura modello: {args.temperature}")
    print("======================================")

    if not wait_server_ready(
        args.server_url,
        args.server_ready_timeout,
    ):
        raise RuntimeError(
            f"Server non pronto: {args.server_url}"
        )

    ppl_value = args.ppl_value

    if ppl_value:
        print(
            f"PPL {args.model_name} = {ppl_value} "
            "(calcolata prima dell'avvio server)"
        )
    else:
        print(
            f"Attenzione: PPL non fornita per "
            f"{args.model_name}"
        )

    # Nella modalità warm eseguiamo una richiesta preliminare
    # non registrata, in modo da popolare la prompt cache.
    if args.warmup_request:
        print("Eseguo richiesta di warm-up non registrata...")

        warmup_result = stream_completion(
            server_url=args.server_url,
            prompt=args.prompt,
            tokens=args.tokens,
            temperature=args.temperature,
            timeout_s=args.timeout,
        )

        print(
            "Warm-up completato: "
            f"{warmup_result['predicted_n']} token generati"
        )

    # Lo script Bash può passare direttamente il percorso
    # del CSV cold o warm.
    if args.csv_path:
        csv_path = Path(args.csv_path)
    else:
        csv_path = results_dir / (
            f"server_results_{args.model_name}_"
            f"temp_{args.temperature}_{args.mode}.csv"
        )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_has_content = (
        csv_path.exists()
        and csv_path.stat().st_size > 0
    )

    open_mode = "a" if args.append_csv else "w"

    with csv_path.open(open_mode, newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Scriviamo l'intestazione:
        # - sempre quando il file viene ricreato;
        # - soltanto se il file è vuoto in modalità append.
        if not args.append_csv or not file_has_content:
            writer.writerow([
                "model",
                "mode",
                "temperature",
                "run",
                "ttft_s",
                "tgs_tokens_s",
                "total_latency_s",
                "rss_peak_mb",
                "cpu_avg_percent",
                "temp_avg_c",
                "monitor_samples",
                "pass1",
                "ppl",
                "stream_chunk_count",
                "predicted_n",
                "log_path",
                "final_path",
            ])

        for offset in range(args.runs):
            run_number = args.run_start + offset

            run_single_benchmark(
                args,
                run_number,
                ppl_value,
                writer,
            )

    print(f"CSV creato: {csv_path}")


if __name__ == "__main__":
    main()
