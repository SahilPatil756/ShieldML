"""
Performance and latency benchmarking script for ShieldML FastAPI service.
Measures sequential single-transaction API response times, p95, p99, and throughput.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("BenchmarkAPI")


def run_benchmark(
    api_url: str = "http://127.0.0.1:8000",
    n_requests: int = 100,
    warmup_requests: int = 10,
    explain: bool = False,  # fast inference mode
):
    logger.info(f"Connecting to ShieldML API at {api_url}...")
    client = httpx.Client(base_url=api_url, timeout=30.0)

    # 1. Health check
    try:
        health_resp = client.get("/health")
        if health_resp.status_code != 200 or not health_resp.json().get("model_ready"):
            logger.error(f"API is not ready: {health_resp.text}")
            return
        logger.info(f"API Health verified: {health_resp.json()}")
    except Exception as e:
        logger.error(f"Failed to connect to API: {e}")
        return

    # Sample transaction payload
    sample_payload = {
        "transaction_id": "BENCH_001",
        "amount": 125.50,
        "time": 43200.0,
        "V1": -1.2,
        "V4": 1.1,
        "V10": -0.8,
        "V12": -0.5,
        "V14": -1.3,
        "V17": -0.4,
        "explain": explain,
    }

    # 2. Warm-up
    logger.info(f"Running {warmup_requests} warm-up requests...")
    for _ in range(warmup_requests):
        client.post("/predict", json=sample_payload)

    # 3. Benchmark sequential requests
    logger.info(f"Executing {n_requests} sequential benchmark requests (explain={explain})...")
    latencies: list[float] = []
    model_latencies: list[float] = []
    success_count = 0
    fail_count = 0

    t_start_total = time.perf_counter()
    for i in range(n_requests):
        payload = sample_payload.copy()
        payload["transaction_id"] = f"BENCH_{i+1:04d}"

        t0 = time.perf_counter()
        try:
            resp = client.post("/predict", json=payload)
            elapsed_http = (time.perf_counter() - t0) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                latencies.append(elapsed_http)
                model_latencies.append(data.get("latency_ms", elapsed_http))
                success_count += 1
            else:
                fail_count += 1
        except Exception:
            fail_count += 1

    total_time_s = time.perf_counter() - t_start_total

    # 4. Compute statistics
    avg_lat = float(np.mean(latencies))
    med_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    rps = round(success_count / max(0.001, total_time_s), 1)

    target_met = avg_lat < 50.0

    benchmark_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "target_latency_ms": 50.0,
        "target_achieved": target_met,
        "configuration": {
            "n_requests": n_requests,
            "warmup_requests": warmup_requests,
            "explain_enabled": explain,
        },
        "http_latency_ms": {
            "mean": round(avg_lat, 2),
            "median": round(med_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2),
        },
        "model_internal_latency_ms": {
            "mean": round(float(np.mean(model_latencies)), 2) if model_latencies else 0.0,
            "median": round(float(np.median(model_latencies)), 2) if model_latencies else 0.0,
        },
        "throughput_rps": rps,
        "successful_requests": success_count,
        "failed_requests": fail_count,
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "cpu_architecture": platform.machine(),
        },
    }

    print("\n" + "=" * 65)
    print(" SHIELDML REAL-TIME API LATENCY BENCHMARK REPORT")
    print("=" * 65)
    print(f"Target Latency:          < 50.0 ms")
    print(f"Measured Mean Latency:   {avg_lat:.2f} ms ({'PASSED' if target_met else 'INVESTIGATE'})")
    print(f"Median Latency:          {med_lat:.2f} ms")
    print(f"95th Percentile (p95):   {p95_lat:.2f} ms")
    print(f"99th Percentile (p99):   {p99_lat:.2f} ms")
    print(f"Min / Max:               {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"Throughput:              {rps} requests/sec")
    print(f"Success / Failed:        {success_count} / {fail_count}")
    print("=" * 65)

    out_file = Path("artifacts/reports/latency_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")
    logger.info(f"Saved benchmark results to {out_file}")


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    run_benchmark(api_url=url)
