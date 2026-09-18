#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU vs GPU PDHG crossover benchmark (issue #19).

Generates synthetic LPs of increasing size from a known KKT optimum (same construction as
generate_large_lp.py) and solves each with both the CPU and GPU PDHG engines at two
tolerances.  The goal is the crossover point: the size at which the GPU backend overtakes
the CPU backend in wall-clock time, and by how much at the largest sizes.

Usage:
    python bench/runners/gpu_report.py --binary build_gpu/sankhya
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"

CSV_COLUMNS = [
    "instance", "rows", "cols", "nnz", "algorithm", "tolerance",
    "status", "objective", "published_objective", "relative_error",
    "iterations", "seconds", "reached_tolerance",
    "git_commit", "machine", "timestamp_utc",
]

# Problem sizes: (rows, cols, nnz_per_col)
SIZES = [
    (200,   200,   5),
    (500,   500,   5),
    (1000,  1000,  5),
    (2000,  2000,  5),
    (5000,  5000,  8),
    (10000, 10000, 8),
]

TOLERANCES = [1e-4, 1e-8]


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() or "unknown"


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def generate_lp(rows: int, cols: int, nnz_per_col: int, seed: int, out: Path) -> float:
    """Generate a KKT LP and return its analytic optimum."""
    rng = random.Random(seed)
    # Choose a feasible primal point
    x_star = [rng.randint(1, 10) if rng.random() > 0.3 else 0 for _ in range(cols)]
    # Choose positive dual multipliers on a subset of rows
    y_star = [rng.randint(1, 5) if rng.random() > 0.5 else 0 for _ in range(rows)]
    # Build sparse A (each column has nnz_per_col entries)
    A: list[list[tuple[int, int]]] = [[] for _ in range(cols)]
    for j in range(cols):
        row_indices = rng.sample(range(rows), min(nnz_per_col, rows))
        for i in row_indices:
            A[j].append((i, rng.randint(1, 5)))
    # c_j = sum_i A_ij * y*_i + d_j, where d_j >= 0 and d_j=0 when x*_j > 0
    c = []
    for j in range(cols):
        dual_term = sum(a * y_star[i] for i, a in A[j])
        d_j = 0 if x_star[j] > 0 else rng.randint(0, 3)
        c.append(dual_term + d_j)
    # b_i = A_i x* (active rows are tight; inactive rows get a slack)
    b = [0.0] * rows
    for j in range(cols):
        for i, a in A[j]:
            b[i] += a * x_star[j]
    for i in range(rows):
        if y_star[i] == 0:
            b[i] -= rng.randint(1, 3)  # make row inactive (slack > 0)
    analytic_optimum = sum(c[j] * x_star[j] for j in range(cols))
    # Write MPS
    lines = ["NAME          KKT_LP", "ROWS", " N  obj"]
    for i in range(rows):
        lines.append(f" G  r{i}")
    lines.append("COLUMNS")
    for j in range(cols):
        if c[j] != 0:
            lines.append(f"    x{j}  obj  {c[j]}")
        for i, a in A[j]:
            lines.append(f"    x{j}  r{i}  {a}")
    lines.append("RHS")
    for i in range(rows):
        if b[i] != 0:
            lines.append(f"    rhs  r{i}  {b[i]}")
    lines.append("BOUNDS")
    for j in range(cols):
        ub = x_star[j] + rng.randint(0, 3) + 1
        lines.append(f" UP BND  x{j}  {ub}")
    lines.append("ENDATA")
    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    return float(analytic_optimum)


def run_solve(binary: Path, mps: Path, algorithm: str, tolerance: float,
              time_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "s.json"
        # Both CPU and GPU use algorithm=pdhg; GPU adds gpu=true.
        command = [
            str(binary), "solve", str(mps),
            "--stats", str(stats),
            "--time-limit", str(time_limit),
            "--option", "log_to_console=false",
            "--option", "algorithm=pdhg",
            "--option", f"pdhg_tolerance={tolerance:g}",
        ]
        if algorithm == "pdhg-cuda":
            command += ["--option", "gpu=true"]
        started = time.perf_counter()
        subprocess.run(command, capture_output=True, text=True)
        seconds = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": None, "iterations": "", "seconds": seconds}
        blob = json.loads(stats.read_text())
        return {
            "status": blob.get("result", {}).get("status", "unknown"),
            "objective": as_number(blob.get("result", {}).get("objective")),
            "iterations": blob.get("effort", {}).get("iterations", ""),
            "algorithm_used": blob.get("result", {}).get("algorithm", algorithm),
            "seconds": seconds,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    commit = git_commit()
    machine = f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []

    algorithms = ["pdhg-cpu", "pdhg-cuda"]

    print("GPU vs CPU PDHG crossover benchmark")
    print(f"binary: {args.binary}")
    print(f"commit: {commit}  machine: {machine}\n")
    print(f"{'size':>12}  {'alg':>12}  {'tol':>6}  {'status':>12}  {'seconds':>9}  {'iters':>8}  "
          f"{'speedup':>8}")
    print("-" * 80)

    cpu_times: dict[tuple, float] = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for (nrows, ncols, nnz) in SIZES:
            mps_path = Path(tmp_dir) / f"kkt_{nrows}x{ncols}.mps"
            optimum = generate_lp(nrows, ncols, nnz, args.seed, mps_path)
            nnz_total = nrows * nnz  # approximate

            for tol in TOLERANCES:
                for alg in algorithms:
                    result = run_solve(args.binary, mps_path, alg, tol, args.time_limit)
                    obj = result["objective"]
                    err = (abs(obj - optimum) / max(1.0, abs(optimum))) if obj is not None else None
                    key = (nrows, tol, "pdhg-cpu")
                    if alg == "pdhg-cpu":
                        cpu_times[key] = result["seconds"]
                    cpu_t = cpu_times.get((nrows, tol, "pdhg-cpu"))
                    speedup = (cpu_t / result["seconds"]) if (alg != "pdhg-cpu" and cpu_t) else 1.0

                    rows.append({
                        "instance": f"kkt_{nrows}x{ncols}",
                        "rows": nrows, "cols": ncols, "nnz": nnz_total,
                        "algorithm": result.get("algorithm_used", alg),
                        "tolerance": f"{tol:g}",
                        "status": result["status"],
                        "objective": "" if obj is None else repr(obj),
                        "published_objective": repr(optimum),
                        "relative_error": "" if err is None else repr(err),
                        "iterations": result["iterations"],
                        "seconds": round(result["seconds"], 6),
                        "reached_tolerance": int(result["status"] in ("optimal", "feasible")),
                        "git_commit": commit, "machine": machine, "timestamp_utc": timestamp,
                    })

                    speedup_str = f"{speedup:.2f}x" if alg != "pdhg-cpu" else "baseline"
                    print(f"{nrows:>5}x{ncols:<6}  {alg:>12}  {tol:>6.0e}  "
                          f"{result['status']:>12}  {result['seconds']:>9.3f}  "
                          f"{str(result['iterations']):>8}  {speedup_str:>8}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (RESULTS_DIR / f"gpu-{commit}.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
