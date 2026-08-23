"""
解析 torch_npu profiler 输出（rankN/*_ascend_pt）。

默认扫描 --profile-dir 下所有名为 rank<N> 的子目录，每个目录一个线程。
可用 --ranks 只解析指定卡号。

采集侧默认只落 PROF_* 原始数据。本脚本在找不到 kernel_details.csv 时
会先调用 torch_npu.profiler.analyse（或 msprof --export=on）再读 CSV。

目录约定（test_dit_longseq.py）:
  <profile-dir>/
    seq50000/rank0/<worker>_ascend_pt/PROF_*/          # 原始
    seq50000/rank0/<worker>_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv  # 解析后

示例:
  python parse_dit_profile.py --profile-dir ./msprof_output_20260821_095900
  python parse_dit_profile.py --profile-dir ./msprof_output_xxx/seq50000 --ranks 0
  python parse_dit_profile.py --profile-dir ./msprof_output_xxx --ranks 0 1 --top 20
  python parse_dit_profile.py --profile-dir ./msprof_output_xxx --no-export
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RANK_DIR_RE = re.compile(r"^rank(\d+)$")

CSV_CANDIDATES = (
    "kernel_details.csv",
    "op_statistic.csv",
    "op_summary.csv",
    "operator_details.csv",
)

NAME_KEYS = ("Name", "Op Name", "op_name", "Kernel Name", "OpName", "name")
COUNT_KEYS = ("Count", "count", "Calls", "calls", "Number")
DURATION_KEYS = (
    "Duration(us)",
    "Task Duration(us)",
    "Duration(µs)",
    "Time(us)",
    "Total Time(us)",
    "Total Time(ms)",
    "Duration",
    "duration(us)",
    "Elapse Time(us)",
)
AVG_KEYS = ("Avg Time(us)", "Average Time(us)", "Avg(us)", "avg_us")
MAC_KEYS = ("mac_time(us)", "Mac Time(us)", "mac Time(us)")
VEC_KEYS = ("vec_time(us)", "Vec Time(us)", "vec Time(us)")
AICORE_KEYS = ("aicore_time(us)", "AI Core Time(us)", "aicore Time(us)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse NPU torch_npu profile under rankN/ dirs")
    parser.add_argument(
        "--profile-dir",
        type=str,
        required=True,
        help="Profile 根目录（其下或递归子目录中含 rank0/rank1/...）",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=None,
        help="只解析这些卡号（如 --ranks 0 2）。默认解析所有 rank* 子目录",
    )
    parser.add_argument("--top", type=int, default=15, help="每个 rank 打印耗时最高的 kernel 条数")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="摘要 JSON 输出目录（默认 <profile-dir>/parsed）",
    )
    parser.add_argument("--max-workers", type=int, default=None, help="线程数，默认等于待解析的 rank 目录数")
    parser.add_argument(
        "--export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="若没有 kernel_details.csv，先离线解析 PROF_*（默认开）。--no-export 跳过",
    )
    return parser.parse_args()


def discover_rank_dirs(root: Path) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    if not root.is_dir():
        raise SystemExit(f"profile dir not found: {root}")
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        m = RANK_DIR_RE.match(path.name)
        if m:
            found.append((int(m.group(1)), path))
    found.sort(key=lambda x: (x[0], str(x[1])))
    return found


def find_ascend_pt_dirs(rank_dir: Path) -> list[Path]:
    return sorted(p for p in rank_dir.rglob("*_ascend_pt") if p.is_dir())


def export_raw_profile(rank_dir: Path) -> str | None:
    """Turn PROF_* raw dumps under rank_dir into ASCEND_PROFILER_OUTPUT.

    Must run on the main process: torch_npu.analyse uses multiprocessing and
    refuses to parse inside a daemon / worker thread.
    """
    pt_dirs = find_ascend_pt_dirs(rank_dir)
    if not pt_dirs:
        logger.warning("no *_ascend_pt under %s", rank_dir)
        return None

    try:
        from torch_npu.profiler.profiler import analyse
    except Exception as e:
        analyse = None
        logger.warning("cannot import torch_npu.profiler.analyse: %s", e)

    if analyse is not None:
        for pt in pt_dirs:
            logger.info("torch_npu.profiler.analyse(%s)", pt)
            try:
                analyse(str(pt))
            except Exception as e:
                logger.warning("analyse failed for %s: %s", pt, e)
                continue
        return "torch_npu.analyse"

    msprof = shutil.which("msprof")
    if not msprof:
        logger.warning("msprof not in PATH, cannot export %s", rank_dir)
        return None
    for pt in pt_dirs:
        cmd = [msprof, "--export=on", f"--output={pt}"]
        logger.info("run: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning("msprof failed (%s): %s", proc.returncode, (proc.stderr or proc.stdout)[-800:])
            return None
    return "msprof"


def _pick(row: dict[str, str], keys: Iterable[str]) -> str | None:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
        for actual in row:
            if actual.strip() == k:
                val = row[actual]
                if val not in (None, ""):
                    return val
    return None


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    s = str(text).strip().replace(",", "")
    if not s or s in ("N/A", "NA", "-", "None"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _duration_us(row: dict[str, str], raw: str | None, key_used: str | None) -> float | None:
    val = _to_float(raw)
    if val is None:
        return None
    if key_used and "ms" in key_used.lower() and "us" not in key_used.lower():
        return val * 1000.0
    return val


def classify_kernel(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ("hcom", "hccl", "alltoall", "all_to_all", "allreduce", "all_reduce",
                             "allgather", "all_gather", "reducescatter", "reduce_scatter",
                             "broadcast", "sendrecv", "p2p")):
        return "HCCL"
    if any(x in n for x in ("flash", "fused_atten", "fusedattention", "promptflash",
                            "increflash", "pagedatten", "mindie", "ifa", "bsa")):
        return "FA"
    if any(x in n for x in ("matmul", "batch_matmul", "gemm", "linear", "bmm")):
        return "MatMul"
    if any(x in n for x in ("layernorm", "rms_norm", "rmsnorm", "addlayernorm", "norm")):
        return "Norm"
    if any(x in n for x in ("softmax", "gelu", "silu", "swiglu", "fastgelu")):
        return "Act"
    if any(x in n for x in ("addcmul", "mul", "add", "cast", "transpose", "copy", "concat")):
        return "Elementwise"
    return "Other"


def _open_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = f.readlines()
    header_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        header_idx = i
        break
    else:
        return []
    reader = csv.DictReader(lines[header_idx:])
    rows = []
    for row in reader:
        cleaned = {(k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
        if cleaned:
            rows.append(cleaned)
    return rows


def find_csv_files(rank_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    files = {p.name: p for p in rank_dir.rglob("*.csv")}
    for name in CSV_CANDIDATES:
        if name in files:
            found[name] = files[name]
    return found


def _aggregate_kernels(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_name: dict[str, dict[str, float]] = defaultdict(lambda: {
        "count": 0.0,
        "total_us": 0.0,
        "max_us": 0.0,
        "mac_us": 0.0,
        "vec_us": 0.0,
        "aicore_us": 0.0,
    })
    name_key_used = None
    dur_key_used = None
    for row in rows:
        raw_name = _pick(row, NAME_KEYS)
        if not raw_name:
            continue
        name_key_used = name_key_used or next((k for k in NAME_KEYS if k in row), "Name")
        dur_raw = None
        dur_key = None
        for k in DURATION_KEYS:
            if k in row and row[k]:
                dur_raw, dur_key = row[k], k
                break
        if dur_raw is None:
            dur_raw = _pick(row, DURATION_KEYS)
        dur_us = _duration_us(row, dur_raw, dur_key)
        count = _to_float(_pick(row, COUNT_KEYS)) or 1.0
        if dur_us is None:
            avg = _to_float(_pick(row, AVG_KEYS))
            if avg is not None:
                dur_us = avg * count
        if dur_us is None:
            continue
        dur_key_used = dur_key_used or dur_key
        stats = by_name[raw_name]
        stats["count"] += count
        stats["total_us"] += dur_us
        stats["max_us"] = max(stats["max_us"], dur_us if count <= 1 else dur_us / max(count, 1.0))
        mac = _to_float(_pick(row, MAC_KEYS))
        vec = _to_float(_pick(row, VEC_KEYS))
        aic = _to_float(_pick(row, AICORE_KEYS))
        if mac is not None:
            stats["mac_us"] += mac
        if vec is not None:
            stats["vec_us"] += vec
        if aic is not None:
            stats["aicore_us"] += aic

    kernels = []
    cat_us: dict[str, float] = defaultdict(float)
    total_us = 0.0
    for name, s in by_name.items():
        cat = classify_kernel(name)
        cat_us[cat] += s["total_us"]
        total_us += s["total_us"]
        avg_us = s["total_us"] / s["count"] if s["count"] else 0.0
        kernels.append({
            "name": name,
            "category": cat,
            "count": int(round(s["count"])),
            "total_ms": round(s["total_us"] / 1000.0, 3),
            "avg_us": round(avg_us, 1),
            "max_us": round(s["max_us"], 1),
            "mac_ms": round(s["mac_us"] / 1000.0, 3) if s["mac_us"] else 0.0,
            "vec_ms": round(s["vec_us"] / 1000.0, 3) if s["vec_us"] else 0.0,
        })
    kernels.sort(key=lambda x: x["total_ms"], reverse=True)
    categories = []
    for cat, us in sorted(cat_us.items(), key=lambda x: -x[1]):
        categories.append({
            "category": cat,
            "total_ms": round(us / 1000.0, 3),
            "pct": round(100.0 * us / total_us, 2) if total_us else 0.0,
        })
    return {
        "n_rows": len(rows),
        "n_unique_kernels": len(kernels),
        "total_kernel_ms": round(total_us / 1000.0, 3),
        "duration_column": dur_key_used,
        "categories": categories,
        "kernels": kernels,
    }


def parse_rank_dir(rank: int, rank_dir: Path, top: int) -> dict[str, Any]:
    csvs = find_csv_files(rank_dir)
    result: dict[str, Any] = {
        "rank": rank,
        "path": str(rank_dir),
        "csv_files": {k: str(v) for k, v in csvs.items()},
        "error": None,
    }
    src = csvs.get("kernel_details.csv") or csvs.get("op_statistic.csv") or csvs.get("op_summary.csv") or csvs.get("operator_details.csv")
    if src is None:
        listed = [str(p.relative_to(rank_dir)) for p in rank_dir.rglob("*") if p.is_file()][:40]
        result["error"] = "no kernel/op csv found"
        result["dir_sample"] = listed
        return result
    try:
        rows = _open_csv_rows(src)
        agg = _aggregate_kernels(rows)
        agg["source_csv"] = src.name
        agg["source_path"] = str(src)
        agg["top_kernels"] = agg["kernels"][:top]
        result.update(agg)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def _print_rank(summary: dict[str, Any], top: int) -> None:
    rank = summary["rank"]
    header = f"=== rank{rank}  {summary['path']} ==="
    print(header)
    if summary.get("error"):
        print(f"  ERROR: {summary['error']}")
        sample = summary.get("dir_sample") or []
        if sample:
            print("  files:")
            for rel in sample:
                print(f"    {rel}")
        print()
        return
    print(f"  csv: {summary.get('source_csv')}  rows={summary.get('n_rows')}  unique={summary.get('n_unique_kernels')}")
    print(f"  total kernel time: {summary.get('total_kernel_ms')} ms")
    print("  by category:")
    for cat in summary.get("categories") or []:
        print(f"    {cat['category']:<12} {cat['total_ms']:>10.3f} ms  {cat['pct']:>6.2f}%")
    print(f"  top {top} kernels:")
    print(f"    {'ms':>10} {'cnt':>6} {'avg_us':>10}  name")
    for k in (summary.get("top_kernels") or [])[:top]:
        print(f"    {k['total_ms']:>10.3f} {k['count']:>6} {k['avg_us']:>10.1f}  {k['name']}")
    print()


def main() -> None:
    args = parse_args()
    root = Path(args.profile_dir).expanduser().resolve()
    jobs = discover_rank_dirs(root)
    if args.ranks is not None:
        want = set(args.ranks)
        jobs = [(r, p) for r, p in jobs if r in want]
        missing = want - {r for r, _ in jobs}
        if missing:
            logger.warning("requested ranks not found under %s: %s", root, sorted(missing))
    if not jobs:
        raise SystemExit(f"no rank* directories found under {root}")

    logger.info("parse %d rank dir(s) under %s", len(jobs), root)
    for rank, path in jobs:
        logger.info("  rank%d -> %s", rank, path)

    if args.export:
        logger.info("offline analyse on main thread (torch_npu.analyse cannot run in worker threads)")
        for rank, path in jobs:
            if find_csv_files(path):
                continue
            exported = export_raw_profile(path)
            logger.info("  rank%d export=%s", rank, exported)

    n_threads = args.max_workers or len(jobs)
    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futs = {pool.submit(parse_rank_dir, rank, path, args.top): (rank, path) for rank, path in jobs}
        for fut in as_completed(futs):
            rank, path = futs[fut]
            try:
                summaries.append(fut.result())
            except Exception as e:
                summaries.append({"rank": rank, "path": str(path), "error": f"{type(e).__name__}: {e}"})

    summaries.sort(key=lambda s: (s.get("rank", -1), s.get("path", "")))
    for s in summaries:
        _print_rank(s, args.top)

    ok = [s for s in summaries if not s.get("error")]
    if len(ok) > 1:
        print("=== ranks compare (total kernel ms / FA / HCCL) ===")
        print(f"  {'rank':<8} {'total_ms':>10} {'FA_ms':>10} {'HCCL_ms':>10}")
        for s in ok:
            cats = {c["category"]: c["total_ms"] for c in s.get("categories") or []}
            print(f"  {s['rank']:<8} {s.get('total_kernel_ms', 0):>10.3f} {cats.get('FA', 0):>10.3f} {cats.get('HCCL', 0):>10.3f}")
        print()

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = []
    for s in summaries:
        item = dict(s)
        kernels = item.pop("kernels", None)
        if kernels is not None:
            item["kernels"] = kernels
        slim.append(item)
        rank_name = Path(s["path"]).name
        parent = Path(s["path"]).parent.name
        fname = f"{parent}_{rank_name}_summary.json" if parent else f"{rank_name}_summary.json"
        (out_dir / fname).write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    all_path = out_dir / "all_ranks.json"
    all_path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("wrote %d summary file(s) + %s", len(summaries), all_path)


if __name__ == "__main__":
    main()
