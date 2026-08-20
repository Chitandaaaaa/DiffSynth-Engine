"""
4 卡 CFG×2 + Ulysses×2 图生图 — Ascend e2e profiling（含可选 Python stack）

适配 DiffSynth-Engine 分支 feat/npu-ulysses4-v1-profiling。
case 参数与 0812/test_ulysses4_img2img_v1.py 保持一致。
8 卡 with_stack 已放弃；本脚本默认 4 卡，开 stack 时只采 rank0（与 0804 成功路径一致）。

默认只落盘 raw（快）；kernel_details.csv / trace_view.json 用事后解析：
  python profile_ulysses4_img2img_v1.py --analyse-dir /path/to/profiling/e2e/case2

用法：
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
  export PYTHONPATH=/path/to/DiffSynth-Engine:$PYTHONPATH

  # 4 卡采集（默认全 4 rank，无 stack）
  python profile_ulysses4_img2img_v1.py --case-index 2 --num-inference-steps 10

  # 4 卡 + Python stack（只采 rank0）
  python profile_ulysses4_img2img_v1.py --case-index 2 --num-inference-steps 10 \\
    --with-stack --rank0-only

  # 事后解析
  python profile_ulysses4_img2img_v1.py --analyse-dir /path/to/profiling/e2e/case2
  python profile_ulysses4_img2img_v1.py --analyse-dir /path/to/case2 --analyse-ranks 0
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import torch
import torch_npu
from PIL import Image

# ======================================================================
# 配置（改这里）
# ======================================================================

CASE_INDEX = 2
WARMUP_STEPS = 2
NUM_INFERENCE_STEPS = 10  # profiling 正式推理步数
MASTER_PORT = 29500

# 4 卡：cfg_degree=2 × ulysses=2 × ring=1（与 0804 成功 with_stack 配置一致）
PIPELINE = dict(
    device="npu",
    parallelism=4,
    use_cfg_parallel=True,
    sp_ulysses_degree=2,
    sp_ring_degree=1,
    use_torch_compile=False,
    use_fsdp=False,
)

# Ascend e2e profiler（透传给 engine.start_profile → TorchProfiler）
# Worker 是 daemon 进程，在线 analyse 会被 torch_npu 拒绝。
#
# 采集哪些 rank：
#   PROFILE_RANKS = None  + profile_rank0_only=False → 全卡
#   PROFILE_RANKS = [0,1]                            → 只采部分 rank
#   profile_rank0_only=True / --rank0-only           → 只采 rank0（开 stack 推荐）
PROFILE_RANKS = None

PROFILE = dict(
    profile_rank0_only=False,
    with_stack=False,  # 开 stack：命令行加 --with-stack --rank0-only
    profiler_level="Level1",
    aic_metrics="PipeUtilization",
    data_simplification=True,
    schedule_active=1,
    analyse_flag=False,  # 只落盘 raw
    profile_memory=False,
    record_shapes=True,
)

# 采集流程默认不做 analyse；也可用 --analyse-dir 事后解析
# True 时在本进程解析
OFFLINE_ANALYSE = False
# None=全部 rank；例如 [0] 只解析 rank0
ANALYSE_RANKS = None
# 多 rank 并行 analyse：每批同时解 4 个，剩余下一批（避免一次 8 路打满内存）
ANALYSE_PARALLEL = 4
# 单个 analyse() 内部并发（保持 1，避免嵌套多进程炸内存）
ANALYSE_MAX_PROCESS_NUMBER = 1

TRUE_CFG_SCALE = 4.0
WARMUP_SEED = 0
INFER_SEED = 1

TEST_LIST = [
    (["case_3_input_1.png"], "生成3D毛绒效果，珊瑚绒质感，白色背景", 1024, 1024),
    (["case_1_input_1.jpg", "case_1_input_2.jpg"], "将图一中的柜子替换成图二中的柜子", 1024, 1024),
    (
        [
            "case_28_input_1.png",
            "case_28_input_2.png",
            "case_28_input_3.png",
            "case_28_input_4.png",
        ],
        "请以黑衣男生的第一人称看白衣女生，不要出现男生，镜头正对着女生，女生手上捏着蛊虫，比例 「16:9」",
        1280,
        720,
    ),
]

# ======================================================================

date_str = datetime.now().strftime("%Y%m%d")
time_str = datetime.now().strftime("%H%M%S")
output_root = os.path.abspath(f"./output/{date_str}/{time_str}")

os.environ.setdefault("ASCEND_WORK_PATH", os.path.join(output_root, "ascend_workspace"))
os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit CFG2xUlysses4 Ascend e2e profiling (v1, all ranks)"
    )
    parser.add_argument("--case-index", type=int, choices=[0, 1, 2], default=CASE_INDEX)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--master-port", type=int, default=MASTER_PORT)
    parser.add_argument(
        "--with-stack",
        action="store_true",
        default=PROFILE["with_stack"],
        help="Record Python stacks (use with --rank0-only on 4-card)",
    )
    parser.add_argument(
        "--no-with-stack",
        action="store_false",
        dest="with_stack",
        help="Force with_stack=False",
    )
    parser.add_argument(
        "--analyse-dir",
        type=str,
        default=None,
        help="Only offline-analyse an existing profiling/e2e/caseN directory (no inference)",
    )
    parser.add_argument(
        "--profile-ranks",
        type=str,
        default=None,
        help="Comma-separated ranks to collect, e.g. 0,1,2,3 (8-card run, sample 4). "
        "Default: PROFILE_RANKS / all ranks",
    )
    parser.add_argument(
        "--analyse-ranks",
        type=str,
        default=None,
        help="Comma-separated ranks to analyse, e.g. 0 or 0,1 (default: ANALYSE_RANKS / profile ranks / all)",
    )
    parser.add_argument(
        "--analyse-parallel",
        type=int,
        default=ANALYSE_PARALLEL,
        help="How many ranks to analyse in parallel per batch (default: 4)",
    )
    parser.add_argument(
        "--rank0-only",
        action="store_true",
        default=False,
        help="Only collect profiler on rank0 (others still run inference)",
    )
    parser.add_argument(
        "--offline-analyse",
        action="store_true",
        default=OFFLINE_ANALYSE,
        help="After capture, run batched parallel analyse in this process",
    )
    parser.add_argument(
        "--no-offline-analyse",
        action="store_false",
        dest="offline_analyse",
        help="Dump raw only and exit (default)",
    )
    return parser.parse_args()


def _parse_ranks(text: str | None, fallback: list[int] | None) -> list[int] | None:
    if text is None or text.strip() == "":
        return fallback
    ranks = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ranks.append(int(part))
    return ranks or fallback


def _load_images(dataset_dir: str, image_names: list[str]) -> list[Image.Image]:
    images = []
    for name in image_names:
        path = os.path.join(dataset_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"input image not found: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def _default_trace_dirs(profiling_dir: str, world_size: int) -> list[str]:
    return [os.path.join(profiling_dir, f"e2e_rank{rank}") for rank in range(world_size)]


def _select_analyse_roots(roots: list[str], ranks: list[int] | None) -> list[str]:
    if ranks is None:
        return roots
    rank_set = set(int(r) for r in ranks)
    selected = []
    for root in roots:
        base = os.path.basename(root.rstrip(os.sep))
        # e2e_rank3
        if base.startswith("e2e_rank"):
            try:
                rank = int(base[len("e2e_rank") :])
            except ValueError:
                continue
            if rank in rank_set:
                selected.append(root)
    return selected


def _analyse_one_root_inline(root: str, max_process_number: int) -> tuple[str, float, str | None]:
    """Run analyse in the current (non-daemon) process — preferred for 1 rank."""
    t0 = time.perf_counter()
    try:
        from torch_npu.profiler.profiler import analyse

        analyse(root, max_process_number=max_process_number)
        return root, time.perf_counter() - t0, None
    except Exception as e:
        return root, time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def _analyse_one_root_subprocess(root: str, max_process_number: int) -> tuple[str, float, str | None]:
    """Isolate one rank in a fresh interpreter (avoids nested fork with ProcessPool)."""
    code = (
        "from torch_npu.profiler.profiler import analyse; "
        f"analyse({root!r}, max_process_number={int(max_process_number)})"
    )
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or f"exit={proc.returncode}").strip()
            return root, elapsed, err[-2000:]
        return root, elapsed, None
    except Exception as e:
        return root, time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def _offline_analyse_batched(analyse_roots: list[str], parallel: int) -> None:
    """Analyse ranks in batches of ``parallel``; remaining go to the next batch.

    - 1 root / parallel=1: call analyse() inline in this process (fast path).
    - multi-root: one subprocess per rank (no ProcessPool+fork nesting, which can hang).
    """
    roots = []
    for root in analyse_roots:
        if not root or not os.path.isdir(root):
            logger.warning("skip analyse, path missing: %s", root)
            continue
        roots.append(root)

    if not roots:
        logger.warning("no valid analyse roots")
        return

    parallel = max(1, int(parallel))
    n_batch = (len(roots) + parallel - 1) // parallel
    logger.info(
        "Offline analyse batched: %d roots, parallel=%d → %d batch(es), "
        "analyse_max_process_number=%d",
        len(roots),
        parallel,
        n_batch,
        ANALYSE_MAX_PROCESS_NUMBER,
    )
    t_all = time.perf_counter()
    failures: list[tuple[str, str]] = []

    for batch_idx in range(n_batch):
        batch = roots[batch_idx * parallel : (batch_idx + 1) * parallel]
        logger.info(
            "Analyse batch %d/%d (%d ranks): %s",
            batch_idx + 1,
            n_batch,
            len(batch),
            batch,
        )
        t_batch = time.perf_counter()

        if len(batch) == 1:
            root, elapsed, err = _analyse_one_root_inline(batch[0], ANALYSE_MAX_PROCESS_NUMBER)
            if err:
                failures.append((root, err))
                logger.error("Offline analyse FAILED %s (%.1fs): %s", root, elapsed, err)
            else:
                logger.info("Offline analyse done %s (%.1fs)", root, elapsed)
        else:
            # ThreadPool only waits on subprocesses; each rank is its own python process.
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures = {
                    pool.submit(_analyse_one_root_subprocess, root, ANALYSE_MAX_PROCESS_NUMBER): root
                    for root in batch
                }
                for fut in as_completed(futures):
                    root, elapsed, err = fut.result()
                    if err:
                        failures.append((root, err))
                        logger.error("Offline analyse FAILED %s (%.1fs): %s", root, elapsed, err)
                    else:
                        logger.info("Offline analyse done %s (%.1fs)", root, elapsed)

        logger.info(
            "Analyse batch %d/%d finished in %.1fs",
            batch_idx + 1,
            n_batch,
            time.perf_counter() - t_batch,
        )
        gc.collect()

    logger.info("Offline analyse all done in %.1fs", time.perf_counter() - t_all)
    if failures:
        detail = "; ".join(f"{r}: {e}" for r, e in failures)
        raise RuntimeError(f"analyse failed for {len(failures)} root(s): {detail}")


def run_analyse_only(analyse_dir: str, ranks: list[int] | None, parallel: int) -> None:
    analyse_dir = os.path.abspath(analyse_dir)
    if not os.path.isdir(analyse_dir):
        raise SystemExit(f"--analyse-dir not found: {analyse_dir}")

    roots = sorted(
        os.path.join(analyse_dir, name)
        for name in os.listdir(analyse_dir)
        if name.startswith("e2e_rank") and os.path.isdir(os.path.join(analyse_dir, name))
    )
    if not roots:
        roots = [analyse_dir]
    roots = _select_analyse_roots(roots, ranks)
    if not roots:
        raise SystemExit(f"no e2e_rank* dirs to analyse under {analyse_dir} (ranks={ranks})")
    _offline_analyse_batched(roots, parallel=parallel)


def main():
    args = parse_args()
    profile_ranks = _parse_ranks(args.profile_ranks, PROFILE_RANKS)
    # 未显式指定 analyse ranks 时，默认只解析本次采集的 ranks
    analyse_ranks = _parse_ranks(args.analyse_ranks, ANALYSE_RANKS if ANALYSE_RANKS is not None else profile_ranks)
    analyse_parallel = max(1, int(args.analyse_parallel))

    if args.analyse_dir:
        run_analyse_only(args.analyse_dir, analyse_ranks, analyse_parallel)
        return

    if args.warmup_steps < 0:
        raise SystemExit(f"--warmup-steps must be >= 0, got {args.warmup_steps}")
    if args.num_inference_steps < 1:
        raise SystemExit(f"--num-inference-steps must be >= 1, got {args.num_inference_steps}")

    from diffsynth_engine import DiffSynthEngine
    from diffsynth_engine.configs import QwenImagePipelineConfig

    image_dir = os.path.join(output_root, "output_image")
    profiling_dir = os.path.join(output_root, "profiling", "e2e", f"case{args.case_index}")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(profiling_dir, exist_ok=True)

    model_root = os.environ.get(
        "QWEN_IMAGE_EDIT_MODEL",
        "/data/models/Qwen-Image-Edit-2511",
    )
    dataset_dir = os.environ.get("EDIT_DATASET_DIR", "edit_multiple_eval")

    profile_kwargs = {**PROFILE, "with_stack": args.with_stack}
    if args.with_stack and not args.rank0_only and (profile_ranks is None or len(profile_ranks) > 1):
        logger.warning("with_stack=True → forcing --rank0-only (recommended for 4-card stack)")
        args.rank0_only = True
    if args.rank0_only:
        profile_kwargs["profile_rank0_only"] = True
        profile_kwargs.pop("profile_ranks", None)
        profile_ranks = [0]
    elif profile_ranks is not None:
        profile_kwargs["profile_rank0_only"] = False
        profile_kwargs["profile_ranks"] = profile_ranks
    if args.analyse_ranks is None and ANALYSE_RANKS is None:
        analyse_ranks = profile_ranks
    world_size = int(PIPELINE["parallelism"])

    logger.info("加载模型: %s", model_root)
    logger.info("数据集目录: %s", os.path.abspath(dataset_dir))
    logger.info("PIPELINE: %s", PIPELINE)
    logger.info("PROFILE: %s", profile_kwargs)
    logger.info("case-index: %d", args.case_index)
    logger.info("warmup-steps: %d", args.warmup_steps)
    logger.info("num_inference_steps: %d", args.num_inference_steps)
    logger.info("offline_analyse: %s", args.offline_analyse)
    logger.info("profile_ranks: %s", profile_ranks if profile_ranks is not None else f"all 0..{world_size - 1}")
    logger.info("analyse_ranks: %s", analyse_ranks)
    if args.rank0_only or profile_kwargs.get("profile_rank0_only"):
        logger.info("profiling: rank0 only → %s", profiling_dir)
    elif profile_ranks is not None:
        logger.info("profiling: ranks=%s (of %d) → %s", profile_ranks, world_size, profiling_dir)
    else:
        logger.info("profiling(all ranks=0..%d): %s", world_size - 1, profiling_dir)
    logger.info("输出目录: %s", output_root)

    config = QwenImagePipelineConfig(
        model_path=model_root,
        **PIPELINE,
    )

    engine = DiffSynthEngine.from_pretrained(config, master_port=args.master_port)
    try:
        image_names, prompt, width, height = TEST_LIST[args.case_index]
        gen_common = dict(
            prompt=prompt,
            negative_prompt=" ",
            image=_load_images(dataset_dir, image_names),
            true_cfg_scale=TRUE_CFG_SCALE,
            width=width,
            height=height,
        )

        if args.warmup_steps > 0:
            logger.info(
                "Warmup: Case %d (images=%d, %dx%d), %d steps, seed=%d",
                args.case_index,
                len(image_names),
                width,
                height,
                args.warmup_steps,
                WARMUP_SEED,
            )
            engine.generate(
                **gen_common,
                num_inference_steps=args.warmup_steps,
                generator=torch.Generator(device="cpu").manual_seed(WARMUP_SEED),
            )
            torch_npu.npu.synchronize()
            torch_npu.npu.empty_cache()
            logger.info("Warmup 完成（不计入 profiling）")
        else:
            logger.info("跳过 warmup")

        profile_template = os.path.join(profiling_dir, "e2e")
        engine.start_profile(path=profile_template, **profile_kwargs)
        logger.info("all-rank profiler started: %s", profile_kwargs)

        logger.info(
            "=== Case %d e2e profile (images=%d, %dx%d, steps=%d, ranks=%d) ===",
            args.case_index,
            len(image_names),
            width,
            height,
            args.num_inference_steps,
            world_size,
        )
        t0 = time.perf_counter_ns()
        result = engine.generate(
            **gen_common,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cpu").manual_seed(INFER_SEED),
        )
        torch_npu.npu.synchronize()
        infer_ms = (time.perf_counter_ns() - t0) / 1_000_000

        t_stop = time.perf_counter()
        profile_result = engine.stop_profile()
        stop_s = time.perf_counter() - t_stop
        traces = [t for t in (profile_result or {}).get("traces", []) if t]
        logger.info("stop_profile done in %.1fs, traces=%d", stop_s, len(traces))

        save_path = os.path.join(image_dir, f"case{args.case_index}_cfg2_uly4_profile.jpg")
        result.images[0].save(save_path)
        logger.info("输出图: %s", save_path)
        logger.info("infer_ms=%.0f (host wall; includes all-rank profiler overhead)", infer_ms)
        logger.info("Profile traces: %s", traces)

        logger.info("Shutting down engine...")
        engine.shutdown()
        del engine
        engine = None
        try:
            torch_npu.npu.empty_cache()
        except Exception:
            pass
        gc.collect()

        analyse_roots = _select_analyse_roots(
            traces or _default_trace_dirs(profiling_dir, world_size),
            analyse_ranks,
        )
        if args.offline_analyse:
            logger.info(
                "Starting offline analyse (batched parallel=%d)...",
                analyse_parallel,
            )
            _offline_analyse_batched(analyse_roots, parallel=analyse_parallel)
        else:
            logger.info(
                "Skip offline analyse (raw dump only). Later run:\n"
                "  python %s --analyse-dir %s --analyse-parallel %d",
                os.path.basename(__file__),
                profiling_dir,
                analyse_parallel,
            )

        meta_path = os.path.join(output_root, "profiling_meta.txt")
        ranks_meta = (
            ",".join(str(r) for r in profile_ranks)
            if profile_ranks is not None
            else f"0-{world_size - 1}"
        )
        with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("script=profile_ulysses4_img2img_v1\n")
            f.write("engine_branch=feat/npu-ulysses4-v1-profiling\n")
            f.write(f"case_index={args.case_index}\n")
            f.write("profile_mode=e2e\n")
            f.write(f"profile_ranks={ranks_meta}\n")
            f.write(f"profile_world_size={world_size}\n")
            for k, v in PIPELINE.items():
                f.write(f"{k}={v}\n")
            for k, v in profile_kwargs.items():
                f.write(f"profile_{k}={v}\n")
            f.write(f"warmup_steps={args.warmup_steps}\n")
            f.write("warmup_same_case=1\n")
            f.write(f"num_inference_steps={args.num_inference_steps}\n")
            f.write(f"infer_ms={infer_ms:.0f}\n")
            f.write(f"stop_profile_s={stop_s:.1f}\n")
            f.write("api=DiffSynthEngine.start_profile\n")
            f.write(f"offline_analyse={int(args.offline_analyse)}\n")
            f.write(f"analyse_max_process_number={ANALYSE_MAX_PROCESS_NUMBER}\n")
            f.write(f"analyse_parallel={analyse_parallel}\n")
            f.write(f"analyse_ranks={analyse_ranks}\n")
            f.write(f"profile_rank0_only={int(bool(profile_kwargs.get('profile_rank0_only')))}\n")
            f.write(f"profiling_dir={profiling_dir}\n")
            f.write(f"traces={traces}\n")
            f.write(f"output_dir={output_root}\n")

        logger.info("=== e2e profiling 完成 ===")
        logger.info("Trace ranks=%s → %s", ranks_meta, profiling_dir)
        logger.info("Meta: %s", meta_path)
        logger.info("Output: %s", output_root)
    finally:
        if engine is not None:
            engine.shutdown()
            del engine


if __name__ == "__main__":
    main()
