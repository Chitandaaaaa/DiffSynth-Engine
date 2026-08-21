"""
4 卡 Ulysses SPx4 — Qwen-Image-Edit DiT mock 长序列降噪

只跑 DiT 降噪循环，跳过 VAE / VL encoder。
joint 序列: total_seq_len = text_len + (latents_len + image_latents_len)
latents 由 1280x720 打包得到；剩余 token 分给 prompt_embeds 与 image_latents。

NPU profile（schedule，只采某几步）:
  warmup / 前几次 repeat 不开；仅每个 seq 的最后一次 repeat 开启。
  wait=--profile-start-step, warmup=0, active=--profile-num-steps。

示例:
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
  export PYTHONPATH=/path/to/DiffSynth-Engine:$PYTHONPATH
  python mytest/mock/test_dit_longseq.py
  python mytest/mock/test_dit_longseq.py --total-seq-len 50000
  python mytest/mock/test_dit_longseq.py --total-seq-len 50000 100000 --warmup-steps 1 --num-inference-steps 4
  python mytest/mock/test_dit_longseq.py --total-seq-len 50000 --profile \\
      --profile-start-step 2 --profile-num-steps 1 --profile-ranks rank0

  python  --parallelism 8 --sp-ulysses-degree 4  --use-cfg-parallel  --total-seq-len 50000    --profile  --profile-start-step 0 --profile-num-steps 1
"""

import argparse
import logging
import os
import statistics
import sys
import time
from datetime import datetime

import torch
import torch_npu

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

date_str = datetime.now().strftime("%Y%m%d")
time_str = datetime.now().strftime("%H%M%S")
output_root = os.path.abspath(f"./output/{date_str}/{time_str}")

os.environ.setdefault("ASCEND_WORK_PATH", os.path.join(output_root, "ascend_workspace"))
os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
os.environ.setdefault("TASK_QUEUE_ENABLE", "2")
os.environ.setdefault("CPU_AFFINITY_CONF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit Ulysses4 DiT mock long-seq timing")
    parser.add_argument(
        "--total-seq-len",
        type=int,
        nargs="+",
        default=[50000, 100000],
        help="joint attention lengths: text_len + latents_len + image_latents_len",
    )
    parser.add_argument("--text-seq-len", type=int, default=512)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--interval-s", type=float, default=2.0)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--use-cfg-parallel", action="store_true", default=False)
    parser.add_argument("--sp-ulysses-degree", type=int, default=4)
    parser.add_argument("--sp-ring-degree", type=int, default=1)
    parser.add_argument("--use-torch-compile", action="store_true", default=False)
    parser.add_argument("--use-fsdp", action="store_true", default=False)
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="开启 NPU profiling（仅每个 seq 的最后一次 repeat，schedule 只采指定 DiT step）",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Profile 输出根目录（默认 ./msprof_output_<timestamp>，每个 seq 写入 seq<len>/）",
    )
    parser.add_argument(
        "--profile-ranks",
        choices=["all", "rank0"],
        default="rank0",
        help="采集哪些卡：all=每张卡 / rank0=仅第 0 卡（长序列默认 rank0）",
    )
    parser.add_argument(
        "--profile-start-step",
        type=int,
        default=0,
        help="从第几个 DiT step 开始采集（0-based，对应 schedule wait）",
    )
    parser.add_argument(
        "--profile-num-steps",
        type=int,
        default=1,
        help="连续采集几个 DiT step（对应 schedule active，默认 1）",
    )
    return parser.parse_args()


def _run_once(engine, args, total_seq_len: int, num_inference_steps: int) -> dict:
    return engine.run_dit_mock_denoise(
        total_seq_len=total_seq_len,
        height=args.height,
        width=args.width,
        text_seq_len=args.text_seq_len,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        seed=args.seed,
    )


def main():
    args = parse_args()
    if args.warmup_steps < 0:
        raise SystemExit(f"--warmup-steps must be >= 0, got {args.warmup_steps}")
    if args.num_inference_steps < 1:
        raise SystemExit(f"--num-inference-steps must be >= 1, got {args.num_inference_steps}")
    if args.num_repeats < 1:
        raise SystemExit(f"--num-repeats must be >= 1, got {args.num_repeats}")
    if any(seq_len <= 0 for seq_len in args.total_seq_len):
        raise SystemExit(f"--total-seq-len must be positive, got {args.total_seq_len}")
    if args.profile_start_step < 0:
        raise SystemExit(f"--profile-start-step must be >= 0, got {args.profile_start_step}")
    if args.profile_num_steps < 1:
        raise SystemExit(f"--profile-num-steps must be >= 1, got {args.profile_num_steps}")
    if args.profile_start_step + args.profile_num_steps > args.num_inference_steps:
        raise SystemExit(
            f"--profile-start-step ({args.profile_start_step}) + --profile-num-steps "
            f"({args.profile_num_steps}) exceeds --num-inference-steps ({args.num_inference_steps})"
        )

    from diffsynth_engine import DiffSynthEngine
    from diffsynth_engine.configs import QwenImagePipelineConfig
    from diffsynth_engine.layers.attention import AttentionType

    os.makedirs(output_root, exist_ok=True)

    model_root = os.environ.get(
        "QWEN_IMAGE_EDIT_MODEL",
        "/home/lxm/cann_test/Qwen-Image-Edit-2511",
    )

    logger.info("加载模型: %s", model_root)
    logger.info(
        "并行: parallelism=%d, ulysses=%d, ring=%d, cfg_parallel=%s, torch_compile=%s, fsdp=%s",
        args.parallelism,
        args.sp_ulysses_degree,
        args.sp_ring_degree,
        args.use_cfg_parallel,
        args.use_torch_compile,
        args.use_fsdp,
    )
    logger.info(
        "mock: total_seq_len=%s text_seq_len=%d resolution=%dx%d cfg=%.1f steps=%d",
        args.total_seq_len,
        args.text_seq_len,
        args.width,
        args.height,
        args.true_cfg_scale,
        args.num_inference_steps,
    )
    logger.info("输出目录: %s", output_root)

    profile_dir = args.profile_dir or os.path.abspath(
        f"./msprof_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    profile_rank0_only = args.profile_ranks == "rank0"
    if args.profile:
        logger.info(
            "Profile: dir=%s ranks=%s wait=%d active=%d (last repeat only)",
            profile_dir,
            args.profile_ranks,
            args.profile_start_step,
            args.profile_num_steps,
        )

    config = QwenImagePipelineConfig(
        model_path=model_root,
        device="npu",
        parallelism=args.parallelism,
        use_cfg_parallel=args.use_cfg_parallel,
        sp_ulysses_degree=args.sp_ulysses_degree,
        sp_ring_degree=args.sp_ring_degree,
        use_torch_compile=args.use_torch_compile,
        use_fsdp=args.use_fsdp,
    )
    config.attn_type = AttentionType.MINDIE

    engine = DiffSynthEngine.from_pretrained(config, master_port=args.master_port)
    try:
        for total_seq_len in args.total_seq_len:
            logger.info("=== Mock joint_seq=%d (%dx%d latents) ===", total_seq_len, args.width, args.height)

            if args.warmup_steps > 0:
                logger.info("Warmup: joint_seq=%d, %d steps, seed=%d", total_seq_len, args.warmup_steps, args.seed)
                _run_once(engine, args, total_seq_len, args.warmup_steps)
                torch_npu.npu.synchronize()
                torch_npu.npu.empty_cache()
                logger.info("Warmup 完成（不计入耗时）")
            else:
                logger.info("跳过 warmup")

            elapsed_list = []
            last_stats = None
            for i in range(args.num_repeats):
                do_profile = args.profile and i == args.num_repeats - 1
                if do_profile:
                    seq_profile_dir = os.path.join(profile_dir, f"seq{total_seq_len}")
                    engine.start_profile(
                        seq_profile_dir,
                        profile_rank0_only=profile_rank0_only,
                        wait=args.profile_start_step,
                        warmup=0,
                        active=args.profile_num_steps,
                    )
                    logger.info(
                        "Profiling 已开启: %s wait=%d active=%d rank0_only=%s",
                        seq_profile_dir,
                        args.profile_start_step,
                        args.profile_num_steps,
                        profile_rank0_only,
                    )
                try:
                    t0 = time.perf_counter_ns()
                    last_stats = _run_once(engine, args, total_seq_len, args.num_inference_steps)
                    torch_npu.npu.synchronize()
                    elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
                    elapsed_list.append(elapsed_ms)
                finally:
                    if do_profile:
                        prof_result = engine.stop_profile()
                        traces = prof_result.get("traces", []) if isinstance(prof_result, dict) else []
                        for trace in traces:
                            logger.info("Profile trace: %s", trace)
                logger.info(
                    "joint_seq=%d 第 %d/%d 次耗时: %.2fms%s",
                    total_seq_len,
                    i + 1,
                    args.num_repeats,
                    elapsed_ms,
                    "（含 profile）" if do_profile else "",
                )
                if i < args.num_repeats - 1 and args.interval_s > 0:
                    time.sleep(args.interval_s)

            total_ms = sum(elapsed_list)
            avg_ms = total_ms / len(elapsed_list)
            med_ms = statistics.median(elapsed_list)
            logger.info(
                "split: requested=%s actual_joint=%s text=%s latents=%s image_latents=%s img_shapes=%s",
                last_stats["requested_total_seq_len"],
                last_stats["joint_seq_len"],
                last_stats["text_len"],
                last_stats["latents_len"],
                last_stats["image_latents_len"],
                last_stats["img_shapes"],
            )
            logger.info(
                "joint_seq=%d 正式推理 %d 次: median=%.0fms, avg=%.0fms, min=%.0fms, max=%.0fms",
                total_seq_len,
                args.num_repeats,
                med_ms,
                avg_ms,
                min(elapsed_list),
                max(elapsed_list),
            )
            logger.info(
                "joint_seq=%d raw %d次耗时(ms): %s",
                total_seq_len,
                args.num_repeats,
                ", ".join(f"{x:.2f}" for x in elapsed_list),
            )

        logger.info("=== DiT mock 长序列测试完成 ===")
        logger.info("Output: %s", output_root)
    finally:
        engine.shutdown()
        del engine


if __name__ == "__main__":
    main()
