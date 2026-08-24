"""
4 卡 Ulysses SPx4 图生图推理 + NPU profile — Qwen-Image-Edit-2511

基于 test_v1.py，新增命令行配置的 NPU profiling：
  - --profile              开启 profiling（默认关闭，关闭时行为与 test_v1.py 一致）
  - --profile-dir          输出根目录（默认 ./msprof_output_<timestamp>）
  - --profile-ranks        all=每张卡各出一份 / rank0=仅第 0 卡（默认 all）
  - --profile-start-step   从第几个 DiT step 开始采集（0-based，对应 schedule wait）
  - --profile-num-steps    连续采集几个 DiT step（对应 schedule active，默认 1）

NPU profile（schedule，只采某几步 DiT）:
  warmup 不开；正式 generate 开启。
  skip_first=1 把 VL/VAE encode 从 DiT 窗口里剥出去；
  wait=--profile-start-step, warmup=0, active=--profile-num-steps。
  VAE decode 落在 active 结束之后，不会进 trace。

多卡时通过 engine.start_profile(profile_rank0_only=False) 让每个 worker（每张卡）
各自用 torch_npu.profiler（CPU+NPU）采集，stop_profile 从所有 worker 收集并打印每卡 trace。

示例：
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
  # 只计时（不开 profile）
  python test_v1_profile.py --case-index 2
  # 只采第 0 个 DiT step（4 卡每卡各一份）
  python test_v1_profile.py --case-index 2 --profile
  # 从第 2 个 DiT step 起采 1 步，只采 rank0
  python test_v1_profile.py --case-index 2 --profile \\
      --profile-start-step 2 --profile-num-steps 1 --profile-ranks rank0
  # 指定输出目录
  python test_v1_profile.py --case-index 2 --profile --profile-dir ./prof
"""

import argparse
import logging
import os
import time
from datetime import datetime

import torch
import torch_npu
from PIL import Image

# ── 输出目录（与 0716 / 0807 脚本一致）──
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


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit Ulysses4 single-case inference timing (v1)")
    parser.add_argument("--case-index", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--use-cfg-parallel", action="store_true", default=False)
    parser.add_argument("--sp-ulysses-degree", type=int, default=1)
    parser.add_argument("--sp-ring-degree", type=int, default=1)
    parser.add_argument("--use-torch-compile", action="store_true", default=False)
    parser.add_argument("--use-fsdp", action="store_true", default=False)
    # ── Profiling 配置 ──
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="开启 NPU profiling（schedule 只采指定 DiT step，多卡时每卡各一份）",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Profile 输出根目录（默认 ./msprof_output_<timestamp>）",
    )
    parser.add_argument(
        "--profile-ranks",
        choices=["all", "rank0"],
        default="all",
        help="采集哪些卡的 profile：all=每张卡 / rank0=仅第 0 卡（默认 all）",
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


def _load_images(dataset_dir: str, image_names: list[str]) -> list[Image.Image]:
    images = []
    for name in image_names:
        path = os.path.join(dataset_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"input image not found: {path}")
        images.append(Image.open(path).convert("RGB"))
    return images


def main():
    args = parse_args()
    if args.warmup_steps < 0:
        raise SystemExit(f"--warmup-steps must be >= 0, got {args.warmup_steps}")
    if args.num_inference_steps < 1:
        raise SystemExit(f"--num-inference-steps must be >= 1, got {args.num_inference_steps}")
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

    image_dir = os.path.join(output_root, "output_image")
    os.makedirs(image_dir, exist_ok=True)

    model_root = os.environ.get(
        "QWEN_IMAGE_EDIT_MODEL",
        "/home/lxm/cann_test/Qwen-Image-Edit-2511",
    )
    dataset_dir = os.environ.get("EDIT_DATASET_DIR", "edit_multiple_eval")

    # ── Profiling 输出目录 ──
    profile_dir = args.profile_dir or os.path.abspath(
        f"./msprof_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    profile_rank0_only = args.profile_ranks == "rank0"

    logger.info("加载模型: %s", model_root)
    logger.info("数据集目录: %s", os.path.abspath(dataset_dir))
    logger.info("并行: parallelism=%d, ulysses=%d, ring=%d, cfg_parallel=%s, torch_compile=%s, fsdp=%s",
                args.parallelism, args.sp_ulysses_degree, args.sp_ring_degree,
                args.use_cfg_parallel, args.use_torch_compile, args.use_fsdp)
    logger.info("case-index: %d", args.case_index)
    logger.info("输出目录: %s", output_root)
    if args.profile:
        logger.info(
            "Profile: dir=%s ranks=%s wait=%d active=%d skip_first=1 (DiT steps only)",
            profile_dir,
            args.profile_ranks,
            args.profile_start_step,
            args.profile_num_steps,
        )

    # 与 0807 main 脚本 config 数值一致；v1 用整仓 model_path，pipeline 由 model_index 自动识别。
    # use_zero_cond_t 不在 v1 PipelineConfig 中，2511 的 zero_cond_t 由模型 config 决定。
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
        image_names, prompt, width, height = TEST_LIST[args.case_index]

        if args.warmup_steps > 0:
            warmup_names, warmup_prompt, warmup_w, warmup_h = TEST_LIST[args.case_index]
            logger.info("Warmup: Case %d, %d steps, seed=1", args.case_index, args.warmup_steps)
            engine.generate(
                prompt=warmup_prompt,
                negative_prompt=" ",
                image=_load_images(dataset_dir, warmup_names),
                true_cfg_scale=4.0,
                width=warmup_w,
                height=warmup_h,
                num_inference_steps=args.warmup_steps,
                generator=torch.Generator(device="cpu").manual_seed(1),
            )
            torch_npu.npu.synchronize()
            torch_npu.npu.empty_cache()
            logger.info("Warmup 完成（不计入耗时）")
        else:
            logger.info("跳过 warmup")

        logger.info("=== Case %d (images=%d, %dx%d) ===", args.case_index, len(image_names), width, height)

        # 正式推理前开启 profile（只覆盖正式推理的指定 DiT step，不含 warmup）
        if args.profile:
            engine.start_profile(
                profile_dir,
                profile_rank0_only=profile_rank0_only,
                wait=args.profile_start_step,
                warmup=0,
                active=args.profile_num_steps,
                skip_first=1,
            )
            logger.info(
                "Profiling 已开启: wait=%d active=%d skip_first=1 rank0_only=%s",
                args.profile_start_step,
                args.profile_num_steps,
                profile_rank0_only,
            )

        t0 = time.perf_counter_ns()
        result = engine.generate(
            prompt=prompt,
            negative_prompt=" ",
            image=_load_images(dataset_dir, image_names),
            true_cfg_scale=4.0,
            width=width,
            height=height,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cpu").manual_seed(1),
        )
        torch_npu.npu.synchronize()
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000

        # 正式推理后停止 profile 并收集每张卡的 trace
        if args.profile:
            prof_result = engine.stop_profile()
            traces = prof_result.get("traces", []) if isinstance(prof_result, dict) else []
            for trace in traces:
                logger.info("Profile trace: %s", trace)

        save_path = os.path.join(image_dir, f"case{args.case_index}_ulysses4.jpg")
        result.images[0].save(save_path)
        logger.info("Case %d 输出图: %s", args.case_index, save_path)
        logger.info("Case %d 正式推理耗时(不含 warmup): %.0fms", args.case_index, elapsed_ms)
        logger.info("=== 测试完成（%s）===", "有 profile" if args.profile else "无 profile")
        logger.info("case=%d, elapsed_ms=%.0f", args.case_index, elapsed_ms)
        logger.info("Output: %s", output_root)
    finally:
        engine.shutdown()
        del engine


if __name__ == "__main__":
    main()
