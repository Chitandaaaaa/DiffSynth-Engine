"""
4 卡 Ulysses SPx4 图生图推理（无 profiling）— Qwen-Image-Edit-2511

适配 DiffSynth-Engine v1 线 DiffSynthEngine（feat/npu-ulysses4-v1）。
使用内置 mp.spawn 多进程，不需要 torchrun。
并行配置：与 0807/test_ulysses4_img2img_main.py 中 config 数值保持一致。

case 参数（TEST_LIST / seed / cfg / steps）与 0807 main 脚本保持一致。

示例：
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
  # 工作目录需能 import 当前分支的 diffsynth_engine，例如：
  #   export PYTHONPATH=/path/to/DiffSynth-Engine:$PYTHONPATH
  python test_v1.py --case-index 2
  python test_v1.py --case-index 0 --warmup-steps 0 --num-inference-steps 40
"""

import argparse
import logging
import os
import statistics
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

    logger.info("加载模型: %s", model_root)
    logger.info("数据集目录: %s", os.path.abspath(dataset_dir))
    logger.info("并行: parallelism=%d, ulysses=%d, ring=%d, cfg_parallel=%s, torch_compile=%s, fsdp=%s",
                args.parallelism, args.sp_ulysses_degree, args.sp_ring_degree,
                args.use_cfg_parallel, args.use_torch_compile, args.use_fsdp)
    logger.info("case-index: %d", args.case_index)
    logger.info("输出目录: %s", output_root)

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

        # 正式推理重复采集 5 次，每次间隔 2s，记录单次/总/平均耗时
        num_repeats = 5
        interval_s = 2
        elapsed_list = []
        result = None
        for i in range(num_repeats):
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
            elapsed_list.append(elapsed_ms)
            logger.info("Case %d 第 %d/%d 次耗时: %.2fms", args.case_index, i + 1, num_repeats, elapsed_ms)
            if i < num_repeats - 1:
                time.sleep(interval_s)

        total_ms = sum(elapsed_list)
        avg_ms = total_ms / len(elapsed_list)
        med_ms = statistics.median(elapsed_list)

        save_path = os.path.join(image_dir, f"case{args.case_index}_ulysses4.jpg")
        result.images[0].save(save_path)
        logger.info("Case %d 输出图: %s", args.case_index, save_path)
        logger.info(
            "Case %d 正式推理 %d 次: median=%.0fms, avg=%.0fms, min=%.0fms, max=%.0fms",
            args.case_index, num_repeats, med_ms, avg_ms, min(elapsed_list), max(elapsed_list),
        )
        logger.info("Case %d raw 5次耗时(ms): %s", args.case_index, ", ".join(f"{x:.2f}" for x in elapsed_list))
        logger.info("=== 测试完成（无 profiling）===")
        logger.info(
            "case=%d, total_ms=%.0f, median_ms=%.0f, avg_ms=%.0f, min_ms=%.0f, max_ms=%.0f",
            args.case_index, total_ms, med_ms, avg_ms, min(elapsed_list), max(elapsed_list),
        )
        logger.info("Output: %s", output_root)
    finally:
        engine.shutdown()
        del engine


if __name__ == "__main__":
    main()
