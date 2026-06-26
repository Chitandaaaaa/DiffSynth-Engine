"""
NPU 4-card Ulysses SP verification script.

Runs staged checks so you can isolate failures before loading the full model.

Stages:
  dist   — HCCL init, SP groups, kwargs broadcast (no model)
  comm   — all_to_all_4D roundtrip on NPU tensors (no model)
  infer  — end-to-end DiffSynthEngine inference
  all    — dist + comm + infer

Usage:
    bash test_4card.sh --stage dist
    bash test_4card.sh --stage comm
    bash test_4card.sh --stage infer --model_path /path/to/model
    bash test_4card.sh --stage all --model_path /path/to/model
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch_npu

from diffsynth_engine.configs import PipelineConfig
from diffsynth_engine.distributed.comm import all_to_all_4D
from diffsynth_engine.distributed.parallel_state import (
    get_ulysses_parallel_world_size,
    get_ulysses_process_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from diffsynth_engine.engine import DiffSynthEngine
from diffsynth_engine.utils.platform import get_device


def log(rank: int, message: str) -> None:
    print(f"[rank {rank}] {message}", flush=True)


def init_parallel_groups(world_size: int, rank: int, local_rank: int) -> None:
    torch_npu.npu.set_device(local_rank)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank)

    config = PipelineConfig(model_path="", parallelism=world_size)
    cfg_degree = 2 if config.use_cfg_parallel else 1
    sp_ulysses_degree = config.sp_ulysses_degree
    sp_ring_degree = config.sp_ring_degree
    sp_degree = sp_ulysses_degree * sp_ring_degree
    tp_degree = config.tp_degree
    vae_parallel_size = world_size if config.use_vae_parallel else 0

    initialize_model_parallel(
        classifier_free_guidance_degree=cfg_degree,
        sequence_parallel_degree=sp_degree,
        ulysses_degree=sp_ulysses_degree,
        ring_degree=sp_ring_degree,
        tensor_parallel_degree=tp_degree,
        vae_parallel_size=vae_parallel_size,
    )


def test_dist(rank: int) -> None:
    if not dist.is_initialized():
        raise RuntimeError("dist is not initialized")

    world_size = dist.get_world_size()
    ulysses_size = get_ulysses_parallel_world_size()
    if ulysses_size != world_size:
        raise RuntimeError(f"ulysses world size {ulysses_size} != world size {world_size}")

    world_group = get_world_group()
    if rank == 0:
        payload = {
            "method": "__call__",
            "kwargs": {"prompt": "broadcast_check", "seed": 42},
        }
        world_group.broadcast_tensor_dict(payload, src=0)
    else:
        payload = world_group.broadcast_tensor_dict(src=0)

    if payload["kwargs"]["prompt"] != "broadcast_check":
        raise RuntimeError(f"broadcast mismatch on rank {rank}: {payload}")

    world_group.barrier()
    log(rank, "PASS dist: HCCL init, SP groups, broadcast_tensor_dict")


def test_comm(local_rank: int, rank: int) -> None:
    device = get_device(local_rank)
    group = get_ulysses_process_group()
    sp_size = dist.get_world_size(group)

    bs, total_seq, hc, hs = 1, 32, 24, 64
    if total_seq % sp_size != 0:
        raise RuntimeError(f"total_seq={total_seq} not divisible by sp_size={sp_size}")
    if hc % sp_size != 0:
        raise RuntimeError(f"hc={hc} not divisible by sp_size={sp_size}")

    local_seq = total_seq // sp_size
    x = torch.randn(bs, local_seq, hc, hs, device=device, dtype=torch.bfloat16)

    y = all_to_all_4D(x, scatter_idx=2, gather_idx=1, group=group)
    expected_fwd = (bs, total_seq, hc // sp_size, hs)
    if y.shape != expected_fwd:
        raise RuntimeError(f"all_to_all forward shape {tuple(y.shape)} != {expected_fwd}")

    z = all_to_all_4D(y, scatter_idx=1, gather_idx=2, group=group)
    expected_bwd = (bs, local_seq, hc, hs)
    if z.shape != expected_bwd:
        raise RuntimeError(f"all_to_all backward shape {tuple(z.shape)} != {expected_bwd}")

    get_world_group().barrier()
    log(rank, "PASS comm: all_to_all_4D roundtrip with device_synchronize")


def test_infer(args: argparse.Namespace, rank: int) -> None:
    if not args.model_path:
        raise RuntimeError("--model_path is required for infer stage")

    config = PipelineConfig(model_path=args.model_path, parallelism=args.world_size)
    engine = DiffSynthEngine.from_pretrained(config)

    gen_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.cfg_scale,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )

    if args.test_rank0_broadcast:
        if rank == 0:
            output = engine.generate(**gen_kwargs)
        else:
            output = engine.generate()
    else:
        output = engine.generate(**gen_kwargs)

    if rank == 0:
        if output is None:
            raise RuntimeError("rank 0 received None from generate()")
        if not hasattr(output, "images") or len(output.images) == 0:
            raise RuntimeError("generate() returned no images")
        output.images[0].save(args.output)
        log(rank, f"PASS infer: saved image to {args.output}")
    else:
        if output is not None:
            raise RuntimeError(f"rank {rank} expected None from generate(), got {type(output)}")
        log(rank, "PASS infer: non-zero rank finished forward")


def run_stage(stage: str, args: argparse.Namespace, rank: int, local_rank: int) -> None:
    if stage == "dist":
        test_dist(rank)
    elif stage == "comm":
        test_comm(local_rank, rank)
    elif stage == "infer":
        test_infer(args, rank)
    else:
        raise ValueError(f"unknown stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NPU 4-card SP verification")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["dist", "comm", "infer", "all"],
        help="which check to run",
    )
    parser.add_argument("--model_path", type=str, default="", help="required for infer/all")
    parser.add_argument("--prompt", type=str, default="A cat sitting on a windowsill")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--output", type=str, default="test_4card_output.png")
    parser.add_argument(
        "--test_rank0_broadcast",
        action="store_true",
        help="infer stage: only rank 0 passes kwargs, others call generate() with no args",
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    args.world_size = world_size

    if args.stage in ("infer", "all") and not args.model_path:
        if rank == 0:
            print("ERROR: --model_path is required for infer/all stage", file=sys.stderr)
        sys.exit(1)

    init_parallel_groups(world_size, rank, local_rank)

    stages = ["dist", "comm", "infer"] if args.stage == "all" else [args.stage]
    t0 = time.perf_counter()

    try:
        for stage in stages:
            if rank == 0:
                print(f"\n=== stage: {stage} ===", flush=True)
            run_stage(stage, args, rank, local_rank)
    except Exception as exc:
        log(rank, f"FAIL {exc}")
        raise

    get_world_group().barrier()
    if rank == 0:
        elapsed = time.perf_counter() - t0
        print(f"\nALL PASSED ({', '.join(stages)}) in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
