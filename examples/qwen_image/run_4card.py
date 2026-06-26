"""
NPU Ulysses SPx4 inference entry point.

Each torchrun process runs this script independently,
matching what Worker.__init__ does in the mp.spawn path.

Usage:
    bash run_4card.sh --model_path /path/to/model --prompt "..."

Env vars (set by run_4card.sh):
    TOKENIZERS_PARALLELISM=false  — avoid tokenizer fork deadlocks
    PYTORCH_NPU_ALLOC_CONF='expandable_segments:True' — NPU memory allocator
"""

import argparse
import os

import torch
import torch_npu

from diffsynth_engine.configs import PipelineConfig
from diffsynth_engine.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from diffsynth_engine.engine import DiffSynthEngine


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 1. set_device before init_process_group (HCCL requirement)
    torch_npu.npu.set_device(local_rank)

    # 2. HCCL init + WORLD group
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank)

    # 3. parse args and build config
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--output", type=str, default="output.png")
    args = parser.parse_args()

    config = PipelineConfig(model_path=args.model_path, parallelism=world_size)
    # __post_init__ auto-derives: sp_ulysses_degree=world_size, sp_ring_degree=1

    # 4. SP communication groups — same as Worker.__init__
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

    # 5. dist.is_initialized()=True -> from_pretrained takes _init_pipeline path
    engine = DiffSynthEngine.from_pretrained(config)

    # 6. All ranks must call generate(); rank 0 kwargs are broadcast inside engine
    gen_kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.cfg_scale,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )
    output = engine.generate(**gen_kwargs)
    if rank == 0:
        output.images[0].save(args.output)


if __name__ == "__main__":
    main()
