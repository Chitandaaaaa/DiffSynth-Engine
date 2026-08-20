# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM-Omni vllm_omni/diffusion/profiler/torch_profiler.py.

import os
import subprocess
from contextlib import nullcontext
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


def _is_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        npu = getattr(torch, "npu", None)
        return npu is not None and npu.is_available()
    except Exception:
        return False


class TorchProfiler:
    """
    Torch-based profiler configured for End-to-End continuous recording.

    CUDA: chrome-trace export (+ optional gzip).
    NPU: torch_npu + tensorboard_trace_handler; knobs come from start() kwargs
    (typically set by the caller / profiling script).
    """

    _profiler: Any | None = None
    _trace_template: str = ""
    _backend: str = "cuda"  # "cuda" | "npu"
    _with_stack: bool = True

    @classmethod
    def start(
        cls,
        trace_path_template: str,
        profile_rank0_only: bool = True,
        with_stack: bool = True,
        profile_ranks: list[int] | tuple[int, ...] | None = None,
        **npu_kwargs,
    ) -> str:
        """
        Start the profiler with the given trace path template.

        Rank selection:
          - profile_ranks=[0,1,2,3]: only those ranks collect (8-card run, sample 4)
          - else profile_rank0_only=True: only rank0
          - else: all ranks

        Extra kwargs (NPU only, optional):
          profiler_level, aic_metrics, data_simplification, schedule_active,
          analyse_flag (default False: dump raw only; call analyse() in a
          non-daemon process afterwards — worker processes are daemons),
          profile_memory, record_shapes
        """
        if cls._profiler is not None:
            logger.warning("[Rank %s] Stopping existing Torch profiler", cls._get_rank())
            try:
                cls._profiler.stop()
            except Exception:
                pass
            cls._profiler = None

        rank = cls._get_rank()
        trace_path_template = os.path.abspath(trace_path_template)
        cls._trace_template = trace_path_template
        cls._with_stack = bool(with_stack)

        if profile_ranks is not None:
            allow = {int(r) for r in profile_ranks}
            if rank not in allow:
                logger.info(
                    "[Rank %s] Skip profiler (profile_ranks=%s)",
                    rank,
                    sorted(allow),
                )
                return ""
        elif profile_rank0_only and rank != 0:
            return ""

        if _is_npu_available():
            return cls._start_npu(
                rank,
                trace_path_template,
                with_stack=cls._with_stack,
                **npu_kwargs,
            )
        return cls._start_cuda(rank, trace_path_template, with_stack=cls._with_stack)

    @classmethod
    def _start_cuda(cls, rank: int, trace_path_template: str, *, with_stack: bool) -> str:
        cls._backend = "cuda"
        json_file = f"{trace_path_template}_rank{rank}.json"
        os.makedirs(os.path.dirname(json_file) or ".", exist_ok=True)
        logger.info(
            "[Rank %s] Starting End-to-End Torch profiler (cuda, with_stack=%s)",
            rank,
            with_stack,
        )

        def trace_handler(p):
            nonlocal json_file
            try:
                p.export_chrome_trace(json_file)
                logger.info(f"[Rank {rank}] Trace exported to {json_file}")
                try:
                    subprocess.Popen(["gzip", "-f", json_file])
                    logger.info(f"[Rank {rank}] Triggered background compression for {json_file}")
                    json_file = f"{json_file}.gz"
                except Exception as compress_err:
                    logger.warning(f"[Rank {rank}] Background gzip failed to start: {compress_err}")
            except Exception as e:
                logger.warning(f"[Rank {rank}] Failed to export trace: {e}")

        cls._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=100000,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=with_stack,
            with_flops=True,
        )
        cls._profiler.start()
        return f"{trace_path_template}_rank{rank}.json.gz"

    @classmethod
    def _start_npu(
        cls,
        rank: int,
        trace_path_template: str,
        *,
        with_stack: bool,
        profiler_level: str = "Level1",
        aic_metrics: str = "PipeUtilization",
        data_simplification: bool = True,
        schedule_active: int = 1,
        # Workers are daemon processes; online analyse() is rejected by torch_npu.
        analyse_flag: bool = False,
        profile_memory: bool = True,
        record_shapes: bool = True,
        **_ignored,
    ) -> str:
        import torch_npu

        cls._backend = "npu"
        output_dir = f"{trace_path_template}_rank{rank}"
        os.makedirs(output_dir, exist_ok=True)
        analyse_flag = bool(analyse_flag)
        profile_memory = bool(profile_memory)
        record_shapes = bool(record_shapes)
        logger.info(
            "[Rank %s] Starting End-to-End Torch profiler "
            "(npu, with_stack=%s, level=%s, active=%s, analyse_flag=%s, "
            "profile_memory=%s) → %s",
            rank,
            with_stack,
            profiler_level,
            schedule_active,
            analyse_flag,
            profile_memory,
            output_dir,
        )

        level = getattr(torch_npu.profiler.ProfilerLevel, profiler_level)
        metrics = getattr(torch_npu.profiler.AiCMetrics, aic_metrics)
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=level,
            aic_metrics=metrics,
            data_simplification=bool(data_simplification),
            msprof_tx=False,
            l2_cache=False,
            op_attr=False,
            record_op_args=False,
        )
        cls._profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0,
                warmup=0,
                active=int(schedule_active),
                repeat=1,
                skip_first=0,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                output_dir,
                analyse_flag=analyse_flag,
            ),
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=False,
            experimental_config=experimental_config,
        )
        cls._profiler.start()
        return output_dir

    @classmethod
    def stop(cls) -> dict | None:
        if cls._profiler is None:
            return None

        rank = cls._get_rank()
        base_path = f"{cls._trace_template}_rank{rank}"
        backend = cls._backend

        try:
            if backend == "npu":
                import torch_npu

                torch_npu.npu.synchronize()
                cls._profiler.step()
            cls._profiler.stop()
        except Exception as e:
            logger.warning(f"[Rank {rank}] Profiler stop failed: {e}")

        cls._profiler = None

        if backend == "npu":
            return {"trace": base_path, "table": None}
        return {"trace": f"{base_path}.json.gz", "table": None}

    @classmethod
    def step(cls):
        if cls._profiler is not None:
            cls._profiler.step()

    @classmethod
    def is_active(cls) -> bool:
        return cls._profiler is not None

    @classmethod
    def get_step_context(cls):
        return nullcontext()

    @classmethod
    def _get_rank(cls) -> int:
        return int(os.getenv("RANK", "0"))
