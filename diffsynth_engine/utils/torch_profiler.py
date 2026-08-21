# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM-Omni vllm_omni/diffusion/profiler/torch_profiler.py.

import os
import subprocess
from contextlib import nullcontext
from typing import Any

import torch

from diffsynth_engine.utils import logging

logger = logging.get_logger(__name__)


def _use_npu_profiler() -> bool:
    try:
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _npu_experimental_config(torch_npu_mod):
    """Level1 + PipeUtilization, with fallbacks for older torch_npu APIs."""
    ExperimentalConfig = torch_npu_mod.profiler._ExperimentalConfig
    attempts = [
        dict(
            export_type=torch_npu_mod.profiler.ExportType.Text,
            profiler_level=torch_npu_mod.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu_mod.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            data_simplification=False,
        ),
        dict(
            profiler_level=torch_npu_mod.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu_mod.profiler.AiCMetrics.PipeUtilization,
        ),
        dict(),
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return ExperimentalConfig(**kwargs)
        except (TypeError, AttributeError) as e:
            last_error = e
            continue
    logger.warning("Falling back to empty torch_npu _ExperimentalConfig: %s", last_error)
    return ExperimentalConfig()


def _make_profiler_schedule(profiler_mod, wait: int, warmup: int, active: int, skip_first: int = 0, repeat: int = 1):
    """Build wait/warmup/active schedule; try torch_npu then torch.profiler."""
    schedule_fn = getattr(profiler_mod, "schedule", None)
    if schedule_fn is None:
        schedule_fn = torch.profiler.schedule
    attempts = [
        dict(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first),
        dict(wait=wait, warmup=warmup, active=active, repeat=repeat),
        dict(wait=wait, warmup=warmup, active=active),
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return schedule_fn(**kwargs)
        except TypeError as e:
            last_error = e
            continue
    logger.warning("Failed to build profiler schedule: %s", last_error)
    return None


class TorchProfiler:
    """
    End-to-end profiler.

    On NPU uses torch_npu.profiler (CPU + NPU, Level1 + PipeUtilization) and
    writes MindStudio/TensorBoard ``*_ascend_pt`` directories.
    On CUDA keeps the original torch.profiler chrome-trace path.
    """

    _profiler: Any | None = None
    _trace_template: str = ""
    _trace_path: str = ""
    _backend: str = ""

    @classmethod
    def start(
        cls,
        trace_path_template: str,
        profile_rank0_only: bool = True,
        wait: int = 0,
        warmup: int = 0,
        active: int | None = None,
        skip_first: int = 0,
    ) -> str:
        if cls._profiler is not None:
            logger.warning("[Rank %s] Stopping existing profiler", cls._get_rank())
            cls._profiler.stop()
            cls._profiler = None

        rank = cls._get_rank()
        trace_path_template = os.path.abspath(trace_path_template)
        cls._trace_template = trace_path_template
        cls._backend = ""
        cls._trace_path = ""

        if rank != 0 and profile_rank0_only:
            return ""

        if _use_npu_profiler():
            return cls._start_npu(
                rank, trace_path_template, wait=wait, warmup=warmup, active=active, skip_first=skip_first
            )
        return cls._start_cuda(
            rank, trace_path_template, wait=wait, warmup=warmup, active=active, skip_first=skip_first
        )

    @classmethod
    def _start_npu(
        cls,
        rank: int,
        trace_path_template: str,
        wait: int = 0,
        warmup: int = 0,
        active: int | None = None,
        skip_first: int = 0,
    ) -> str:
        import torch_npu

        rank_dir = os.path.join(trace_path_template, f"rank{rank}")
        os.makedirs(rank_dir, exist_ok=True)
        cls._backend = "npu"
        cls._trace_path = rank_dir

        experimental_config = _npu_experimental_config(torch_npu)
        logger.info(
            "[Rank %s] Starting torch_npu profiler (CPU+NPU, Level1, PipeUtilization) -> %s",
            rank,
            rank_dir,
        )

        handler_kwargs = {"dir_name": rank_dir, "worker_name": f"rank{rank}"}
        try:
            on_trace_ready = torch_npu.profiler.tensorboard_trace_handler(**handler_kwargs)
        except TypeError:
            on_trace_ready = torch_npu.profiler.tensorboard_trace_handler(rank_dir)

        profile_kwargs = dict(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=False,
            experimental_config=experimental_config,
        )
        if active is not None:
            schedule = _make_profiler_schedule(
                torch_npu.profiler, wait=wait, warmup=warmup, active=active, skip_first=skip_first
            )
            if schedule is not None:
                profile_kwargs["schedule"] = schedule
                logger.info(
                    "[Rank %s] Profiler schedule wait=%d warmup=%d active=%d skip_first=%d",
                    rank,
                    wait,
                    warmup,
                    active,
                    skip_first,
                )
        cls._profiler = torch_npu.profiler.profile(**profile_kwargs)
        cls._profiler.start()
        return rank_dir

    @classmethod
    def _start_cuda(
        cls,
        rank: int,
        trace_path_template: str,
        wait: int = 0,
        warmup: int = 0,
        active: int | None = None,
        skip_first: int = 0,
    ) -> str:
        from torch.profiler import ProfilerActivity, profile

        json_file = f"{trace_path_template}_rank{rank}.json"
        os.makedirs(os.path.dirname(json_file) or ".", exist_ok=True)
        cls._backend = "cuda"
        cls._trace_path = f"{json_file}.gz"
        cuda_active = 100000 if active is None else active

        logger.info("[Rank %s] Starting End-to-End Torch profiler (CUDA)", rank)

        def trace_handler(p):
            nonlocal json_file
            try:
                p.export_chrome_trace(json_file)
                logger.info("[Rank %s] Trace exported to %s", rank, json_file)
                try:
                    subprocess.Popen(["gzip", "-f", json_file])
                    logger.info("[Rank %s] Triggered background compression for %s", rank, json_file)
                    json_file = f"{json_file}.gz"
                except Exception as compress_err:
                    logger.warning("[Rank %s] Background gzip failed to start: %s", rank, compress_err)
            except Exception as e:
                logger.warning("[Rank %s] Failed to export trace: %s", rank, e)

        schedule = _make_profiler_schedule(
            torch.profiler, wait=wait, warmup=warmup, active=cuda_active, skip_first=skip_first
        )
        cls._profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule,
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
        cls._profiler.start()
        return cls._trace_path

    @classmethod
    def stop(cls) -> dict | None:
        if cls._profiler is None:
            return None

        rank = cls._get_rank()
        trace_path = cls._trace_path
        backend = cls._backend

        try:
            if backend == "npu":
                try:
                    import torch_npu

                    torch_npu.npu.synchronize()
                except Exception:
                    pass
            cls._profiler.stop()
            logger.info("[Rank %s] Profiler stopped, trace: %s", rank, trace_path)
        except Exception as e:
            logger.warning("[Rank %s] Profiler stop failed: %s", rank, e)

        cls._profiler = None
        cls._backend = ""
        cls._trace_path = ""
        return {"trace": trace_path, "table": None}

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
