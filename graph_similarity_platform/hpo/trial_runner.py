from __future__ import annotations

import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Mapping

from .adapters import ModelHPOAdapter
from .types import ProgressCallback, TrialContext, TrialResult


FLOAT_TOKEN = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
VALIDATION_MSE_PATTERNS = (
    re.compile(rf"validation_mse={FLOAT_TOKEN}"),
    re.compile(rf"validation similarity MSE:\s*{FLOAT_TOKEN}", re.IGNORECASE),
    re.compile(rf"best validation checkpoint \(MSE={FLOAT_TOKEN}\)", re.IGNORECASE),
)
SPEARMAN_PATTERN = re.compile(rf"validation_spearman={FLOAT_TOKEN}")
MAE_PATTERN = re.compile(rf"validation_(?:norm_ged_)?mae={FLOAT_TOKEN}")
STEP_PATTERNS = (
    re.compile(r"step=(\d+)"),
    re.compile(r"Epoch\s+(\d+)\s+validation", re.IGNORECASE),
)


class TrialRunner:
    def run(
        self,
        *,
        adapter: ModelHPOAdapter,
        context: TrialContext,
        config: Mapping[str, Any],
        trial: Any | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> TrialResult:
        context.trial_dir.mkdir(parents=True, exist_ok=True)
        command, cwd = adapter.command(context, config)
        output_path = context.trial_dir / "output.log"
        started = time.monotonic()
        intermediate: list[dict[str, float | int | None]] = []
        best_mse: float | None = None
        best_spearman: float | None = None
        best_mae: float | None = None
        best_step: int | None = None
        peak_memory_mb: float | None = None
        return_code: int | None = None
        exception: str | None = None

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        queue: Queue[str | None] = Queue()

        def read_output() -> None:
            try:
                for line in process.stdout:
                    queue.put(line)
            finally:
                queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        pruned = False
        with output_path.open("w") as output:
            stream_finished = False
            while not stream_finished:
                try:
                    line = queue.get(timeout=0.25)
                except Empty:
                    line = ""
                if line is None:
                    stream_finished = True
                    continue
                if line:
                    output.write(line)
                    output.flush()
                    print(line, end="", flush=True)
                    parsed = _parse_intermediate(line, len(intermediate) + 1)
                    if parsed is not None:
                        intermediate.append(parsed)
                        mse = float(parsed["validation_mse"])
                        if best_mse is None or mse < best_mse:
                            best_mse = mse
                            best_spearman = _finite(parsed.get("validation_spearman"))
                            best_mae = _finite(parsed.get("validation_mae"))
                            best_step = int(parsed["step"])
                        if trial is not None:
                            trial.report(mse, int(parsed["step"]))
                            if trial.should_prune():
                                pruned = True
                                process.terminate()
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "current_validation_mse": mse,
                                    "current_validation_spearman": parsed.get(
                                        "validation_spearman"
                                    ),
                                    "best_validation_mse": best_mse,
                                    "best_step": best_step,
                                    "current_step": int(parsed["step"]),
                                    "resource": int(context.resource),
                                }
                            )
                memory = _resident_memory_mb(process.pid)
                if memory is not None:
                    peak_memory_mb = max(peak_memory_mb or 0.0, memory)
                if pruned and process.poll() is not None:
                    stream_finished = True

        try:
            return_code = process.wait(timeout=10 if pruned else None)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait()
        duration = time.monotonic() - started

        if pruned:
            _clean_checkpoint(context.checkpoint)
            result = TrialResult(
                status="pruned",
                validation_mse=best_mse,
                validation_spearman=best_spearman,
                validation_mae=best_mae,
                validation_rmse=(math.sqrt(best_mse) if best_mse is not None else None),
                duration_seconds=duration,
                peak_memory_mb=peak_memory_mb,
                best_step=best_step,
                checkpoint=None,
                command=command,
                return_code=return_code,
                intermediate_values=intermediate,
            )
            self._write_result(context, config, result)
            return result

        if return_code != 0:
            exception = f"Training process exited with code {return_code}."
        if best_mse is None and return_code == 0:
            exception = "Training completed without a finite validation MSE."
        status = "completed" if exception is None else "failed"
        checkpoint = (
            str(context.checkpoint)
            if context.checkpoint.exists()
            or any(context.checkpoint.parent.glob(context.checkpoint.name + "*"))
            else None
        )
        if status == "failed":
            _clean_checkpoint(context.checkpoint)
            checkpoint = None
        result = TrialResult(
            status=status,
            validation_mse=best_mse,
            validation_spearman=best_spearman,
            validation_mae=best_mae,
            validation_rmse=(math.sqrt(best_mse) if best_mse is not None else None),
            duration_seconds=duration,
            peak_memory_mb=peak_memory_mb,
            best_step=best_step,
            checkpoint=checkpoint,
            command=command,
            return_code=return_code,
            exception=exception,
            intermediate_values=intermediate,
        )
        self._write_result(context, config, result)
        return result

    @staticmethod
    def _write_result(
        context: TrialContext,
        config: Mapping[str, Any],
        result: TrialResult,
    ) -> None:
        payload = {
            "dataset_id": context.dataset_id,
            "model_id": context.model_id,
            "seed": context.seed,
            "split_seed": context.split_seed,
            "dataset_fingerprint": context.profile.fingerprint,
            "hyperparameters": dict(config),
            **result.to_dict(),
        }
        path = context.trial_dir / "trial.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(path)


def _parse_intermediate(line: str, fallback_step: int) -> dict[str, float | int | None] | None:
    mse = None
    for pattern in VALIDATION_MSE_PATTERNS:
        match = pattern.search(line)
        if match:
            mse = _finite(match.group(1))
            break
    if mse is None:
        return None
    step = fallback_step
    for pattern in STEP_PATTERNS:
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            break
    spearman_match = SPEARMAN_PATTERN.search(line)
    mae_match = MAE_PATTERN.search(line)
    return {
        "step": step,
        "validation_mse": mse,
        "validation_spearman": (
            _finite(spearman_match.group(1)) if spearman_match else None
        ),
        "validation_mae": _finite(mae_match.group(1)) if mae_match else None,
    }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _resident_memory_mb(pid: int) -> float | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip()) / 1024.0
    except ValueError:
        return None


def _clean_checkpoint(checkpoint: Path) -> None:
    for path in checkpoint.parent.glob(checkpoint.name + "*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
