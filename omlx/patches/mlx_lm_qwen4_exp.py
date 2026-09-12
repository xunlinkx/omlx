# SPDX-License-Identifier: Apache-2.0
"""mlx-lm qwen4_exp fixes that must apply outside the mlx-vlm compat path."""

from __future__ import annotations

import bisect
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_mlx_lm_qwen4_exp_patch(
    model_name: str | Path | None = None,
    model_settings: Any | None = None,
) -> bool:
    """Inject ``bisect_right`` and configure PLE runtime for ``mlx_lm.models.qwen4_exp``.

    The flat mlx-lm model resolves its PLE ngram shard with a bare ``bisect_right``
    call but only imports ``bisect``, so the name is undefined at runtime and every
    prefill that reaches the ngram embedding raises ``NameError``. The vendored
    mlx-vlm copy imports the name directly, but a cluster rank (or any text-only
    load) serves the flat mlx-lm model and never passes through the VLM compat
    path, so bind the name here on the shared mlx-lm load path instead.

    Additionally, export ``OMLX_QWEN4_PLE_PATH`` and ``OMLX_QWEN4_PLE_MODE`` so
    that mlx-lm model construction binds ``DiskBackedShardedEmbedding`` rather
    than falling back to resident dummy tables.
    """
    global _APPLIED
    try:
        import mlx_lm.models.qwen4_exp as qwen4_exp  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    if not hasattr(qwen4_exp, "bisect_right"):
        qwen4_exp.bisect_right = bisect.bisect_right
        logger.debug("Injected mlx-lm qwen4_exp.bisect_right (upstream missing import)")

    if model_name:
        resolved_path = str(Path(model_name).expanduser().resolve())
        os.environ["OMLX_QWEN4_PLE_PATH"] = resolved_path
        mode = "mmap"
        if model_settings is not None and not getattr(
            model_settings, "qwen4_ple_ssd_offload", True
        ):
            mode = "resident"
        os.environ["OMLX_QWEN4_PLE_MODE"] = mode
        logger.debug(
            "Configured mlx-lm qwen4_exp PLE runtime: path=%s mode=%s",
            resolved_path,
            mode,
        )

    try:
        from .mlx_vlm_qwen4_exp_compat import _patch_prompt_loop

        _patch_prompt_loop()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to patch prompt loop for PLE lookahead: %s", exc)

    _APPLIED = True
    return True
