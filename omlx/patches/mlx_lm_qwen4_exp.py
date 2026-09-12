# SPDX-License-Identifier: Apache-2.0
"""mlx-lm qwen4_exp fixes that must apply outside the mlx-vlm compat path."""

from __future__ import annotations

import bisect
import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_mlx_lm_qwen4_exp_patch() -> bool:
    """Inject ``bisect_right`` into ``mlx_lm.models.qwen4_exp``.

    The flat mlx-lm model resolves its PLE ngram shard with a bare ``bisect_right``
    call but only imports ``bisect``, so the name is undefined at runtime and every
    prefill that reaches the ngram embedding raises ``NameError``. The vendored
    mlx-vlm copy imports the name directly, but a cluster rank (or any text-only
    load) serves the flat mlx-lm model and never passes through the VLM compat
    path, so bind the name here on the shared mlx-lm load path instead.
    """
    global _APPLIED
    if _APPLIED:
        return False
    try:
        import mlx_lm.models.qwen4_exp as qwen4_exp  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    if not hasattr(qwen4_exp, "bisect_right"):
        qwen4_exp.bisect_right = bisect.bisect_right
        logger.debug("Injected mlx-lm qwen4_exp.bisect_right (upstream missing import)")
    _APPLIED = True
    return True