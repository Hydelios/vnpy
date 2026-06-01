from __future__ import annotations

"""Compatibility wrapper for the renamed AI cross-sectional factor pool.

Use ai_cs_pool.AiCsPoolFactors for new code.
"""

from .ai_cs_pool import AiCsPoolFactors


Ai876Factors = AiCsPoolFactors

__all__ = ["AiCsPoolFactors", "Ai876Factors"]
