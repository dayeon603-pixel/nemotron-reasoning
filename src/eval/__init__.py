"""Offline evaluation harness for the NVIDIA Nemotron Model Reasoning Challenge.

Verbatim replication of the official competition scorer so that local CV
results are comparable byte-for-byte with the public/private leaderboard.
"""

__all__ = [
    "extract_final_answer",
    "verify",
    "CVResult",
    "run_cv",
    "load_split",
]

from src.eval.metric import extract_final_answer, verify
from src.eval.cv import CVResult, run_cv
from src.eval.holdout import load_split
