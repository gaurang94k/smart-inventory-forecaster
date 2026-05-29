"""Demo script: runs 3 inventory planning queries through the LLM agent.

Usage (from project root):
    uv run python -m scripts.demo_agent
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.agent import run_agent

QUERIES = [
    "What is the reorder point for FOODS_3_090_CA_1?",
    "What happens to the safety stock for FOODS_3_090_CA_1 if we increase service level to 99%?",
    "Which SKU has the best forecast accuracy?",
]


def main() -> None:
    for i, question in enumerate(QUERIES, 1):
        print(f"Q{i}: {question}")
        print(run_agent(question))
        print("-" * 70)


if __name__ == "__main__":
    main()
