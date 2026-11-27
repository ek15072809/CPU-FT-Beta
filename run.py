#!/usr/bin/env python3
"""CPU-FineTuning entry point.

    python run.py            -> start the Web UI (recommended)
    python run.py --lang en  -> English UI
    python run.py train --config config.json
"""
from cpu_finetuning.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
