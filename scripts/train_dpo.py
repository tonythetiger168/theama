#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P1a entry point: LoRA-DPO on Wan TI2V-5B from encoded preference pairs.

    python scripts/train_dpo.py --config configs/dpo_ti2v_5b.yaml \
        --pairs dpo_out/latents/pairs_latents.jsonl --out dpo_out/run1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "dpo_ti2v_5b.yaml"))
    ap.add_argument("--pairs", required=True, help="pairs_latents.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())["dpo"]
    pairs = [
        json.loads(line)
        for line in Path(args.pairs).read_text().splitlines()
        if line.strip()
    ]
    if not pairs:
        print("empty pairs file", file=sys.stderr)
        return 1

    from theama.train.dpo import DPOWanTrainer

    trainer = DPOWanTrainer(cfg)
    metrics = trainer.train(pairs, args.out)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
