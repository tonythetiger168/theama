#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P1a entry point: LoRA-DPO on Wan TI2V-5B from encoded preference pairs.

    python scripts/train_dpo.py --config configs/dpo_ti2v_5b.yaml \
        --pairs dpo_out/latents/pairs_latents.jsonl --out dpo_out/run1

The manifest is validated (data/pair_latent_check.py) before the backbone
is loaded: encode-failure / retry-budget-skipped rows and rows whose .pt
latents are gone are dropped with reasons on stderr, and the run refuses
to start when nothing usable remains — a bad manifest fails fast on CPU
instead of raising KeyError/FileNotFoundError inside the GPU loop. Use
scripts/check_pair_latents.py to gate the file ahead of time.
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

    from theama.data.pair_latent_check import validate_pair_latents

    try:
        text = Path(args.pairs).read_text(encoding="utf-8")
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    report = validate_pair_latents(text)
    if report["dropped_rows"] or report["bad_json_lines"]:
        print(
            f"pairs manifest: {report['usable']}/{report['rows']} rows usable; "
            f"dropped {json.dumps(report['dropped'], sort_keys=True)}"
            + (
                f", {report['bad_json_lines']} bad JSON line(s)"
                if report["bad_json_lines"]
                else ""
            ),
            file=sys.stderr,
        )
    pairs = report["usable_rows"]
    if not pairs:
        print("no usable pairs", file=sys.stderr)
        return 1

    from theama.train.dpo import DPOWanTrainer

    trainer = DPOWanTrainer(cfg)
    metrics = trainer.train(pairs, args.out)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
