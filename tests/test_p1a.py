# SPDX-License-Identifier: Apache-2.0
"""CPU tests for P1a: reward registry, pair building, DPO math."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.reward.config import ALLOWED_LICENSES, REWARD_MODELS, get_reward_spec
from theama.reward.pairs import build_pairs, coverage_report
from theama.train.dpo import dpo_loss, flow_match_pair, velocity_mse


# ---- reward registry -------------------------------------------------------

def test_reward_registry_license_whitelist():
    assert REWARD_MODELS, "registry must not be empty"
    for name, spec in REWARD_MODELS.items():
        assert spec.license in ALLOWED_LICENSES, f"{name}: {spec.license}"
        assert get_reward_spec(name) is spec


def test_reward_registry_unknown_rejected():
    with pytest.raises(KeyError):
        get_reward_spec("minimax-h3-reward")


def test_videoreward_entry():
    spec = get_reward_spec("videoreward")
    assert spec.repo_id == "KlingTeam/VideoReward"
    assert spec.license == "Apache-2.0"
    assert spec.multi_dim


# ---- preference pair building ----------------------------------------------

def _rows(scores_by_prompt):
    rows = []
    for pid, entries in scores_by_prompt.items():
        for i, s in enumerate(entries):
            rows.append({
                "prompt_id": pid,
                "prompt": f"prompt {pid}",
                "video": f"/v/{pid}/cand_{i}.mp4",
                "overall": s,
            })
    return rows


def test_build_pairs_best_vs_worst():
    rows = _rows({"T2V-001": [0.9, 0.5, 0.7, 0.2]})
    pairs = build_pairs(rows, margin=0.5)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.win_video.endswith("cand_0.mp4")   # 0.9
    assert p.lose_video.endswith("cand_3.mp4")  # 0.2
    assert p.margin == pytest.approx(0.7)


def test_build_pairs_margin_filter():
    rows = _rows({"T2V-001": [0.55, 0.50, 0.52]})
    assert build_pairs(rows, margin=0.5) == []


def test_build_pairs_single_candidate_skipped():
    rows = _rows({"T2V-001": [0.9], "T2V-002": [0.1, 0.9]})
    pairs = build_pairs(rows, margin=0.5)
    assert [p.prompt_id for p in pairs] == ["T2V-002"]


def test_build_pairs_deterministic_tie_break():
    rows = _rows({"T2V-001": [0.9, 0.9, 0.1]})
    p1 = build_pairs(rows, margin=0.5)
    p2 = build_pairs(rows, margin=0.5)
    assert p1 == p2
    assert p1[0].win_video.endswith("cand_0.mp4")  # path-sorted tie-break


def test_coverage_report():
    rows = _rows({"A": [0.9, 0.1], "B": [0.5, 0.4], "C": [0.8, 0.2]})
    pairs = build_pairs(rows, margin=0.5)
    rep = coverage_report(rows, pairs)
    assert rep["prompts_total"] == 3
    assert rep["prompts_paired"] == 2
    assert rep["pair_rate"] == pytest.approx(2 / 3)


# ---- DPO math (pure torch, CPU) --------------------------------------------

def test_flow_match_endpoints():
    x = torch.randn(2, 4, 3, 8, 8)
    noise = torch.randn_like(x)
    x_t0, v0 = flow_match_pair(x, noise, 0.0)
    x_t1, _ = flow_match_pair(x, noise, 1.0)
    assert torch.allclose(x_t0, x)
    assert torch.allclose(x_t1, noise)
    assert torch.allclose(v0, noise - x)


def test_dpo_loss_prefers_policy_beating_ref_on_wins():
    zero = torch.tensor(0.0)
    one = torch.tensor(1.0)
    # beta kept small so logits stay unsaturated and the loss is measurable
    good = dpo_loss(zero, one, one, zero, beta=2)   # policy wins the pair
    bad = dpo_loss(one, zero, zero, one, beta=2)    # policy loses the pair
    assert good < bad
    assert float(good) > 0.0  # logsigmoid loss is always positive


def test_dpo_loss_zero_when_policy_equals_ref():
    e = torch.tensor(0.3)
    loss = dpo_loss(e, e.clone(), e.clone(), e, beta=500)
    assert float(loss) == pytest.approx(0.693147, rel=1e-4)  # log 2


def test_dpo_loss_gradient_direction():
    model_err_w = torch.tensor(0.5, requires_grad=True)
    model_err_l = torch.tensor(0.5, requires_grad=True)
    ref = torch.tensor(0.5)
    loss = dpo_loss(model_err_w, ref, model_err_l, ref, beta=500)
    loss.backward()
    # gradient DESCENT moves errors opposite to the grad sign:
    assert model_err_w.grad > 0  # descent shrinks policy error on the win
    assert model_err_l.grad < 0  # descent grows policy error on the lose


def test_velocity_mse_shape():
    pred = torch.randn(3, 2, 5)
    target = torch.randn(3, 2, 5)
    err = velocity_mse(pred, target)
    assert err.shape == (3,)


# ---- config + scripts -------------------------------------------------------

def test_dpo_yaml():
    cfg = yaml.safe_load((ROOT / "configs" / "dpo_ti2v_5b.yaml").read_text())["dpo"]
    assert cfg["base_model"] == "ti2v-5b"
    assert cfg["reward_model"] in REWARD_MODELS
    assert cfg["train"]["beta"] > 0
    assert cfg["lora"]["rank"] >= 8
    gates = cfg["gates"]
    assert 0.5 < gates["blind_win_rate_vs_base"] <= 1.0
    assert gates["vbench_tolerance"] > 0


@pytest.mark.parametrize("script", [
    "gen_candidates.py", "score_eval_set.py", "make_pref_pairs.py",
    "encode_pairs.py", "train_dpo.py",
])
def test_p1a_scripts_compile(script):
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(ROOT / "scripts" / script)],
        check=True,
    )
