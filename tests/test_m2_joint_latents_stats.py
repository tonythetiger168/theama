# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/joint_latents_stats.py (P1b cache profiler).

Covers the pure profile function (row classification, shape mix, caption
duplication, index coverage, join view) and the CLI contract (tolerant
jsonl parsing, unreadable manifest exit 1, deterministic stdout, --out).
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "joint_latents_stats", ROOT / "scripts" / "joint_latents_stats.py"
)
jls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jls)


def _av_row(index: int, caption: str = "a cat", model: str = "ti2v-5b",
            video_shape=(1, 16, 21, 60, 104), audio_shape=(1, 300, 64),
            reused: bool = False) -> dict:
    row = {
        "index": index,
        "caption": caption,
        "source": "local",
        "video": f"v/{index}.mp4",
        "audio": f"a/{index}.wav",
        "model": model,
        "video_latent": f"sample{index:04d}_video.pt",
        "audio_latent": f"sample{index:04d}_audio.pt",
        "video_shape": list(video_shape),
        "audio_shape": list(audio_shape),
    }
    if reused:
        row["reused"] = True
    return row


def _text_row(index: int, caption: str = "a cat",
              model: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
              seq_len: int = 12, reused: bool = False) -> dict:
    row = {
        "index": index,
        "caption": caption,
        "model": model,
        "text_latent": "text_deadbeef.pt",
        "shape": [512, 4096],
        "seq_len": seq_len,
    }
    if reused:
        row["reused"] = True
    return row


def _to_jsonl(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def test_profile_empty_manifests():
    profile = jls.profile_caches("", "")
    av = profile["av"]
    assert av["rows"] == 0
    assert av["success_rows"] == 0
    assert av["error_rows"] == 0
    assert av["skipped_rows"] == 0
    assert av["index"] == {"rows": 0, "unique": 0, "duplicate_rows": 0,
                           "min": None, "max": None, "gaps": 0}
    assert av["captions"]["length_chars"] is None
    assert av["video_shapes"] == {}
    assert profile["text"]["seq_len"] is None
    assert profile["join"] == {
        "av_indices": 0, "text_indices": 0, "served_indices": 0,
        "av_only_indices": 0, "text_only_indices": 0,
        "caption_mismatch_rows": 0,
    }


def test_profile_av_row_classification_and_reused():
    rows = [
        _av_row(0),
        _av_row(1, reused=True),
        {"index": 2, "caption": "x", "model": "ti2v-5b", "error": "oom"},
        {"index": 3, "caption": "y", "model": "ti2v-5b", "skipped": True,
         "retries": 3},
    ]
    profile = jls.profile_caches(_to_jsonl(rows), "")
    av = profile["av"]
    assert av["rows"] == 4
    assert av["success_rows"] == 2
    assert av["error_rows"] == 1
    assert av["skipped_rows"] == 1
    assert av["reused_rows"] == 1
    # Skipped/error rows do not leak into the shape or caption stats.
    assert av["video_shapes"] == {'[1, 16, 21, 60, 104]': 2}
    assert av["captions"]["rows"] == 2


def test_profile_av_index_coverage_gaps_and_duplicates():
    rows = [_av_row(0), _av_row(0), _av_row(2), _av_row(4)]
    profile = jls.profile_caches(_to_jsonl(rows), "")
    index = profile["av"]["index"]
    assert index["rows"] == 4
    assert index["unique"] == 3
    assert index["duplicate_rows"] == 1
    assert index["min"] == 0
    assert index["max"] == 4
    assert index["gaps"] == 2  # indices 1 and 3 missing


def test_profile_av_shape_mix_and_caption_sharing():
    rows = [
        _av_row(0, caption="shared"),
        _av_row(1, caption="shared"),
        _av_row(2, caption="long caption here",
                video_shape=(1, 16, 41, 60, 104)),
    ]
    profile = jls.profile_caches(_to_jsonl(rows), "")
    av = profile["av"]
    assert av["video_shapes"] == {
        '[1, 16, 21, 60, 104]': 2,
        '[1, 16, 41, 60, 104]': 1,
    }
    caps = av["captions"]
    assert caps["unique"] == 2
    assert caps["shared_by_multiple_rows"] == 1
    assert caps["max_rows_per_caption"] == 2
    assert caps["length_chars"]["max"] == float(len("long caption here"))


def test_profile_text_dedup_seq_len_and_models():
    rows = [
        _text_row(0, seq_len=10),
        _text_row(1, caption="a cat", seq_len=10, reused=True),  # dup caption
        _text_row(2, caption="dog", seq_len=20),
        {"index": 3, "caption": "z", "model": "m", "error": "cuda"},
    ]
    profile = jls.profile_caches("", _to_jsonl(rows))
    text = profile["text"]
    assert text["success_rows"] == 3
    assert text["error_rows"] == 1
    assert text["reused_rows"] == 1
    assert text["captions"]["unique"] == 2
    assert text["seq_len"] == {"count": 3, "min": 10.0, "median": 10.0,
                               "mean": round(40 / 3, 3), "max": 20.0}
    assert text["shapes"] == {'[512, 4096]': 3}
    assert text["models"] == {"Wan-AI/Wan2.2-T2V-A14B-Diffusers": 3}


def test_profile_join_served_only_mismatch():
    av_rows = [_av_row(0), _av_row(1), _av_row(2, caption="av caption")]
    text_rows = [_text_row(1), _text_row(2, caption="text caption"),
                 _text_row(3)]
    profile = jls.profile_caches(_to_jsonl(av_rows), _to_jsonl(text_rows))
    join = profile["join"]
    assert join["av_indices"] == 3
    assert join["text_indices"] == 3
    assert join["served_indices"] == 2  # indices 1 and 2
    assert join["av_only_indices"] == 1  # index 0
    assert join["text_only_indices"] == 1  # index 3
    assert join["caption_mismatch_rows"] == 1  # index 2 drifted


def test_profile_deterministic_key_order():
    av = _to_jsonl([_av_row(2), _av_row(0), _av_row(1, caption="b")])
    text = _to_jsonl([_text_row(1), _text_row(0)])
    first = json.dumps(jls.profile_caches(av, text), sort_keys=True)
    second = json.dumps(jls.profile_caches(av, text), sort_keys=True)
    assert first == second
    parsed = json.loads(first)
    assert list(parsed.keys()) == ["av", "join", "text"]


def _write_caches(tmp_path: Path, av_lines: list[str],
                  text_lines: list[str]) -> tuple[Path, Path]:
    av_dir = tmp_path / "latents"
    text_dir = tmp_path / "texts"
    av_dir.mkdir()
    text_dir.mkdir()
    (av_dir / "av_latents.jsonl").write_text(
        "\n".join(av_lines) + "\n", encoding="utf-8")
    (text_dir / "text_latents.jsonl").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8")
    return av_dir, text_dir


def _run_cli(av_dir: Path, text_dir: Path,
             *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "joint_latents_stats.py"),
         "--av-cache", str(av_dir), "--text-cache", str(text_dir), *extra],
        capture_output=True, text=True, timeout=60,
    )


def test_cli_tolerates_bad_json_lines(tmp_path):
    av_dir, text_dir = _write_caches(
        tmp_path,
        [json.dumps(_av_row(0)), "{not json", json.dumps(_av_row(1))],
        [json.dumps(_text_row(0)), "[1,2]"],
    )
    result = _run_cli(av_dir, text_dir)
    assert result.returncode == 0, result.stderr
    profile = json.loads(result.stdout)
    assert profile["av"]["bad_json_lines"] == 1
    assert profile["av"]["success_rows"] == 2
    assert profile["text"]["bad_json_lines"] == 1
    assert profile["join"]["served_indices"] == 1


def test_cli_missing_manifest_exits_1(tmp_path):
    av_dir, text_dir = _write_caches(tmp_path, [json.dumps(_av_row(0))],
                                     [json.dumps(_text_row(0))])
    (text_dir / "text_latents.jsonl").unlink()
    result = _run_cli(av_dir, text_dir)
    assert result.returncode == 1
    assert "cannot read manifest" in result.stderr


def test_cli_deterministic_stdout_and_out_file(tmp_path):
    av_dir, text_dir = _write_caches(
        tmp_path,
        [json.dumps(_av_row(0, caption="shared")),
         json.dumps(_av_row(1, caption="shared"))],
        [json.dumps(_text_row(0, caption="shared")),
         json.dumps(_text_row(1, caption="shared"))],
    )
    first = _run_cli(av_dir, text_dir)
    assert first.returncode == 0, first.stderr
    out = tmp_path / "stats.json"
    second = _run_cli(av_dir, text_dir, "--out", str(out))
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
    assert out.read_text(encoding="utf-8") == second.stdout
    profile = json.loads(second.stdout)
    assert profile["av"]["captions"]["shared_by_multiple_rows"] == 1
    assert profile["join"]["served_indices"] == 2
