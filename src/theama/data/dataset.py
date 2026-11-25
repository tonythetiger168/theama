# SPDX-License-Identifier: Apache-2.0
"""Audio-video dataset with data-card manifest ingestion (M2 data pipeline).

Hard rule (license-risks.md): no MiniMax H3 / LTX weights, code, or generated
outputs may enter the training set. Every sample must carry a data-card
source record; ``check_source`` enforces the red line at manifest load time
and again at ``__getitem__`` (defence in depth).

Manifest format: one JSON object per line (jsonl) with keys
``video``, ``audio``, ``caption``, ``source``. Relative paths resolve
against the manifest's parent directory. Unknown keys are ignored so the
format can grow (duration, fps, license url, ...) without breaking readers.

Tolerant mode (default) drops bad rows into ``report.errors`` and keeps
going — one corrupt line must not kill a multi-hour training launch. Strict
mode raises on the first bad row for CI/validation use. Forbidden sources
are rejected in *both* modes and never reach the sample list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import Dataset

from theama.data.decode import decode_audio, decode_video

FORBIDDEN_SOURCES = ("minimax", "hailuo", "h3-", "ltx", "lightricks")

REQUIRED_KEYS = ("video", "audio", "caption", "source")

_RAW_SNIPPET = 120


@dataclass
class AVSample:
    video_path: Path
    audio_path: Path
    caption: str
    source: str  # data-card source id, e.g. "pexels:12345"


@dataclass(frozen=True)
class ManifestRowError:
    """One rejected manifest row. ``raw`` is a truncated copy for debugging."""

    lineno: int  # 1-based line number in the manifest file
    kind: str  # json | schema | forbidden_source
    message: str
    raw: str = ""

    def as_dict(self) -> dict:
        return {
            "lineno": self.lineno,
            "kind": self.kind,
            "message": self.message,
            "raw": self.raw,
        }


@dataclass
class ManifestReport:
    """Result of a tolerant manifest load."""

    samples: list[AVSample] = field(default_factory=list)
    errors: list[ManifestRowError] = field(default_factory=list)
    duplicates: int = 0

    def summary(self) -> dict:
        sources: dict[str, int] = {}
        for s in self.samples:
            sources[s.source] = sources.get(s.source, 0) + 1
        return {
            "samples": len(self.samples),
            "errors": len(self.errors),
            "duplicates": self.duplicates,
            "sources": dict(sorted(sources.items())),
            "forbidden_rejected": sum(
                1 for e in self.errors if e.kind == "forbidden_source"
            ),
        }


def check_source(source: str) -> None:
    low = source.lower()
    if any(bad in low for bad in FORBIDDEN_SOURCES):
        raise ValueError(f"forbidden training source (license red line): {source}")


def _snippet(raw: str) -> str:
    raw = raw.strip()
    return raw if len(raw) <= _RAW_SNIPPET else raw[: _RAW_SNIPPET - 1] + "…"


def parse_manifest_row(
    obj: object, *, lineno: int, base_dir: Path, raw: str = ""
) -> AVSample:
    """Validate one parsed JSON row and build an :class:`AVSample`.

    Raises ``ValueError`` on schema violations and on license red-line
    sources; callers decide whether that is fatal (strict) or collected
    (tolerant). Relative media paths resolve against ``base_dir``.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"row must be a JSON object, got {type(obj).__name__}")
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        raise ValueError(f"missing required keys: {missing}")
    values: dict[str, str] = {}
    for key in REQUIRED_KEYS:
        value = obj[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"key {key!r} must be a non-empty string")
        values[key] = value.strip()
    check_source(values["source"])

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (base_dir / path)

    return AVSample(
        video_path=_resolve(values["video"]),
        audio_path=_resolve(values["audio"]),
        caption=values["caption"],
        source=values["source"],
    )


def load_manifest(manifest_path: str | Path, *, strict: bool = False) -> ManifestReport:
    """Load a jsonl data-card manifest into validated samples.

    Tolerant mode (default): unparseable/invalid/forbidden rows are dropped
    into ``report.errors`` and exact ``(video, audio)`` duplicates are
    de-duplicated (first occurrence wins, counted in ``report.duplicates``).
    Strict mode raises on the first bad row. A missing manifest file always
    raises ``FileNotFoundError`` — silently training on nothing is worse.
    """
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    report = ManifestReport()
    seen: set[tuple[Path, Path]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                err = ManifestRowError(lineno, "json", str(exc), _snippet(line))
                if strict:
                    raise ValueError(
                        f"manifest line {lineno}: invalid JSON: {exc}"
                    ) from exc
                report.errors.append(err)
                continue
            try:
                sample = parse_manifest_row(
                    obj, lineno=lineno, base_dir=path.parent, raw=_snippet(line)
                )
            except ValueError as exc:
                kind = (
                    "forbidden_source"
                    if "forbidden training source" in str(exc)
                    else "schema"
                )
                err = ManifestRowError(lineno, kind, str(exc), _snippet(line))
                if strict:
                    raise ValueError(f"manifest line {lineno}: {exc}") from exc
                report.errors.append(err)
                continue
            key = (sample.video_path, sample.audio_path)
            if key in seen:
                report.duplicates += 1
                continue
            seen.add(key)
            report.samples.append(sample)
    return report


class AVDataset(Dataset):
    """Reads (video, audio, caption) triples gated by a data-card manifest.

    The manifest is fully loaded and validated at construction: forbidden
    sources can never slip in via a lazily-evaluated row. Rejected rows are
    kept on ``self.errors`` for inspection (empty in strict mode, which
    raises instead). ``__getitem__`` still re-checks the source as defence
    in depth before any decode happens.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        seconds: int = 5,
        fps: int = 24,
        sample_rate: int = 44100,
        *,
        height: int = 240,
        width: int = 432,
        strict: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.seconds, self.fps, self.sample_rate = seconds, fps, sample_rate
        self.height, self.width = height, width
        report = load_manifest(self.manifest_path, strict=strict)
        self.samples: list[AVSample] = report.samples
        self.errors: list[ManifestRowError] = report.errors
        self.duplicates: int = report.duplicates

    @staticmethod
    def check_source(source: str) -> None:
        check_source(source)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        """Decode one triple into latents-ready tensors.

        Returns ``video`` float32 ``(3, T, H, W)`` in ``[-1, 1]`` (VAE-ready),
        ``audio`` float32 ``(2, N)`` stereo in ``[-1, 1]``, plus ``caption``
        and ``source`` metadata. Decode failures raise
        :class:`~theama.data.decode.MediaDecodeError` — a corrupt clip must
        crash the worker loudly rather than slip zeros into a batch; wrap the
        loader if a training run should skip instead. Text encoding stays
        outside the dataset (cached upstream, like the VAE latents).
        """
        sample = self.samples[idx]
        check_source(sample.source)  # defence in depth; manifest load already gated
        frames = decode_video(
            sample.video_path,
            fps=self.fps,
            seconds=self.seconds,
            height=self.height,
            width=self.width,
        )
        wave = decode_audio(
            sample.audio_path,
            sample_rate=self.sample_rate,
            seconds=self.seconds,
        )
        video = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 127.5 - 1.0
        audio = torch.from_numpy(wave)
        return {
            "video": video,
            "audio": audio,
            "caption": sample.caption,
            "source": sample.source,
        }
