# SPDX-License-Identifier: Apache-2.0
"""Official Descript DAC 44.1kHz adapter (MIT code and weights).

Loading is lazy: importing this module does not import the third-party ``dac``
package or download weights. The upstream model returns ``(B,D,T_lat)``;
Theama's codec contract uses ``(B,T_lat,D)``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from theama.model.audio_codec import AudioCodecSpec, DAC_44K

__all__ = ["DACAudioCodec"]


class DACAudioCodec:
    """Frozen official DAC codec with Theama's continuous-latent layout."""

    def __init__(self, model, spec: AudioCodecSpec = DAC_44K):
        self.model = model
        self.spec = spec
        self.model.requires_grad_(False)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> "DACAudioCodec":
        """Load official MIT weights, downloading user-side only when no path is given."""
        import dac  # noqa: PLC0415

        path = model_path if model_path is not None else dac.utils.download(model_type="44khz")
        model = dac.DAC.load(str(path))
        if device is not None:
            model = model.to(device)
        return cls(model)

    def _validate_waveform(self, waveform: torch.Tensor) -> None:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(
                f"expected mono waveform (B, 1, T), got {tuple(waveform.shape)}"
            )
        if waveform.shape[-1] % self.spec.hop_length != 0:
            raise ValueError(
                f"waveform length {waveform.shape[-1]} is not a multiple of "
                f"hop {self.spec.hop_length}; pad to AudioCodecSpec.frames_for_seconds first"
            )

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        self._validate_waveform(waveform)
        preprocessed = self.model.preprocess(waveform, self.spec.sample_rate)
        encoded = self.model.encode(preprocessed)
        continuous = encoded[0]
        if continuous.ndim != 3 or continuous.shape[1] != self.spec.latent_dim:
            raise RuntimeError(
                f"unexpected DAC latent shape {tuple(continuous.shape)}; "
                f"expected (B,{self.spec.latent_dim},T_lat)"
            )
        return continuous.transpose(1, 2).contiguous()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3 or latents.shape[-1] != self.spec.latent_dim:
            raise ValueError(
                f"expected latents (B, T_lat, {self.spec.latent_dim}), "
                f"got {tuple(latents.shape)}"
            )
        continuous = latents.transpose(1, 2).contiguous()
        return self.model.decode(continuous)
