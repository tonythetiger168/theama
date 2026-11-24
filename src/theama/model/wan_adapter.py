# SPDX-License-Identifier: Apache-2.0
"""Wan 2.2 video-latent adapter for the M2 joint A/V trainer.

This is the *external latent-grid bridge* smoke path: audio cross-attention
updates flattened VAE latent tokens, the tokens are restored to 5-D, and the
frozen Wan transformer predicts velocity. It does not claim to implement the
planned bridges inside Wan transformer blocks; that is a later integration.

Only diffusers-layout Wan checkpoints are used. Loading is lazy so CPU tests
never import diffusers or download weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from theama.inference.config import I2V_A14B, T2V_A14B, TI2V_5B

__all__ = [
    "WanAdapterSpec",
    "WAN_ADAPTER_SPECS",
    "WanVideoAdapter",
    "flatten_video_latents",
    "get_wan_adapter_spec",
    "scale_wan_timestep",
    "unflatten_video_tokens",
]


@dataclass(frozen=True)
class WanAdapterSpec:
    """Verified Wan transformer input contract."""

    name: str
    repo_id: str
    latent_channels: int
    text_dim: int = 4096
    timestep_scale: int = 1000
    image_conditioned: bool = False
    subfolder: str = "transformer"


WAN_ADAPTER_SPECS: dict[str, WanAdapterSpec] = {
    spec.name: spec
    for spec in (
        WanAdapterSpec("t2v-a14b", T2V_A14B, latent_channels=16),
        WanAdapterSpec("i2v-a14b", I2V_A14B, latent_channels=16, image_conditioned=True),
        WanAdapterSpec("ti2v-5b", TI2V_5B, latent_channels=48),
    )
}


def get_wan_adapter_spec(name: str) -> WanAdapterSpec:
    try:
        return WAN_ADAPTER_SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown Wan adapter {name!r}; registered: {sorted(WAN_ADAPTER_SPECS)}"
        ) from None


def flatten_video_latents(hidden_states: torch.Tensor) -> torch.Tensor:
    """Convert Wan ``(B,C,F,H,W)`` latents to ``(B,F*H*W,C)`` bridge tokens."""
    if hidden_states.ndim != 5:
        raise ValueError(f"expected Wan latents (B,C,F,H,W), got {tuple(hidden_states.shape)}")
    batch, channels, frames, height, width = hidden_states.shape
    return hidden_states.permute(0, 2, 3, 4, 1).reshape(
        batch, frames * height * width, channels
    )


def unflatten_video_tokens(
    tokens: torch.Tensor,
    latent_shape: tuple[int, int, int, int, int],
) -> torch.Tensor:
    """Restore bridge tokens to the original Wan latent layout without detaching."""
    if len(latent_shape) != 5:
        raise ValueError(f"latent_shape must have five dims, got {latent_shape}")
    batch, channels, frames, height, width = latent_shape
    if tokens.ndim != 3 or tokens.shape != (batch, frames * height * width, channels):
        raise ValueError(
            f"tokens {tuple(tokens.shape)} do not match latent shape {latent_shape}"
        )
    return tokens.reshape(batch, frames, height, width, channels).permute(0, 4, 1, 2, 3)


def scale_wan_timestep(t: torch.Tensor, scale: int = 1000) -> torch.Tensor:
    """Convert continuous flow ``t`` to Wan's integer denoising timestep."""
    if t.ndim != 1:
        raise ValueError(f"t must have shape (batch,), got {tuple(t.shape)}")
    return torch.round(t.detach().float() * scale).clamp(0, scale).to(dtype=torch.long)


class WanVideoAdapter(torch.nn.Module):
    """Frozen ``WanTransformer3DModel`` implementing the trainer adapter contract."""

    def __init__(self, transformer: torch.nn.Module, spec: WanAdapterSpec):
        super().__init__()
        self.transformer = transformer
        self.spec = spec
        self.transformer.requires_grad_(False)
        self.transformer.eval()

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "WanVideoAdapter":
        """Load only the Wan transformer; VAE/text encoding stay outside this adapter."""
        from diffusers import WanTransformer3DModel  # noqa: PLC0415

        spec = get_wan_adapter_spec(name)
        transformer = WanTransformer3DModel.from_pretrained(
            spec.repo_id,
            subfolder=spec.subfolder,
            torch_dtype=dtype,
        )
        if device is not None:
            transformer = transformer.to(device)
        return cls(transformer, spec)

    def train(self, mode: bool = True) -> "WanVideoAdapter":
        super().train(mode)
        self.transformer.eval()
        return self

    def predict_velocity(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the real Wan forward signature and return its velocity tensor."""
        if hidden_states.ndim != 5:
            raise ValueError(
                f"expected Wan latents (B,C,F,H,W), got {tuple(hidden_states.shape)}"
            )
        batch, channels = hidden_states.shape[:2]
        if channels != self.spec.latent_channels:
            raise ValueError(
                f"{self.spec.name} expects {self.spec.latent_channels} latent channels, "
                f"got {channels}"
            )
        if (
            encoder_hidden_states.ndim != 3
            or encoder_hidden_states.shape[0] != batch
            or encoder_hidden_states.shape[-1] != self.spec.text_dim
        ):
            raise ValueError(
                f"expected text embeddings (B,L,{self.spec.text_dim}), "
                f"got {tuple(encoder_hidden_states.shape)}"
            )
        if t.ndim != 1 or t.shape[0] != batch:
            raise ValueError(f"t must have shape ({batch},), got {tuple(t.shape)}")
        if self.spec.image_conditioned and encoder_hidden_states_image is None:
            raise ValueError(f"{self.spec.name} requires image conditioning")

        timestep = scale_wan_timestep(t, self.spec.timestep_scale).to(hidden_states.device)
        output = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=False,
        )
        sample = output[0] if isinstance(output, tuple) else output.sample
        if sample.shape != hidden_states.shape:
            raise ValueError(
                f"Wan output {tuple(sample.shape)} != input {tuple(hidden_states.shape)}"
            )
        return sample
