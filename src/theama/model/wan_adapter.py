# SPDX-License-Identifier: Apache-2.0
"""Wan 2.2 video-latent adapter for the M2 joint A/V trainer.

The default path is the *external latent-grid bridge*: audio cross-attention
updates flattened VAE latent tokens, the tokens are restored to 5-D, and the
frozen Wan transformer predicts velocity. The joint path can instead insert the
same trainable bridges after evenly spaced internal Wan transformer blocks.

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
    "select_wan_bridge_blocks",
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


def select_wan_bridge_blocks(num_blocks: int, num_bridges: int) -> list[int]:
    """Choose distinct, evenly spaced Wan blocks, including the final block.

    Bridges run *after* the returned zero-based block indices. Requiring at
    least as many Wan blocks as bridges prevents two audio segments from
    attaching to the same video block and making execution order ambiguous.
    """
    if num_blocks < 1 or num_bridges < 1:
        raise ValueError("Wan internal bridging requires at least one block and one bridge")
    if num_bridges > num_blocks:
        raise ValueError(
            f"cannot place {num_bridges} bridges on only {num_blocks} Wan blocks"
        )
    indices = [((index + 1) * num_blocks) // num_bridges - 1 for index in range(num_bridges)]
    if len(set(indices)) != num_bridges:
        raise RuntimeError(f"internal bridge schedule is not unique: {indices}")
    return indices


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
        model_path: str | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "WanVideoAdapter":
        """Load only the Wan transformer; VAE/text encoding stay outside this adapter.

        ``model_path`` may point at a user-downloaded Diffusers repository;
        otherwise the registered Hugging Face repo id is resolved user-side.
        """
        from diffusers import WanTransformer3DModel  # noqa: PLC0415

        spec = get_wan_adapter_spec(name)
        transformer = WanTransformer3DModel.from_pretrained(
            model_path or spec.repo_id,
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

    @property
    def inner_dim(self) -> int:
        """Wan patch-token width, inferred without importing Diffusers."""
        patch_embedding = getattr(self.transformer, "patch_embedding", None)
        inner_dim = getattr(patch_embedding, "out_channels", None)
        if not isinstance(inner_dim, int) or inner_dim < 1:
            raise ValueError("Wan transformer must expose patch_embedding.out_channels")
        return inner_dim

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None,
    ) -> None:
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

    def _run_transformer(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None,
    ) -> torch.Tensor:
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

    def predict_velocity(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the real Wan forward signature and return its velocity tensor."""
        self._validate_inputs(hidden_states, t, encoder_hidden_states, encoder_hidden_states_image)
        return self._run_transformer(
            hidden_states, t, encoder_hidden_states, encoder_hidden_states_image
        )

    def predict_joint_velocity(
        self,
        hidden_states: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        audio_dit: torch.nn.Module,
        audio_latent: torch.Tensor,
        audio_cond: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interleave audio segments and bidirectional bridges inside Wan blocks.

        Temporary forward hooks keep the stock Diffusers forward (including
        patchification, conditioning, RoPE and unpatchification) intact. Each
        hook advances the audio stream to its next bridge, then
        applies that bridge to Wan's ``(B,T,inner_dim)`` tokens. Hooks are
        always removed, including on failure. A joint call is deliberately
        serialized on this adapter; concurrent calls on one instance are not
        supported.
        """
        self._validate_inputs(hidden_states, t, encoder_hidden_states, encoder_hidden_states_image)
        batch = hidden_states.shape[0]
        if audio_latent.ndim != 3 or audio_latent.shape[0] != batch:
            raise ValueError("audio_latent must have shape (B,Ta,Da) matching Wan batch")
        if audio_cond.ndim != 2 or audio_cond.shape[0] != batch:
            raise ValueError("audio_cond must have shape (B,Da) matching Wan batch")
        blocks = getattr(self.transformer, "blocks", None)
        bridges = getattr(audio_dit, "bridges", None)
        if blocks is None or bridges is None or not len(bridges):
            raise ValueError("Wan internal bridging requires transformer blocks and audio bridges")
        if torch.is_grad_enabled() and getattr(self.transformer, "gradient_checkpointing", False):
            raise NotImplementedError(
                "wan_blocks bridging does not yet support Wan gradient checkpointing; "
                "disable it for this smoke path so audio segments advance exactly once"
            )
        indices = select_wan_bridge_blocks(len(blocks), len(bridges))
        bridge_dim = bridges[0].v_norm.normalized_shape[0]
        if bridge_dim != self.inner_dim:
            raise ValueError(
                f"audio bridge video_dim {bridge_dim} != Wan inner dim {self.inner_dim}; "
                "build AudioDiT with video_dim=video.inner_dim for wan_blocks"
            )
        if not all(getattr(bridge, "norm_video_kv", False) for bridge in bridges):
            raise ValueError(
                "wan_blocks bridging feeds un-normalised Wan hidden states into the "
                "audio bridge v2a keys/values (issue #4 smoke: audio loss / grad norm "
                "~40x the latent-grid path); build AudioDiT with norm_video_kv=True"
            )

        audio = audio_dit.start_stream(audio_latent)
        handles = []

        def make_hook(bridge_index: int):
            def bridge_hook(_module, _args, video_tokens):
                nonlocal audio
                if not isinstance(video_tokens, torch.Tensor) or video_tokens.ndim != 3:
                    raise RuntimeError("Wan block output must be a (B,T,inner_dim) tensor")
                audio = audio_dit.advance_to_bridge(audio, audio_cond, bridge_index)
                video_tokens, audio = bridges[bridge_index](video_tokens, audio)
                return video_tokens

            return bridge_hook

        try:
            for bridge_index, block_index in enumerate(indices):
                handles.append(blocks[block_index].register_forward_hook(make_hook(bridge_index)))
            sample = self._run_transformer(
                hidden_states, t, encoder_hidden_states, encoder_hidden_states_image
            )
        finally:
            for handle in handles:
                handle.remove()
        audio_pred = audio_dit.finish_stream(audio, audio_cond)
        return sample, audio_pred
