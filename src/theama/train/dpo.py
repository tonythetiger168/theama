# SPDX-License-Identifier: Apache-2.0
"""P1a: LoRA-DPO on Wan 2.2 TI2V-5B with VideoReward preference pairs.

Formulation: Diffusion-DPO adapted to flow matching (same family as
T2V-Turbo v2 / ViPO video DPO). For a preference pair (x_win, x_lose) sharing
a prompt, at a sampled timestep t:

    x_t    = (1 - t) * x + t * noise        (flow interpolation)
    v_true = noise - x                       (velocity target)

The policy's implicit reward advantage over the frozen reference model is
approximated by the difference of velocity-prediction errors:

    advantage = -(mse(v_theta, v_true) - mse(v_ref, v_true))

and the DPO loss is the logistic preference objective on
advantage_win - advantage_lose. The pure-torch pieces (flow_match_pair,
velocity_mse, dpo_loss) are CPU-testable; the trainer lazily loads
diffusers/peft on GPU.
"""

from __future__ import annotations

import random
from pathlib import Path


def flow_match_pair(x, noise, t):
    """Build flow-matching (x_t, v_target) for clean latent x.

    x, noise: tensors of equal shape. t: scalar in [0, 1] or per-sample
    tensor broadcastable to x. Returns (x_t, v_target).
    """
    x_t = (1.0 - t) * x + t * noise
    v_target = noise - x
    return x_t, v_target


def velocity_mse(pred, target):
    """Per-sample MSE of velocity predictions, shape (batch,)."""
    return (pred.float() - target.float()).pow(2).flatten(1).mean(dim=1)


def dpo_loss(model_err_w, ref_err_w, model_err_l, ref_err_l, beta: float):
    """Diffusion-DPO logistic loss from velocity-MSE errors.

    Inputs are scalar tensors (batch-mean squared errors). Lower when the
    policy improves on the win sample and degrades on the lose sample,
    relative to the frozen reference.
    """
    import torch

    advantage_w = -(model_err_w - ref_err_w)
    advantage_l = -(model_err_l - ref_err_l)
    logits = beta * (advantage_w - advantage_l)
    return -torch.nn.functional.logsigmoid(logits).mean()


class DPOWanTrainer:
    """GPU trainer: LoRA on the Wan TI2V-5B DiT against a frozen reference.

    Not instantiable on CPU — all heavy imports live inside __init__/train.
    Latent pairs are pre-encoded by scripts/encode_pairs.py so the VAE stays
    out of the training hot loop.
    """

    def __init__(self, cfg: dict):
        import torch
        from diffusers import WanPipeline
        from peft import LoraConfig, get_peft_model

        from ..inference.config import MODELS

        self.cfg = cfg
        spec = MODELS[cfg["base_model"]]
        dtype = torch.bfloat16

        pipe = WanPipeline.from_pretrained(spec.repo_id, torch_dtype=dtype)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to("cuda")
        self.ref = pipe.transformer.to("cuda")  # frozen reference
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.ref.eval()
        del pipe.vae  # latents are pre-encoded; keep VRAM for the policy

        self.policy = pipe.transformer.__class__.from_pretrained(
            spec.repo_id, subfolder="transformer", torch_dtype=dtype
        ).to("cuda")
        lcfg = cfg["lora"]
        self.policy = get_peft_model(
            self.policy,
            LoraConfig(
                r=lcfg["rank"],
                lora_alpha=lcfg["alpha"],
                lora_dropout=lcfg.get("dropout", 0.0),
                target_modules=list(lcfg["target_modules"]),
            ),
        )
        self.policy.enable_gradient_checkpointing()
        self.policy.train()

        tcfg = cfg["train"]
        self.opt = torch.optim.AdamW(
            (p for p in self.policy.parameters() if p.requires_grad),
            lr=tcfg["lr"],
        )

    def encode_prompt(self, prompts: list[str], max_length: int = 512):
        """umT5 prompt embeddings with padding zeroed (Wan pipeline recipe)."""
        import torch

        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        mask = text_inputs.attention_mask.to("cuda")
        with torch.no_grad():
            embeds = self.text_encoder(
                text_inputs.input_ids.to("cuda"), attention_mask=mask
            )[0]
        return embeds * mask.unsqueeze(-1).to(embeds.dtype)

    def _pair_loss(self, x_w, x_l, prompt_embeds, beta, gen):
        """One DPO loss on a (win, lose) latent pair, batched as 2."""
        import torch

        x = torch.cat([x_w, x_l], dim=0)
        noise = torch.randn(
            x.shape, generator=gen, device=x.device, dtype=x.dtype
        )
        t = torch.rand(2, generator=gen, device=x.device)
        x_t, v_true = flow_match_pair(x, noise, t.view(2, 1, 1, 1, 1))
        timestep = (t * 1000.0).to(x.dtype)
        ctx = prompt_embeds.expand(2, -1, -1)

        with torch.no_grad():
            v_ref = self.ref(
                hidden_states=x_t,
                timestep=timestep,
                encoder_hidden_states=ctx,
                return_dict=False,
            )[0]
        v_pred = self.policy(
            hidden_states=x_t,
            timestep=timestep,
            encoder_hidden_states=ctx,
            return_dict=False,
        )[0]

        err_model = velocity_mse(v_pred, v_true)
        err_ref = velocity_mse(v_ref, v_true)
        return dpo_loss(err_model[0], err_ref[0], err_model[1], err_ref[1], beta)

    def train(self, pairs: list[dict], out_dir: str) -> dict:
        """Run the DPO loop. pairs: [{prompt, win_latent, lose_latent}, ...]
        where *_latent are paths to .pt files from scripts/encode_pairs.py.
        Returns a metrics dict (also written to out_dir/dpo_metrics.json).
        """
        import json

        import torch

        tcfg = self.cfg["train"]
        beta = float(tcfg["beta"])
        accum = int(tcfg["grad_accum"])
        max_steps = int(tcfg["steps"])
        save_every = int(tcfg.get("save_every", 100))
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        seed = int(tcfg.get("seed", 42))
        gen = torch.Generator(device="cuda").manual_seed(seed)
        order = pairs[:]
        random.Random(seed).shuffle(order)

        self.opt.zero_grad(set_to_none=True)
        step, micro, loss_sum = 0, 0, 0.0
        done = False
        while not done:
            for pair in order:
                x_w = torch.load(pair["win_latent"], map_location="cuda")
                x_l = torch.load(pair["lose_latent"], map_location="cuda")
                embeds = self.encode_prompt([pair["prompt"]])
                loss = self._pair_loss(x_w, x_l, embeds, beta, gen) / accum
                loss.backward()
                loss_sum += float(loss.detach()) * accum
                micro += 1
                if micro % accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in self.policy.parameters() if p.requires_grad),
                        1.0,
                    )
                    self.opt.step()
                    self.opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % save_every == 0 or step == max_steps:
                        self.policy.save_pretrained(out / f"lora_step{step:05d}")
                    if step >= max_steps:
                        done = True
                        break
        metrics = {
            "steps": step,
            "pairs": len(pairs),
            "beta": beta,
            "mean_loss": loss_sum / max(micro, 1),
            "final_adapter": str(out / f"lora_step{step:05d}"),
        }
        (out / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
        return metrics
