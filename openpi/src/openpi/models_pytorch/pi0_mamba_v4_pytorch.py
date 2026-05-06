"""PI0 with Mamba SSM as a *pre-input* branch on the PaliGemma prefix.

Generic-SSM control for STU-v4. Same placement and zero-init residual logic,
but the spectral filter is replaced with a diagonal HiPPO-init Mamba block.
If both v4 (STU) and mamba-v4 tie baseline, that strengthens the conclusion
that no sequence-mixing on the VLM-side prefix helps for LIBERO action
prediction (rules out both Hankel structure AND generic recurrence).
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.mamba_layer import MambaLayer
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


class PI0MambaV4Pytorch(PI0Pytorch):
    """PI0 with Mamba SSM branch injected into the PaliGemma prefix input."""

    def __init__(self, config, mamba_state_dim: int = 4):
        super().__init__(config)
        self.mamba_state_dim = mamba_state_dim
        # Lazy-initialised in _build_prefix_mamba on first forward.
        self.prefix_mamba: MambaLayer | None = None
        self.prefix_mamba_out_proj: nn.Linear | None = None

        import openpi.models.gemma as _gemma
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        self._prefix_width = paligemma_config.width

        logging.info(
            "PI0MambaV4Pytorch: Mamba pre-input branch on prefix (image+lang) with "
            "state_dim=%d, width=%d (zero-init out_proj, lazy seq_len)",
            mamba_state_dim, self._prefix_width,
        )

    def _build_prefix_mamba(self, seq_len: int, device: torch.device) -> None:
        # MambaLayer's interface mirrors STULayer; no zero_init flag, so we wrap
        # it with a zero-init linear out_proj to enforce step-0 = baseline.
        self.prefix_mamba = MambaLayer(
            seq_len=seq_len,
            num_filters=self.mamba_state_dim,
            input_dim=self._prefix_width,
            output_dim=self._prefix_width,
            dtype=torch.float32,
        ).to(device)
        out_proj = nn.Linear(self._prefix_width, self._prefix_width, bias=False).to(device)
        nn.init.zeros_(out_proj.weight)
        self.prefix_mamba_out_proj = out_proj
        logging.info(
            "PI0MambaV4Pytorch: built prefix_mamba with seq_len=%d, state_dim=%d "
            "(zero-init out_proj)",
            seq_len, self.mamba_state_dim,
        )

    def _prefix_mamba_residual(
        self, prefix_embs: Tensor, prefix_pad_masks: Tensor
    ) -> Tensor:
        if self.prefix_mamba is None:
            self._build_prefix_mamba(prefix_embs.shape[1], prefix_embs.device)
        ctx = self.prefix_mamba(prefix_embs.to(torch.float32))
        ctx = self.prefix_mamba_out_proj(ctx)
        ctx = ctx.to(prefix_embs.dtype)
        ctx = ctx * prefix_pad_masks[:, :, None].to(ctx.dtype)
        return prefix_embs + ctx

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time
        )

        prefix_embs = self._prefix_mamba_residual(prefix_embs, prefix_pad_masks)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon:].to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return torch.nn.functional.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_embs = self._prefix_mamba_residual(prefix_embs, prefix_pad_masks)

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt * v_t
            time += dt
        return x_t
