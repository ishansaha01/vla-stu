"""PI0 with STU as a *pre-input* spectral branch on the PaliGemma prefix.

This is the VLM-side analog of v2. Reviewer feedback (Asher, midterm) noted
that adding STU only to the action expert may have been too narrow: since
fine-tuning starts from a converged pi05-libero checkpoint, slotting a fresh
spectral block onto the small action-expert input may be too brittle, while
the larger PaliGemma backbone has more redundancy to absorb a new branch.

v4 places the STU on the *prefix* embeddings (image + language tokens) before
they enter the VLM. Output is added as a zero-initialised residual on the
valid (non-pad) prefix positions, so step 0 is bit-identical to baseline.

Notes:
  * Prefix length is a runtime quantity (#images x SigLIP_tokens_per_image +
    lang_tokens). We lazy-initialise the STULayer on the first forward call,
    reading the seq_len from the embedded prefix.
  * Padding positions are masked from the spectral residual to avoid
    spurious gradients on dummy tokens.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.stu_layer import STULayer


class PI0STUv4Pytorch(PI0Pytorch):
    """PI0 with STU spectral branch injected into the PaliGemma prefix input."""

    def __init__(self, config, stu_num_filters: int = 4):
        super().__init__(config)
        self.stu_num_filters = stu_num_filters
        # Lazy-initialised in _build_prefix_stu on first forward.
        self.prefix_stu: STULayer | None = None

        import openpi.models.gemma as _gemma
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        self._prefix_width = paligemma_config.width

        logging.info(
            "PI0STUv4Pytorch: STU pre-input branch on prefix (image+lang) with "
            "K=%d filters, width=%d (zero-init out, lazy seq_len)",
            stu_num_filters, self._prefix_width,
        )

    def _build_prefix_stu(self, seq_len: int, device: torch.device) -> None:
        self.prefix_stu = STULayer(
            seq_len=seq_len,
            num_filters=self.stu_num_filters,
            input_dim=self._prefix_width,
            output_dim=self._prefix_width,
            use_hankel_L=False,
            dtype=torch.float32,
            zero_init=True,
        ).to(device)
        logging.info(
            "PI0STUv4Pytorch: built prefix_stu with seq_len=%d, K=%d",
            seq_len, self.stu_num_filters,
        )

    def _prefix_stu_residual(
        self, prefix_embs: Tensor, prefix_pad_masks: Tensor
    ) -> Tensor:
        """Compute zero-init STU residual; masked to non-pad prefix positions."""
        if self.prefix_stu is None:
            self._build_prefix_stu(prefix_embs.shape[1], prefix_embs.device)
        ctx = self.prefix_stu(prefix_embs.to(torch.float32))
        ctx = ctx.to(prefix_embs.dtype)
        # Zero out spectral context on padding positions.
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

        # --- STU pre-input residual on the VLM prefix ---
        prefix_embs = self._prefix_stu_residual(prefix_embs, prefix_pad_masks)

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
        # STU residual must also be applied at inference, before the VLM cache
        # is computed, so the cached keys/values incorporate the spectral
        # contribution from the prefix.
        prefix_embs = self._prefix_stu_residual(prefix_embs, prefix_pad_masks)

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
