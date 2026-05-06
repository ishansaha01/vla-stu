"""PI0 with STU as a *pre-transformer* spectral context branch.

This is a redesign of the original ``pi0_stu_pytorch.PI0STUPytorch`` informed
by the STU integration in ``stu-dreamer-jax``: post-hoc residual refinement of
hidden states does not help, but injecting STU output as an **additional input
branch** to the recurrent / sequence-mixing stage does. We therefore:

  * Compute the STU spectral context from the LDS *input* signal — the noisy
    action chunk ``x_t \\in R^{B,H,D_a}`` — rather than from post-attention
    hidden states. This matches Theorem 3.1 of the STU paper, which states
    that the spectral filters approximate the response of an LDS to its input.
  * **Add** the projected spectral context to the action token embeddings
    *before* they enter the action expert transformer, analogous to the 4th
    GRU input branch (``dynin3``) in the dreamer integration.
  * **Zero-initialise** the output projection so at step 0 the network is
    bit-identical to the baseline, and the spectral component grows only as
    gradients flow.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.stu_layer import STULayer


class PI0STUv2Pytorch(PI0Pytorch):
    """PI0 with STU spectral context injected into the action-expert input."""

    def __init__(self, config, stu_num_filters: int = 4):
        super().__init__(config)
        self.stu_num_filters = stu_num_filters

        import openpi.models.gemma as _gemma
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        width = action_expert_config.width

        # STU consumes the noisy action chunk directly.
        # Input dim = action_dim, output dim = width (action-expert hidden).
        # zero_init=True so the layer contributes 0 at step 0.
        self.action_stu = STULayer(
            seq_len=config.action_horizon,
            num_filters=stu_num_filters,
            input_dim=config.action_dim,
            output_dim=width,
            use_hankel_L=False,
            dtype=torch.float32,
            zero_init=True,
        )

        logging.info(
            "PI0STUv2Pytorch: STU pre-input branch with K=%d filters over "
            "action_horizon=%d, action_dim=%d -> width=%d (zero-init out)",
            stu_num_filters, config.action_horizon, config.action_dim, width,
        )

    def _stu_context(self, noisy_actions: Tensor, target_dtype: torch.dtype) -> Tensor:
        """Spectral context computed from the noisy action chunk.

        Returns a tensor matching ``action_token_embeddings`` shape so the
        caller can ``+=`` it onto the action positions of the suffix.
        """
        ctx = self.action_stu(noisy_actions.to(torch.float32))
        return ctx.to(target_dtype)

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

        # --- STU pre-transformer context injection ---
        # Add spectral context to the action token positions of the suffix
        # before the transformer sees them. The action tokens are the last
        # action_horizon positions of suffix_embs (per pi0_pytorch.embed_suffix).
        H = self.config.action_horizon
        stu_ctx = self._stu_context(x_t, suffix_embs.dtype)  # (B, H, W)
        suffix_embs = suffix_embs.clone()
        suffix_embs[:, -H:] = suffix_embs[:, -H:] + stu_ctx

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

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

        suffix_out = suffix_out[:, -H:].to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return torch.nn.functional.mse_loss(u_t, v_t, reduction="none")

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep
        )

        # Same STU pre-input injection at inference time.
        H = self.config.action_horizon
        stu_ctx = self._stu_context(x_t, suffix_embs.dtype)
        suffix_embs = suffix_embs.clone()
        suffix_embs[:, -H:] = suffix_embs[:, -H:] + stu_ctx

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -H:].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
