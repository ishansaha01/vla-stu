"""PI0 with STU as a smoothness prior on the *predicted velocity field*.

Where v1 (post-hoc residual) operated on hidden states and v2 (pre-input branch)
operated on the noisy action chunk, v3 operates directly on the predicted
velocity ``v_t \\in R^{B,H,D_a}``. This is the most natural location for a
spectral filter if the inductive bias we are reaching for is "actions in
robotics evolve smoothly over time": smooth the per-step velocity prediction
across the action horizon by adding a zero-initialised spectral residual.

Compared to v1, v3 operates on the actual action-trajectory signal (D_a=32)
rather than the 1024-dim hidden state. Compared to v2, the spectral block sits
on the gradient path between the per-token output projection and the loss,
giving it a direct reason to grow during training.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.stu_layer import STULayer


class PI0STUv3Pytorch(PI0Pytorch):
    """PI0 with STU smoothness prior applied to the predicted velocity v_t."""

    def __init__(self, config, stu_num_filters: int = 4, zero_init: bool = True):
        super().__init__(config)
        self.stu_num_filters = stu_num_filters

        # STU operates on v_t directly: input_dim = output_dim = action_dim.
        self.action_stu = STULayer(
            seq_len=config.action_horizon,
            num_filters=stu_num_filters,
            input_dim=config.action_dim,
            output_dim=config.action_dim,
            use_hankel_L=False,
            dtype=torch.float32,
            zero_init=zero_init,
        )

        logging.info(
            "PI0STUv3Pytorch: STU smoothness on velocity field with K=%d filters "
            "over action_horizon=%d, action_dim=%d (zero_init=%s)",
            stu_num_filters, config.action_horizon, config.action_dim, zero_init,
        )

    def _smooth_velocity(self, v_t: Tensor) -> Tensor:
        """Add zero-initialised spectral residual to predicted velocity."""
        return v_t + self.action_stu(v_t.to(torch.float32)).to(v_t.dtype)

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

        suffix_out = suffix_out[:, -self.config.action_horizon:].to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # --- STU smoothness residual on the predicted velocity field ---
        v_t = self._smooth_velocity(v_t)

        return torch.nn.functional.mse_loss(u_t, v_t, reduction="none")

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, timestep
        )

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
        suffix_out = suffix_out[:, -self.config.action_horizon:].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return self._smooth_velocity(v_t)
