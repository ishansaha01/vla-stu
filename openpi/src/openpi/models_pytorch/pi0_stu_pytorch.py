"""PI0 model with Spectral Transfer Unit (STU) on the action prediction head.

The STU layer is inserted between the transformer suffix output and the
action output projection, applying spectral filtering across the action
horizon dimension to capture long-range temporal dependencies.
"""

import logging

import torch
from torch import Tensor
from torch import nn

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.stu_layer import STULayer


class PI0STUPytorch(PI0Pytorch):
    """PI0 with STU layer on the action head.

    Adds a residual STU block between the transformer output and the
    action output projection:
        suffix_out -> LayerNorm -> STU -> residual add -> action_out_proj
    """

    def __init__(self, config, stu_num_filters: int = 8):
        super().__init__(config)
        self.stu_num_filters = stu_num_filters

        # The transformer expert output width
        import openpi.models.gemma as _gemma
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        width = action_expert_config.width

        # STU operates over the action horizon sequence: [B, action_horizon, width]
        self.action_stu = STULayer(
            seq_len=config.action_horizon,
            num_filters=stu_num_filters,
            input_dim=width,
            output_dim=width,
            use_hankel_L=False,
            dtype=torch.float32,
        )
        self.stu_layer_norm = nn.LayerNorm(width)

        logging.info(
            f"PI0STUPytorch: Added STU with K={stu_num_filters} filters "
            f"over action_horizon={config.action_horizon}, width={width}"
        )

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Training forward pass with STU on action head."""
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

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # --- STU block with residual connection ---
        residual = suffix_out
        suffix_out = self.stu_layer_norm(suffix_out)
        suffix_out = self.action_stu(suffix_out)
        suffix_out = suffix_out + residual

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return torch.nn.functional.mse_loss(u_t, v_t, reduction="none")

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """Single denoising step with STU on action head."""
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
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # --- STU block with residual connection ---
        residual = suffix_out
        suffix_out = self.stu_layer_norm(suffix_out)
        suffix_out = self.action_stu(suffix_out)
        suffix_out = suffix_out + residual

        return self.action_out_proj(suffix_out)
