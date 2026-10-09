"""The `AuroraCS` model: a frozen Aurora conditioned on a scalar forcing term."""

from datetime import timedelta
from typing import Optional

import torch
import torch.nn as nn
from aurora import Aurora, Batch, Swin3DBlockAdapter, Swin3DResidualAdapter

__all__ = ["AuroraCS"]


class AuroraCS(nn.Module):
    """Aurora with a small trainable pathway that conditions every Swin block on a scalar
    forcing term, via zero-initialised per-block adapters (AdaLN-Zero). At construction, the
    adapters are exactly zero, so the model is bit-for-bit identical to a stock, unconditioned
    `Aurora` until the conditioning pathway is trained."""

    def __init__(
        self,
        *,
        surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        window_size: tuple[int, int, int] = (2, 6, 12),
        encoder_depths: tuple[int, ...] = (6, 10, 8),
        encoder_num_heads: tuple[int, ...] = (8, 16, 32),
        decoder_depths: tuple[int, ...] = (8, 10, 6),
        decoder_num_heads: tuple[int, ...] = (32, 16, 8),
        patch_size: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        timestep: timedelta = timedelta(hours=6),
        max_history_size: int = 2,
        use_lora: bool = True,
        forcing_hidden_dim: int = 128,
        adapter_bottleneck_dim: int = 32,
        **aurora_kwargs,
    ) -> None:
        """Construct an instance of the model.

        Args:
            surf_vars (tuple[str, ...], optional): All surface-level variables supported by the
                model. Forwarded to `Aurora`.
            static_vars (tuple[str, ...], optional): All static variables supported by the model.
                Forwarded to `Aurora`.
            atmos_vars (tuple[str, ...], optional): All atmospheric variables supported by the
                model. Forwarded to `Aurora`.
            window_size (tuple[int, int, int], optional): Window size of the underlying Swin
                transformer. Forwarded to `Aurora`.
            encoder_depths (tuple[int, ...], optional): Number of blocks in each encoder layer.
                Forwarded to `Aurora`.
            encoder_num_heads (tuple[int, ...], optional): Number of attention heads in each
                encoder layer. Forwarded to `Aurora`.
            decoder_depths (tuple[int, ...], optional): Number of blocks in each decoder layer.
                Forwarded to `Aurora`.
            decoder_num_heads (tuple[int, ...], optional): Number of attention heads in each
                decoder layer. Forwarded to `Aurora`.
            patch_size (int, optional): Patch size. Forwarded to `Aurora`.
            embed_dim (int, optional): Patch embedding dimension. Forwarded to `Aurora`.
            num_heads (int, optional): Number of attention heads in the aggregation and
                deaggregation blocks. Forwarded to `Aurora`.
            timestep (timedelta, optional): Timestep of the model. Forwarded to `Aurora`.
            max_history_size (int, optional): Maximum number of history steps. Forwarded to
                `Aurora`.
            use_lora (bool, optional): Use LoRA adaptation in the wrapped `Aurora`. Forwarded to
                `Aurora`.
            forcing_hidden_dim (int, optional): Hidden dimension of the forcing encoder. Defaults
                to `128`.
            adapter_bottleneck_dim (int, optional): Width of the shared bottleneck between the
                forcing encoder and the per-block adapter heads. Keeps the helper network's
                parameter count near the ~5-10M target instead of the ~10x larger cost of
                per-block heads taking the full forcing embedding as input. Defaults to `32`.
            **aurora_kwargs: Additional keyword arguments forwarded directly to `Aurora`, e.g.
                `mlp_ratio`, `drop_rate`, `stochastic`, `variable_lead_time`.
        """
        super().__init__()

        self.aurora = Aurora(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            window_size=window_size,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            timestep=timestep,
            max_history_size=max_history_size,
            use_lora=use_lora,
            **aurora_kwargs,
        )
        for p in self.aurora.parameters():
            p.requires_grad_(False)

        # Read at runtime: it depends on the backbone's actual encoder/decoder depths.
        self.adapter_dims = self.aurora.backbone.adapter_dims

        self.forcing_encoder = nn.Sequential(
            nn.Linear(1, forcing_hidden_dim),
            nn.SiLU(),
            nn.Linear(forcing_hidden_dim, adapter_bottleneck_dim),
        )

        self.block_heads = nn.ModuleList(
            nn.Linear(adapter_bottleneck_dim, 6 * dim) for dim in self.adapter_dims
        )
        for head in self.block_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        batch: Batch,
        forcing: float,
        lead_times: Optional[torch.Tensor] = None,
    ) -> Batch:
        """Run the forward pass.

        Args:
            batch (:class:`aurora.Batch`): Batch to run the model on.
            forcing (float): Forcing term (e.g. CO2 concentration), shared across the batch.
            lead_times (:class:`torch.Tensor`, optional): Per-sample lead times of shape
                `(batch,)` in hours. Forwarded to `Aurora.forward`.

        Returns:
            :class:`aurora.Batch`: Prediction for the batch.
        """
        batch_size = next(iter(batch.surf_vars.values())).shape[0]
        ref = next(self.forcing_encoder.parameters())
        forcing_t = torch.full((batch_size, 1), float(forcing), device=ref.device, dtype=ref.dtype)

        z = self.forcing_encoder(forcing_t)

        adapters = []
        for dim, head in zip(self.adapter_dims, self.block_heads):
            scale_a, shift_a, gate_a, scale_m, shift_m, gate_m = head(z).split(dim, dim=-1)
            adapters.append(
                Swin3DBlockAdapter(
                    attention=Swin3DResidualAdapter(scale=scale_a, shift=shift_a, gate=gate_a),
                    mlp=Swin3DResidualAdapter(scale=scale_m, shift=shift_m, gate=gate_m),
                )
            )

        return self.aurora(batch, lead_times=lead_times, backbone_adapters=adapters)
