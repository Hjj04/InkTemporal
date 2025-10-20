# /.../DiffSynth-Studio/diffsynth/modules/temporal_module.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalModule(nn.Module):
    """
    TemporalModule skeleton for latent-space warp & fusion.

    - Warp previous latent to current frame using a flow (pixel offsets).
    - Refine warped latent via small conv block.
    - Fuse warped latent and current latent via learnable scalar alpha gating.
    - Optionally fuse style embeddings via a small MLP.
    """

    def __init__(
        self,
        latent_channels: int,
        style_dim: int = None,
        learnable_alpha: bool = True,
        alpha_init: float = 0.5,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.style_dim = style_dim

        # alpha param stored in logit space (so sigmoid(alpha_param) in (0,1))
        # init_logit = self._inv_sigmoid(alpha_init)
        # if learnable_alpha:
        #     self.alpha_param = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))
        # else:
        #     self.register_buffer("alpha_param", torch.tensor(init_logit, dtype=torch.float32))

        # 修正后的硬编码 (用于评估)
        FIXED_ALPHA = 0.3
        init_logit = self._inv_sigmoid(FIXED_ALPHA)
        self.register_buffer("alpha_param", torch.tensor(init_logit, dtype=torch.float32))

        # small refiner conv block for warped latent
        self.refiner = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
        )

        # optional style fusion MLP
        if style_dim is not None:
            self.style_fuse = nn.Sequential(
                nn.Linear(style_dim * 2, style_dim),
                nn.ReLU(inplace=True),
                nn.Linear(style_dim, style_dim),
            )
        else:
            self.style_fuse = None

    @staticmethod
    def _inv_sigmoid(x: float) -> float:
        # numerically stable inverse sigmoid (logit)
        x = float(max(min(x, 1.0 - 1e-6), 1e-6))
        return math.log(x / (1.0 - x))

    def _get_alpha(self) -> torch.Tensor:
        # returns scalar in (0,1)
        return torch.sigmoid(self.alpha_param)

    def warp_latent(self, z_prev: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp latent z_prev using a flow.

        Args:
            z_prev: [B, C, H, W]
            flow:   [B, 2, H, W] - pixel offsets (dx, dy) in pixel units

        Returns:
            z_warp: [B, C, H, W] warped via grid_sample
        """
        if flow is None:
            return z_prev

        B, C, H, W = z_prev.shape
        device = z_prev.device
        dtype = z_prev.dtype

        # create normalized meshgrid in range [-1, 1]
        grid_y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        grid_x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")  # [H, W]
        # stack as (x, y)
        base_grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

        # flow is pixel offsets (dx, dy). Convert to normalized coords:
        # dx_norm = dx / (W/2), dy_norm = dy / (H/2)
        dx = flow[:, 0:1, :, :]  # [B,1,H,W]
        dy = flow[:, 1:2, :, :]
        dx_norm = dx / (W / 2.0)
        dy_norm = dy / (H / 2.0)
        offset = torch.cat((dx_norm, dy_norm), dim=1)  # [B,2,H,W]
        offset = offset.permute(0, 2, 3, 1)  # [B,H,W,2]

        sample_grid = base_grid + offset  # [B,H,W,2], in normalized coords
        # grid_sample expects grid in shape [B, H, W, 2] with (x,y) order
        z_warp = F.grid_sample(z_prev, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)

        return z_warp

    def forward(
        self,
        z_prev: torch.Tensor,
        z_cur: torch.Tensor,
        s_prev: torch.Tensor = None,
        s_cur: torch.Tensor = None,
        flow: torch.Tensor = None,
    ):
        """
        Args:
            z_prev: [B, C, H, W] previous-frame latent
            z_cur:  [B, C, H, W] current-frame latent
            s_prev: [B, S] or None (previous style embedding)
            s_cur:  [B, S] or None (current style embedding)
            flow:   [B, 2, H, W] pixel offsets (dx,dy), or None

        Returns:
            z_fused: [B, C, H, W]
            s_fused: [B, S] or None
            aux: dict with debug info (alpha, stats)
        """
        assert z_prev.shape == z_cur.shape, "z_prev and z_cur must have same shape"

        # ensure same device/dtype
        device = z_cur.device
        dtype = z_cur.dtype

        if flow is not None:
            flow = flow.to(device=device, dtype=dtype)

        # warp previous latent to current frame
        z_warp = self.warp_latent(z_prev, flow)

        # optional refine
        z_warp_refined = self.refiner(z_warp)

        # gating alpha (scalar)
        alpha = self._get_alpha()  # scalar torch
        B, C, H, W = z_cur.shape
        alpha_exp = alpha.view(1, 1, 1, 1).expand(B, C, 1, 1)

        # fused latent: alpha * warped + (1-alpha) * current
        z_fused = alpha_exp * z_warp_refined + (1.0 - alpha_exp) * z_cur

        # style fusion (if embeddings provided)
        s_fused = None
        if self.style_fuse is not None and (s_prev is not None) and (s_cur is not None):
            # ensure s_prev/s_cur are same device/dtype
            s_prev = s_prev.to(device=device, dtype=dtype)
            s_cur = s_cur.to(device=device, dtype=dtype)
            s_cat = torch.cat([s_prev, s_cur], dim=-1)
            s_fused = self.style_fuse(s_cat)

        aux = {
            "alpha": float(alpha.detach().cpu().item()),
            "z_warp_mean": float(z_warp_refined.mean().detach().cpu().item()),
        }
        return z_fused, s_fused, aux