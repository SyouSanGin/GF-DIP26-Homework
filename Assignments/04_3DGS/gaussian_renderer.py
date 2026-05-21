import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2].clamp(min=1.)  # (N, )
        
        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3] # (N, 2)
        
        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection: J ∈ R^{2×3}
        #   J = [[fx/tz, 0, -fx*tx/tz²],
        #        [0, fy/tz, -fy*ty/tz²]]
        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        fx, fy = K[0, 0], K[1, 1]
        tx, ty, tz = cam_points[:, 0], cam_points[:, 1], cam_points[:, 2]
        tz_clamp = tz.clamp(min=0.1)
        J_proj[:, 0, 0] = fx / tz_clamp
        J_proj[:, 0, 2] = -fx * tx / (tz_clamp * tz_clamp)
        J_proj[:, 1, 1] = fy / tz_clamp
        J_proj[:, 1, 2] = -fy * ty / (tz_clamp * tz_clamp)
        
        # Transform covariance to camera space: Σ' = R Σ R^T
        R_batch = R.unsqueeze(0).expand(N, 3, 3)
        covs_cam = torch.bmm(torch.bmm(R_batch, covs3d), R_batch.transpose(-1, -2))
        
        # Project to 2D: Σ₂D = J Σ' J^T
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))  # (N, 2, 2)
        
        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-2
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)
        
        # 2D Gaussian: exp(-½ (x-μ)^T Σ^{-1} (x-μ))
        # For Σ = [[a,b],[b,c]], Σ^{-1} = 1/(ac-b²) * [[c,-b],[-b,a]]
        a = covs2D[:, 0, 0]
        b = covs2D[:, 0, 1]
        c = covs2D[:, 1, 1]
        inv_det = 1.0 / (a * c - b * b).clamp(min=1e-6)
        inv_a = (c * inv_det).clamp(-1e6, 1e6)
        inv_b = (-b * inv_det).clamp(-1e6, 1e6)
        inv_c = (a * inv_det).clamp(-1e6, 1e6)
        
        # displacement: dx ∈ (N, H, W, 2)
        x_diff = dx[..., 0]  # (N, H, W)
        y_diff = dx[..., 1]  # (N, H, W)
        # quadratic form: [x y] Σ^{-1} [x y]^T
        quad = (inv_a.view(N, 1, 1) * x_diff * x_diff
              + 2.0 * inv_b.view(N, 1, 1) * x_diff * y_diff
              + inv_c.view(N, 1, 1) * y_diff * y_diff)
        
        gaussian = torch.exp(-0.5 * quad)  # (N, H, W)
    
        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor,                # (3, 1)
            return_radii: bool = False,
    ):
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Depth mask
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. Sort by depth
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]       # (N, 3)
        opacities = opacities[indices] # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        
        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. Compute α-blending weights: w_i = α_i * Π_{j<i} (1 - α_j)
        one_minus_alpha = 1.0 - alphas
        T = torch.cumprod(one_minus_alpha, dim=0)
        T_shifted = torch.cat([
            torch.ones(1, self.H, self.W, device=alphas.device),
            T[:-1]
        ], dim=0)
        weights = T_shifted * alphas  # (N, H, W)
        
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        if return_radii:
            # Estimate screen-space radius: sqrt of max eigenvalue of 2D cov
            a = covs2D[:, 0, 0]
            b = covs2D[:, 0, 1]
            c = covs2D[:, 1, 1]
            trace = a + c
            det = a * c - b * b
            # Eigenvalues: (trace ± sqrt(trace² - 4*det)) / 2
            eigen = 0.5 * (trace + torch.sqrt((trace * trace - 4.0 * det).clamp(min=0.0)))
            radii = 3.0 * torch.sqrt(eigen)  # 3-sigma radius
            return rendered, radii, indices
        
        return rendered
