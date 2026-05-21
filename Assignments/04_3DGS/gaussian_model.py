import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class GaussianParameters:
    positions: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    covariance: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor


class GaussianModel(nn.Module):
    def __init__(self, points3D_xyz: torch.Tensor, points3D_rgb: torch.Tensor):
        super().__init__()
        self.n_points = len(points3D_xyz)

        self._init_positions(points3D_xyz)
        self._init_rotations()
        self._init_scales(points3D_xyz)
        self._init_colors(points3D_rgb)
        self._init_opacities()

        self.xyz_gradient_accum = torch.zeros((self.n_points, 1))
        self.denom = torch.zeros((self.n_points, 1))
        self.max_radii2D = torch.zeros((self.n_points,))

    def _init_positions(self, points3D_xyz: torch.Tensor) -> None:
        self.positions = nn.Parameter(
            torch.as_tensor(points3D_xyz, dtype=torch.float32)
        )

    def _init_rotations(self) -> None:
        initial_rotations = torch.zeros((self.n_points, 4))
        initial_rotations[:, 0] = 1.0
        self.rotations = nn.Parameter(initial_rotations)

    def _init_scales(self, points3D_xyz: torch.Tensor) -> None:
        K = min(50, self.n_points - 1)
        dists = torch.cdist(points3D_xyz, points3D_xyz)
        dists, _ = torch.topk(dists, k=K, dim=-1, largest=False)
        mean_dists = torch.mean(torch.sqrt(dists), dim=1, keepdim=True) * 2.0
        mean_dists = mean_dists.clamp(
            0.2 * torch.median(mean_dists), 3.0 * torch.median(mean_dists)
        )
        print('init_scales', torch.min(mean_dists).item(), torch.max(mean_dists).item())

        log_scales = torch.log(mean_dists)
        self.scales = nn.Parameter(log_scales.repeat(1, 3))

    def _init_colors(self, points3D_rgb: torch.Tensor) -> None:
        colors = torch.as_tensor(points3D_rgb, dtype=torch.float32) / 255.0
        colors = colors.clamp(0.001, 0.999)
        self.colors = nn.Parameter(torch.logit(colors))

    def _init_opacities(self) -> None:
        init_val = -1.5
        self.opacities = nn.Parameter(
            init_val * torch.ones((self.n_points, 1), dtype=torch.float32)
        )

    def _compute_rotation_matrices(self) -> torch.Tensor:
        q = F.normalize(self.rotations, dim=-1)
        w, x, y, z = q.unbind(-1)

        R00 = 1 - 2 * y * y - 2 * z * z
        R01 = 2 * x * y - 2 * w * z
        R02 = 2 * x * z + 2 * w * y
        R10 = 2 * x * y + 2 * w * z
        R11 = 1 - 2 * x * x - 2 * z * z
        R12 = 2 * y * z - 2 * w * x
        R20 = 2 * x * z - 2 * w * y
        R21 = 2 * y * z + 2 * w * x
        R22 = 1 - 2 * x * x - 2 * y * y

        return torch.stack(
            [R00, R01, R02, R10, R11, R12, R20, R21, R22], dim=-1
        ).reshape(-1, 3, 3)

    def compute_covariance(self) -> torch.Tensor:
        R = self._compute_rotation_matrices()
        scales = torch.exp(self.scales).clamp(min=1e-6)
        S = torch.diag_embed(scales)
        Covs3d = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
        return Covs3d

    def get_gaussian_params(self) -> GaussianParameters:
        return GaussianParameters(
            positions=self.positions,
            colors=torch.sigmoid(self.colors),
            opacities=torch.sigmoid(self.opacities),
            covariance=self.compute_covariance(),
            rotations=F.normalize(self.rotations, dim=-1),
            scales=torch.exp(self.scales),
        )

    def forward(self) -> Dict[str, torch.Tensor]:
        params = self.get_gaussian_params()
        return {
            'positions': params.positions,
            'covariance': params.covariance,
            'colors': params.colors,
            'opacities': params.opacities,
        }

    # ------------------------------------------------------------------
    #  Densification helpers
    # ------------------------------------------------------------------
    def add_densification_stats(
        self,
        update_filter: torch.Tensor,
    ):
        device = self.positions.device
        self._ensure_stats_device(device)

        grads = self.positions.grad.clone().detach()
        self.xyz_gradient_accum[update_filter] += torch.norm(
            grads[update_filter], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def _ensure_stats_device(self, device):
        if self.xyz_gradient_accum.device != device:
            self.xyz_gradient_accum = self.xyz_gradient_accum.to(device)
            self.denom = self.denom.to(device)
            self.max_radii2D = self.max_radii2D.to(device)

    def _replace_parameter(self, name: str, new_data: torch.Tensor):
        old = getattr(self, name)
        setattr(self, name, nn.Parameter(new_data.requires_grad_(True)))
        old_device = old.device
        getattr(self, name).data = getattr(self, name).data.to(old_device)

    def _prune_points(self, mask: torch.Tensor):
        valid_mask = ~mask
        kept = int(valid_mask.sum())
        if kept == self.positions.shape[0]:
            return

        self._replace_parameter('positions', self.positions.data[valid_mask])
        self._replace_parameter('rotations', self.rotations.data[valid_mask])
        self._replace_parameter('scales', self.scales.data[valid_mask])
        self._replace_parameter('colors', self.colors.data[valid_mask])
        self._replace_parameter('opacities', self.opacities.data[valid_mask])

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_mask]
        self.denom = self.denom[valid_mask]
        self.max_radii2D = self.max_radii2D[valid_mask]
        self.n_points = kept

        print(f'Pruned {int(mask.sum())} points, kept {kept}')

    def _cat_parameters(self, new_params: dict):
        device = self.positions.device
        for name, param_list in new_params.items():
            old = getattr(self, name).data
            new_tensor = torch.cat([old] + [p.to(device) for p in param_list], dim=0)
            self._replace_parameter(name, new_tensor)

        new_count = sum(p.shape[0] for p in new_params['positions'])
        self.xyz_gradient_accum = torch.cat(
            [self.xyz_gradient_accum, torch.zeros((new_count, 1), device=device)], dim=0
        )
        self.denom = torch.cat(
            [self.denom, torch.zeros((new_count, 1), device=device)], dim=0
        )
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros((new_count,), device=device)], dim=0
        )
        self.n_points += new_count

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        percent_dense: float,
        extent: float,
        max_screen_size: float,
    ):
        grads = self.xyz_gradient_accum / self.denom.clamp(min=1)
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()

        grads = grads.squeeze(-1)
        opacities = torch.sigmoid(self.opacities).squeeze(-1)
        scales = torch.exp(self.scales)
        max_scales = scales.max(dim=-1).values

        # ---- 1.  Prune  ----
        prune_mask = (
            (opacities < min_opacity)
            | (self.max_radii2D > max_screen_size)
        )

        # ---- 2.  Densify  ----
        high_grad = grads >= max_grad
        # Grad threshold filtering — skip Gaussians that are too large / too small
        split_mask = high_grad & (max_scales > percent_dense * extent)
        clone_mask = high_grad & (max_scales <= percent_dense * extent)

        if split_mask.any() or clone_mask.any():
            # --- clone ---
            new_params = {k: [] for k in ['positions', 'rotations', 'scales', 'colors', 'opacities']}
            for name in new_params:
                new_params[name].append(getattr(self, name).data[clone_mask])

            # --- split ---
            n_split = int(split_mask.sum())
            if n_split > 0:
                pos = self.positions.data[split_mask]
                rot = self.rotations.data[split_mask]
                scl = self.scales.data[split_mask]
                col = self.colors.data[split_mask]
                opa = self.opacities.data[split_mask]

                scales_real = torch.exp(scl)
                R = self._compute_rotation_matrices()[split_mask]

                # direction of largest scale
                largest_idx = scales_real.argmax(dim=-1)
                dirs = R[torch.arange(n_split), :, largest_idx]

                # two children: μ ± 0.5 * s_max * direction
                s_max = scales_real[torch.arange(n_split), largest_idx].unsqueeze(-1)
                offset = 0.5 * s_max * dirs

                pos1 = pos + offset
                pos2 = pos - offset

                shrink_factor = math.log(1.6)
                scl_new = scl - shrink_factor

                for name in new_params:
                    new_params[name].append(pos1 if name == 'positions' else (
                        scl_new if name == 'scales' else (
                            rot if name == 'rotations' else (
                                col if name == 'colors' else opa
                            )
                        )
                    ))
                for name in new_params:
                    new_params[name].append(pos2 if name == 'positions' else (
                        scl_new if name == 'scales' else (
                            rot if name == 'rotations' else (
                                col if name == 'colors' else opa
                            )
                        )
                    ))

            added = sum(p.shape[0] for p in new_params['positions'])
            if added > 0:
                self._cat_parameters(new_params)
                print(f'Densified: cloned {int(clone_mask.sum())}, split {n_split}→{n_split*2} (+{added} total)')

        # ---- 3.  Prune (after densify, re-evaluate)  ----
        opacities_after = torch.sigmoid(self.opacities).squeeze(-1)
        prune_mask = (
            (opacities_after < min_opacity)
            | (self.max_radii2D > max_screen_size)
        )
        if prune_mask.any():
            self._prune_points(prune_mask)

        # Reset max_radii2D for next round
        self.max_radii2D = torch.zeros((self.n_points,), device=self.positions.device)

    def reset_opacity(self, target_logit: float = -1.5):
        self.opacities.data.fill_(target_logit)
