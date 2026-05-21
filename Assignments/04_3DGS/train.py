import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
import cv2
import os

from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer
from data_utils import ColmapDataset


@dataclass
class TrainConfig:
    num_epochs: int = 100
    batch_size: int = 1
    grad_clip: float = 1.0
    save_every: int = 20
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    debug_every: int = 1
    debug_samples: int = 4

    # Densification
    densify_from_epoch: int = 3
    densify_until_epoch: int = 70
    densify_interval: int = 2
    opacity_reset_interval: int = 20
    densify_grad_threshold: float = 0.0002
    densify_min_opacity: float = 0.005
    densify_percent_dense: float = 0.01
    densify_extent: float = 3.0
    max_screen_size: float = 100.0


class GaussianTrainer:
    def __init__(self, model, renderer, config, device):
        self.model = model.to(device)
        self.renderer = renderer.to(device)
        self.config = config
        self.device = device

        self._setup_optimizer()

        Path(config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(config.log_dir).mkdir(exist_ok=True, parents=True)
        self.debug_indices = None

    def _setup_optimizer(self):
        self.optimizer = torch.optim.Adam([
            {'params': [self.model.positions], 'lr': 0.00016, 'name': 'xyz'},
            {'params': [self.model.colors], 'lr': 0.0025, 'name': 'color'},
            {'params': [self.model.opacities], 'lr': 0.005, 'name': 'opacity'},
            {'params': [self.model.scales], 'lr': 0.005, 'name': 'scaling'},
            {'params': [self.model.rotations], 'lr': 0.001, 'name': 'rotation'},
        ], lr=0.001, eps=1e-15)

    def save_debug_images(self, epoch, rendered_images, gt_images, image_paths):
        rendered = rendered_images.detach().cpu().numpy()
        gt = gt_images.detach().cpu().numpy()

        gt_cells, rendered_cells = [], []
        for b in range(rendered.shape[0]):
            r = (rendered[b] * 255).clip(0, 255).astype(np.uint8)
            g = (gt[b] * 255).clip(0, 255).astype(np.uint8)
            r = cv2.cvtColor(r, cv2.COLOR_RGB2BGR)
            g = cv2.cvtColor(g, cv2.COLOR_RGB2BGR)
            label = Path(image_paths[b]).stem
            cv2.putText(g, label, (6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 255, 0), 1, cv2.LINE_AA)
            gt_cells.append(g)
            rendered_cells.append(r)

        gt_row = np.concatenate(gt_cells, axis=1)
        rendered_row = np.concatenate(rendered_cells, axis=1)
        cv2.putText(gt_row, "GT", (6, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            rendered_row, f"Rendered (#{self.model.n_points})", (6, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )
        grid = np.concatenate([gt_row, rendered_row], axis=0)
        output_path = Path(self.config.log_dir) / f"epoch_{epoch:04d}.png"
        cv2.imwrite(str(output_path), grid)

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{epoch:06d}.pt"
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def visualize_rendering(self, dataset, save_vid_path):
        print("Generating rendering visualization...")
        sample = dataset[0]
        K = sample['K'].to(self.device)
        H, W = sample['image'].shape[:2]
        out = cv2.VideoWriter(
            save_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 3, (W * 2, H)
        )
        with torch.no_grad():
            gp = self.model()
        for data_item in tqdm(dataset, desc="Rendering frames"):
            R_t = data_item['R'].to(self.device)
            t_t = data_item['t'].to(self.device).reshape(-1, 3)
            with torch.no_grad():
                frame = self.renderer(
                    means3D=gp['positions'], covs3d=gp['covariance'],
                    colors=gp['colors'], opacities=gp['opacities'],
                    K=K.squeeze(0), R=R_t.squeeze(0), t=t_t.squeeze(0),
                )
            f = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            o = (data_item['image'].cpu().numpy() * 255).astype(np.uint8)
            vis = cv2.cvtColor(np.concatenate((o, f), axis=1), cv2.COLOR_RGB2BGR)
            out.write(vis)
        out.release()
        print(f"Video saved to: {save_vid_path}")

    def train_step(self, batch, in_train=True):
        images = batch['image'].to(self.device)
        K = batch['K'].to(self.device)
        R = batch['R'].to(self.device)
        t = batch['t'].to(self.device).reshape(-1, 3)

        gp = self.model()
        result = self.renderer(
            means3D=gp['positions'], covs3d=gp['covariance'],
            colors=gp['colors'], opacities=gp['opacities'],
            K=K.squeeze(0), R=R.squeeze(0), t=t.squeeze(0),
            return_radii=in_train,
        )
        if in_train:
            rendered, radii, sort_indices = result
        else:
            rendered = result

        rendered = rendered.unsqueeze(0)

        if not in_train:
            return rendered

        loss = torch.abs(rendered - images).mean()
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

        # Accumulate densification stats
        with torch.no_grad():
            update_filter = torch.ones(self.model.n_points, dtype=torch.bool, device=self.device)
            self.model.add_densification_stats(update_filter)

        # Track max radii
        self.model.max_radii2D = torch.maximum(self.model.max_radii2D, radii[sort_indices])

        self.optimizer.step()
        return loss.item(), rendered

    def _densify_if_needed(self, epoch):
        cfg = self.config
        if (
            epoch < cfg.densify_from_epoch
            or epoch > cfg.densify_until_epoch
            or epoch % cfg.densify_interval != 0
        ):
            return

        extent = cfg.densify_extent
        print(
            f'\n--- Densify @ epoch {epoch} '
            f'(grad_thr={cfg.densify_grad_threshold}, '
            f'n={self.model.n_points}) ---'
        )
        self.model.densify_and_prune(
            max_grad=cfg.densify_grad_threshold,
            min_opacity=cfg.densify_min_opacity,
            percent_dense=cfg.densify_percent_dense,
            extent=extent,
            max_screen_size=cfg.max_screen_size,
        )
        print(f'Points after densify+prune: {self.model.n_points}')

        self._setup_optimizer()

    def _reset_opacity_if_needed(self, epoch):
        if epoch > 0 and epoch % self.config.opacity_reset_interval == 0:
            print(f'\n--- Resetting opacity @ epoch {epoch} ---')
            self.model.reset_opacity()

    def train(self, train_loader):
        if self.debug_indices is None:
            ds_size = len(train_loader.dataset)
            self.debug_indices = np.random.choice(
                ds_size, min(self.config.debug_samples, ds_size), replace=False
            )

        for epoch in range(self.config.num_epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            epoch_loss = 0.0
            num_batches = 0

            for batch in pbar:
                loss, _ = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix({'loss': f"{epoch_loss / num_batches:.4f}"})

            if epoch % self.config.save_every == 0:
                self.save_checkpoint(epoch)

            if epoch % self.config.debug_every == 0:
                rendered_list, gt_list, path_list = [], [], []
                for idx in self.debug_indices:
                    sample = train_loader.dataset[idx]
                    b = {
                        k: (v.unsqueeze(0) if torch.is_tensor(v) else [v])
                        for k, v in sample.items()
                    }
                    with torch.no_grad():
                        r = self.train_step(b, in_train=False)
                    rendered_list.append(r.squeeze(0))
                    gt_list.append(sample['image'])
                    path_list.append(sample['image_path'])
                self.save_debug_images(
                    epoch=epoch,
                    rendered_images=torch.stack(rendered_list, dim=0),
                    gt_images=torch.stack(gt_list, dim=0),
                    image_paths=path_list,
                )

            self._densify_if_needed(epoch)
            self._reset_opacity_if_needed(epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D Gaussian Splatting with densification')
    parser.add_argument('--colmap_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--debug_every', type=int, default=1)
    parser.add_argument('--debug_samples', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = TrainConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=os.path.join(args.checkpoint_dir, "debug_images"),
        debug_every=args.debug_every,
        debug_samples=args.debug_samples,
    )

    dataset = ColmapDataset(args.colmap_dir)
    train_loader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )

    sample = dataset[0]['image']
    H, W = sample.shape[:2]

    model = GaussianModel(
        points3D_xyz=dataset.points3D_xyz,
        points3D_rgb=dataset.points3D_rgb,
    )
    renderer = GaussianRenderer(image_height=H, image_width=W)
    trainer = GaussianTrainer(model, renderer, config, device)

    start_epoch = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        config.num_epochs -= start_epoch

    print(f"Training on {len(dataset)} images for {config.num_epochs} epochs")
    print(f"Initial points: {model.n_points}")
    trainer.train(train_loader)
    print("Training completed!")

    trainer.visualize_rendering(
        dataset,
        os.path.join(args.checkpoint_dir, "debug_rendering.mp4"),
    )


if __name__ == "__main__":
    main()
