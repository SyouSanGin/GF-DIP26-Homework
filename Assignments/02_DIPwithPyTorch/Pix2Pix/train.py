import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import vgg16, VGG16_Weights
from facades_dataset import FacadesDataset
from edges2shoes_dataset import E2SDataset
from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import MultiStepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    max_images = min(num_images, inputs.size(0))
    for i in range(max_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)


def _sobel_gradient(x):
    """Compute Sobel gradient magnitude on RGB batches."""
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)


def total_variation_loss(x):
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def multiscale_l1_loss(pred, target, scales=(1, 2, 4)):
    losses = []
    for s in scales:
        if s == 1:
            p = pred
            t = target
        else:
            p = F.avg_pool2d(pred, kernel_size=s, stride=s)
            t = F.avg_pool2d(target, kernel_size=s, stride=s)
        losses.append(F.l1_loss(p, t))
    return sum(losses) / len(losses)


class PerceptualLoss(nn.Module):
    """VGG16 feature-space L1 loss in [0,1] domain."""

    def __init__(self, device):
        super().__init__()
        try:
            weights = VGG16_Weights.IMAGENET1K_V1
            vgg_features = vgg16(weights=weights).features[:16].eval()
        except Exception:
            # Fallback for offline environments without pretrained weights.
            vgg_features = vgg16(weights=None).features[:16].eval()

        for p in vgg_features.parameters():
            p.requires_grad = False
        self.vgg = vgg_features.to(device)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def _normalize(self, x):
        x01 = (x + 1.0) * 0.5
        return (x01 - self.mean) / self.std

    def forward(self, pred, target):
        pred_n = self._normalize(pred)
        target_n = self._normalize(target)
        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))

def kl_divergence_loss(mu, logvar):
    # Mean KL divergence per element keeps the magnitude stable across batch sizes.
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = torch.nan_to_num(kld, nan=0.0, posinf=1e4, neginf=0.0)
    return kld.mean()


def get_kl_weight(epoch, warmup_epochs=60, max_kl_weight=5e-4):
    # Linearly warm up KL regularization to reduce early training instability.
    progress = min(1.0, float(epoch + 1) / (float(warmup_epochs) + 0.0001))
    return max_kl_weight * progress


def get_latent_noise_scale(epoch, warmup_epochs=40, max_scale=0.35):
    # Start close to deterministic decoding and gradually add stochasticity.
    progress = min(1.0, float(epoch + 1) / float(warmup_epochs))
    return max_scale * progress


def compute_total_loss(recon, target_rgb, mu, logvar, aux_outputs, perceptual_fn, epoch):
    # Dynamic edge loss stabilizes early training before focusing more on details.
    edge_weight = 0.08 if epoch < 5 else 0.12

    recon_l1 = F.l1_loss(recon, target_rgb)
    recon_ms = multiscale_l1_loss(recon, target_rgb)
    recon_edge = F.l1_loss(_sobel_gradient(recon), _sobel_gradient(target_rgb))
    # recon_perc = perceptual_fn(recon, target_rgb)
    recon_perc = torch.zeros_like(recon_l1)  # Disable perceptual loss for now to speed up training and reduce memory usage.
    recon_tv = total_variation_loss(recon)
    recon_kl = kl_divergence_loss(mu, logvar)

    aux_loss = torch.tensor(0.0, device=recon.device)
    if aux_outputs is not None:
        for aux in aux_outputs:
            aux_loss = aux_loss + F.l1_loss(aux, target_rgb)
        aux_loss = aux_loss / max(len(aux_outputs), 1)

    kl_weight = get_kl_weight(epoch, warmup_epochs=1, max_kl_weight=1e-6)
    total = (
        0.6 * recon_l1
        + 0.7 * recon_ms
        + edge_weight * recon_edge
        + 0.15 * recon_perc
        + 0.015 * recon_tv * 0 # no TV
        + 0.15 * aux_loss
        + kl_weight * recon_kl
    )
    loss_terms = {
        'total': total,
        'l1': recon_l1,
        'ms': recon_ms,
        'edge': recon_edge,
        'perc': recon_perc,
        'tv': recon_tv,
        'aux': aux_loss,
        'kl': recon_kl,
        'kl_weight': torch.tensor(kl_weight, device=recon.device),
    }
    return loss_terms


def train_one_epoch(model, dataloader, optimizer, perceptual_fn, device, epoch, num_epochs, scaler):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.train()
    running_loss = 0.0
    valid_steps = 0

    for i, (image_edge, image_rgb) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device, non_blocking=True)
        image_edge = image_edge.to(device, non_blocking=True)

        # Zero the gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision.
        latent_noise_scale = get_latent_noise_scale(epoch, warmup_epochs=5, max_scale=0.35)
        with autocast(enabled=(device.type == 'cuda')):
            # Edge map -> RGB image
            recon, mu, logvar, aux_outputs = model(image_edge, latent_noise_scale=latent_noise_scale)
            losses = compute_total_loss(recon, image_rgb, mu, logvar, aux_outputs, perceptual_fn, epoch)
            loss = losses['total']

        if not torch.isfinite(loss):
            print(f'Skip non-finite train loss at epoch {epoch + 1}, step {i + 1}.')
            continue

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_edge, image_rgb, recon, 'train_results', epoch)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Update running loss
        running_loss += loss.item()
        valid_steps += 1

        # Print loss information
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], '
            f"Total: {loss.item():.4f}, L1: {losses['l1'].item():.4f}, MS: {losses['ms'].item():.4f}, Edge: {losses['edge'].item():.4f}, "
            f"Perc: {losses['perc'].item():.4f}, Aux: {losses['aux'].item():.4f}, KL: {losses['kl'].item():.4f}, "
            f"KLw: {losses['kl_weight'].item():.6f}, Znoise: {latent_noise_scale:.3f}"
        )

    if valid_steps == 0:
        return float('inf')
    return running_loss / valid_steps

def validate(model, dataloader, perceptual_fn, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for i, (image_edge, image_rgb) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device, non_blocking=True)
            image_edge = image_edge.to(device, non_blocking=True)

            # Forward pass
            # Edge map -> RGB image
            recon, mu, logvar, aux_outputs = model(image_edge)
            losses = compute_total_loss(recon, image_rgb, mu, logvar, aux_outputs, perceptual_fn, epoch)
            loss = losses['total']

            if not torch.isfinite(loss):
                print(f'Skip non-finite val loss at epoch {epoch + 1}, step {i + 1}.')
                continue

            val_loss += loss.item()
            valid_batches += 1

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_edge, image_rgb, recon, 'val_results', epoch)

    # Calculate average validation loss
    if valid_batches == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: NaN (all batches invalid)')
        return float('inf')
    else:
        avg_val_loss = val_loss / valid_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        return avg_val_loss

def main():
    """
    Main function to set up the training and validation processes.
    """
    parser = argparse.ArgumentParser(description='Train Pix2Pix Model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to the checkpoint to resume training from.')
    args = parser.parse_args()

    # Set device to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    # Initialize datasets and dataloaders
    # train_dataset = FacadesDataset(list_file='train_list.txt', augment=True)
    # val_dataset = FacadesDataset(list_file='val_list.txt', augment=False)
    train_dataset = E2SDataset(dataset_root='datasets/edges2shoes/train', augment=True)
    val_dataset = E2SDataset(dataset_root='datasets/edges2shoes/val', augment=False)

    BS = 64
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = FullyConvNetwork().to(device)

    # Optional: Resume from checkpoint
    resume_checkpoint = args.checkpoint
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint: {resume_checkpoint}")
        state_dict = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        # Handle possible 'module.' prefix if saved inconsistently
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        # Try to infer start epoch from filename
        try:
            parts = os.path.basename(resume_checkpoint).split('_')
            epoch_str = parts[-1].split('.')[0]
            start_epoch = int(epoch_str)
            print(f"Resuming from epoch {start_epoch}")
        except:
            print("Could not infer epoch from checkpoint name, starting from 0")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)

    perceptual_fn = PerceptualLoss(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, betas=(0.5, 0.999), weight_decay=5e-5)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Add a learning rate scheduler for decay
    # scheduler = StepLR(optimizer, step_size=195, gamma=0.2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.5)
    # Fast forward scheduler if resuming
    for _ in range(start_epoch):
        scheduler.step()

    best_val_loss = float('inf')

    # Training loop
    num_epochs = 100
    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            perceptual_fn,
            device,
            epoch,
            num_epochs,
            scaler,
        )
        val_loss = validate(model, val_loader, perceptual_fn, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler.step()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}] summary: '
            f'train={train_loss:.4f}, val={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6e}'
        )

        # Save best checkpoint by validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, 'checkpoints/pix2pix_model_best.pth')

        # Save model checkpoint every 5 epochs
        if (epoch + 1) % 1 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
