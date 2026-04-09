import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR, MultiStepLR

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

def kl_divergence_loss(mu, logvar):
    # Mean KL divergence per element keeps the magnitude stable across batch sizes.
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = torch.nan_to_num(kld, nan=0.0, posinf=1e4, neginf=0.0)
    return kld.mean()


def get_kl_weight(epoch, warmup_epochs=60, max_kl_weight=5e-4):
    # Linearly warm up KL regularization to reduce early training instability.
    progress = min(1.0, float(epoch + 1) / float(warmup_epochs))
    return max_kl_weight * progress


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs, scaler, kl_weight=1e-4):
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

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Zero the gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision.
        with autocast(enabled=(device.type == 'cuda')):
            recon, mu, logvar = model(image_rgb)
            recon_loss = criterion(recon, image_semantic)
            kl_loss = kl_divergence_loss(mu, logvar)
            loss = recon_loss + kl_weight * kl_loss

        if not torch.isfinite(loss):
            print(f'Skip non-finite train loss at epoch {epoch + 1}, step {i + 1}.')
            continue

        # Save sample images every 5 epochs
        if epoch % 100 == 0 and i == 0:
            save_images(image_rgb, image_semantic, recon, 'train_results', epoch)

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
            f'Total: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}'
        )

    if valid_steps == 0:
        return float('inf')
    return running_loss / valid_steps

def validate(model, dataloader, criterion, device, epoch, num_epochs, kl_weight=1e-4):
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
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            recon, mu, logvar = model(image_rgb)

            # Compute the loss
            recon_loss = criterion(recon, image_semantic)
            kl_loss = kl_divergence_loss(mu, logvar)
            loss = recon_loss + kl_weight * kl_loss

            if not torch.isfinite(loss):
                print(f'Skip non-finite val loss at epoch {epoch + 1}, step {i + 1}.')
                continue

            val_loss += loss.item()
            valid_batches += 1

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, recon, 'val_results', epoch)

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
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt', augment=True)
    val_dataset = FacadesDataset(list_file='val_list.txt', augment=False)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = FullyConvNetwork().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.5, 0.999), weight_decay=1e-4)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Add a learning rate scheduler for decay
    # scheduler = StepLR(optimizer, step_size=195, gamma=0.2)
    scheduler = MultiStepLR(optimizer, milestones=[1000, 2000,3000,4000], gamma=0.2)
    best_val_loss = float('inf')

    # Training loop
    num_epochs = 5000
    for epoch in range(num_epochs):
        kl_weight = get_kl_weight(epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            num_epochs,
            scaler,
            kl_weight=kl_weight,
        )
        val_loss = validate(model, val_loader, criterion, device, epoch, num_epochs, kl_weight=kl_weight)

        # Step the scheduler after each epoch
        scheduler.step()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}] summary: '
            f'train={train_loss:.4f}, val={val_loss:.4f}, kl_weight={kl_weight:.6f}'
        )

        # Save best checkpoint by validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/pix2pix_model_best.pth')

        # Save model checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
