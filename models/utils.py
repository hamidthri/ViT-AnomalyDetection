import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from utils.device_utils import get_device


def get_vgg_model(device):
    """Load a pre-trained VGG16 model for perceptual loss."""
    vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval().to(device)
    return vgg

# Perceptual Loss
def perceptual_loss(output, target, vgg):
    """
    Compute the perceptual loss using features extracted from a pretrained VGG16 network.

    Parameters:
        output: Reconstructed image from the model (Tensor).
        target: Original ground truth image (Tensor).
        vgg: Pretrained VGG16 model (features part).

    Returns:
        Perceptual loss (MSE) between the VGG features of the output and target images.
    """

    # Extract VGG features for both output and target images
    output_features = vgg(output)
    target_features = vgg(target)

    # Ensure that both features have the same size (interpolation if needed)
    if output_features.size() != target_features.size():
        target_features = F.interpolate(target_features, size=output_features.shape[2:], mode='bilinear',
                                        align_corners=False)
    return F.mse_loss(output_features, target_features)

# Combined Loss Function
def combined_loss(output, target, vgg):
    """
    Compute the combined loss, which is a weighted sum of perceptual loss and pixel-wise MSE.

    Parameters:
        output: Reconstructed image from the model (Tensor).
        target: Original ground truth image (Tensor).
        vgg: Pretrained VGG16 model (features part).

    Returns:
        Combined loss: 0.1 * Perceptual loss + 0.9 * MSE loss.
    """
    # Compute perceptual loss
    perceptual = perceptual_loss(output, target, vgg)
    # Compute pixel-wise MSE loss
    mse = F.mse_loss(output, target)
    # Return combined loss
    return 0.1 * perceptual + 0.9 * mse

# Extract patches from an image tensor
def extract_patches(imgs, patch_size=16, stride=16):
    """
    Extract patches from an image tensor.

    Parameters:
        imgs: Input tensor of shape (batch_size, channels, height, width).
        patch_size: Size of the patches to extract (default: 16x16).
        stride: Stride between patches (default: 16).

    Returns:
        Tensor of shape (batch_size, num_patches, channels, patch_size, patch_size)
        containing the extracted patches.
    """
    unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
    patches = unfold(imgs)
    patches = patches.permute(0, 2, 1)  # Shape: (batch_size, num_patches, patch_size * patch_size * channels)
    patches = patches.view(imgs.size(0), -1, imgs.size(1), patch_size, patch_size)  # Reshape into patch format
    return patches

# Compute the patch-wise mean squared error (MSE) between input and reconstructed patches
def compute_patch_errors(input_patches, reconstructed_patches):
    """
    Compute the reconstruction error (MSE) for each patch.

    Parameters:
        input_patches: Tensor of patches from the input image.
        reconstructed_patches: Tensor of patches from the reconstructed image.

    Returns:
        Tensor of shape (batch_size, num_patches) containing the MSE error for each patch.
    """
    # MSE over each patch: compute the mean squared difference over the channel and spatial dimensions
    errors = torch.mean((input_patches - reconstructed_patches) ** 2, dim=[2, 3, 4])
    return errors
