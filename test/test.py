import torch
import matplotlib.pyplot as plt
from models.utils import combined_loss, extract_patches, compute_patch_errors
from utils.checkpoint import load_checkpoint



def test_model_with_patch_analysis(model, test_loader, checkpoint_path=None, patch_size=16, stride=16, threshold=0.01, device='cpu'):
    if checkpoint_path:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)  # Dummy optimizer to load state
        load_checkpoint(model, optimizer, checkpoint_path)

    model = model.to(device)
    model.eval()

    total_reconstruction_error = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            outputs = model(images)

            reconstruction_error = combined_loss(outputs, images)
            total_reconstruction_error += reconstruction_error.item()

            input_patches = extract_patches(images, patch_size=patch_size, stride=stride)
            reconstructed_patches = extract_patches(outputs, patch_size=patch_size, stride=stride)

            patch_errors = compute_patch_errors(input_patches, reconstructed_patches)

            batch_size = patch_errors.size(0)
            num_patches = patch_errors.size(1)
            h = w = int(num_patches ** 0.5)
            error_maps = patch_errors.view(batch_size, h, w)

            anomaly_masks = (error_maps > threshold).float()

            plt.figure(figsize=(20, 12))
            for i in range(min(len(images), 4)):
                plt.subplot(4, 4, i + 1)
                plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.title("Original")
                plt.axis('off')

                plt.subplot(4, 4, i + 5)
                plt.imshow(outputs[i].permute(1, 2, 0).cpu().numpy())
                plt.title("Reconstructed")
                plt.axis('off')

                diff_image = torch.abs(images[i] - outputs[i]).permute(1, 2, 0).cpu().numpy()
                plt.subplot(4, 4, i + 9)
                plt.imshow(diff_image)
                plt.title("Difference")
                plt.axis('off')

                plt.subplot(4, 4, i + 13)
                plt.imshow(error_maps[i].cpu().detach().numpy(), cmap='hot')
                plt.title("Patch-wise Error Map")
                plt.colorbar()
                plt.axis('off')

            plt.show()

            print(f"Batch Reconstruction Error: {reconstruction_error.item():.4f}")

            anomaly_percentage = torch.mean(anomaly_masks).item() * 100
            print(f"Anomalous Patches: {anomaly_percentage:.2f}%")

    return total_reconstruction_error / len(test_loader)
