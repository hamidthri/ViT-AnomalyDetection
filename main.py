import torch
import os
from config.args_parser import get_args
from models.vit_autoencoder import ViTEncoderDecoder, freeze_vit_layers
from models.utils import get_vgg_model
from utils.checkpoint import load_checkpoint
from train.train import train_model
from test.test import test_model_with_patch_analysis
from utils.data_loader import get_data_loaders
from utils.device_utils import get_device

if __name__ == "__main__":
    args = get_args()
    device = get_device()

    # Initialize models
    model = ViTEncoderDecoder().to(device)
    freeze_vit_layers(model)
    vgg = get_vgg_model(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # Train or test mode
    if args.mode == 'train':
        train_loader, _ = get_data_loaders(args.train_data_path)
        start_epoch = 0
        if args.checkpoint:
            start_epoch = load_checkpoint(model, optimizer, args.checkpoint) + 1

        num_epochs = 50
        model_save_path = args.model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        train_model(model, optimizer, train_loader, num_epochs, model_save_path, start_epoch=start_epoch, vgg=vgg, device=device)

    elif args.mode == 'test':
        _, test_loader = get_data_loaders(args.test_data_path)
        test_error = test_model_with_patch_analysis(model, test_loader, args.checkpoint, patch_size=8, stride=1, threshold=0.01, device=device)
        print(f"Average Test Reconstruction Error: {test_error:.4f}")
