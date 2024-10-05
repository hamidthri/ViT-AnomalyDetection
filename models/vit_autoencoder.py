import torch.nn as nn
from timm import create_model

class ViTEncoderDecoder(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', image_size=224):
        super(ViTEncoderDecoder, self).__init__()

        self.encoder = create_model(vit_model_name, pretrained=True)
        self.encoder.reset_classifier(0)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=768, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, padding=1, output_padding=1),  # Final output layer
            nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded_features = self.encoder.forward_features(x)  # Shape: (batch_size, num_patches, hidden_dim)

        batch_size, num_patches, hidden_dim = encoded_features.shape
        h = w = int(num_patches ** 0.5)
        encoded_features = encoded_features[:, 1:, :].permute(0, 2, 1).view(batch_size, hidden_dim, h, w)

        reconstructed_image = self.decoder(encoded_features)
        return reconstructed_image


def freeze_vit_layers(model):
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.encoder.blocks[-1].parameters():
        param.requires_grad = True
