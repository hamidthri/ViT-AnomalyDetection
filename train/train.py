from utils.checkpoint import save_checkpoint
from models.utils import combined_loss
import torch

from utils.device_utils import get_device

def train_model(model, optimizer, train_loader, num_epochs, model_save_path, start_epoch=0, vgg=None, device=None):
    model.train()
    loss_values = []

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = combined_loss(outputs, images, vgg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        loss_values.append(epoch_loss)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(epoch, model, optimizer, epoch_loss,
                            f'{model_save_path}/vit_autoencoder_epoch_{epoch + 1}.pth')

    # Plot loss curve
    from utils.plot import save_loss_plot
    save_loss_plot(loss_values, model_save_path, start_epoch, num_epochs)
