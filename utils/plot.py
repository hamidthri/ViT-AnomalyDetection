import matplotlib.pyplot as plt
import os

def save_loss_plot(loss_values, model_save_path, start_epoch, num_epochs):
    os.makedirs(model_save_path, exist_ok=True)

    plt.plot(range(start_epoch + 1, num_epochs + 1), loss_values, marker='o', label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(model_save_path, "training_loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()

    print(f"Loss plot saved to {loss_plot_path}")
