from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_data_loaders(train_data_path, test_data_path=None, batch_size=4):
    train_data = ImageFolder(root=train_data_path, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    if test_data_path:
        test_data = ImageFolder(root=test_data_path, transform=transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    return train_loader, test_loader
