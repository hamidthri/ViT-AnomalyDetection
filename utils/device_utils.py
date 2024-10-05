import torch


def get_device():
    """
    Returns the best available device for computations, prioritizing MPS (Metal), CUDA (GPU), and falling back to CPU.

    Returns:
        torch.device: The best available device ('mps', 'cuda', or 'cpu').
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) backend for computations on Apple Silicon.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device
