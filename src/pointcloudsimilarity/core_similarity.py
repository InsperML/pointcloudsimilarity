import torch


class Similarity:
    def __call__(self, pc1, pc2, **kwargs):
        raise NotImplementedError


def resolve_device(device=None):
    """Return device, or auto-detect 'cuda', 'mps' or 'cpu' if None."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
