"""Pick an inference device: --device auto -> CUDA, then Apple MPS, then CPU."""


def resolve_device(name):
    if name != 'auto':
        return name
    import torch
    if torch.cuda.is_available():
        return '0'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
