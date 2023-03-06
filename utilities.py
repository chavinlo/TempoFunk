from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torch

def pil_to_torch(image, device = 'cpu'):
    return (2 * (pil_to_tensor(image).to(dtype=torch.float32, device=device)) / 255) - 1

def torch_to_pil(x):
    return to_pil_image((x + 1) / 2)