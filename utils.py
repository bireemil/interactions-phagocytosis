import numpy as np
import torch

def convert_string_to_tuple(pos:str):
    # print(pos, type(pos))
    tuple_of_floats = tuple(map(int, pos.strip("()").split(",")))
    return tuple_of_floats

def hex_distance(q1, r1, q2, r2):
    """Computes the hexagonal distance between two tiles in axial coordinates."""
    return (abs(q1 - q2) + abs(r1 - r2) + abs((q1 + r1) - (q2 + r2))) // 2


def linear_gradient(source_position, target_position):
    q1,r1 = source_position
    q2,r2 = target_position
    dist = hex_distance(q1, r1, q2, r2)
    if dist == 0:
        return 1
    else:
        return np.exp(-dist/3)
    
def lerp(start, end, alpha):
    return start + alpha * (end - start)


def weight_func(q, r, center_q,center_r):
    d = hex_distance(q, r, center_q, center_r)
    return torch.exp(-d/2)

def create_hex_kernel(radius=1, weight_func=None, device='cuda'):
    """
    Create a hexagonal kernel using PyTorch operations.
    
    Args:
        radius: Integer radius of the hexagonal kernel
        weight_func: Function that takes q, r coordinates as PyTorch tensors and returns weights
        device: 'cuda' or 'cpu'
    
    Returns:
        PyTorch tensor of shape (2*radius+1, 2*radius+1)
    """
    # Create coordinate grids
    size = 2 * radius + 1
    q_coords = torch.arange(-radius, radius + 1, device=device).view(-1, 1).repeat(1, size)
    r_coords = torch.arange(-radius, radius + 1, device=device).view(1, -1).repeat(size, 1)
    
    # Convert to cube coordinates for easier distance calculation
    x = q_coords
    z = r_coords
    y = -x - z

        # Apply weight function to coordinates
        # Convert to proper shape for broadcasting
    kernel = weight_func(q_coords, r_coords, 0, 0)
    
    # Normalize if sum is not zero
    if kernel.sum() != 0:
        kernel = kernel #/ kernel.sum()
    
    return kernel.unsqueeze(0).unsqueeze(0)


def convolve(dense_grid, kernel):
    """
    Perform convolution with given kernel on GPU.
    kernel: PyTorch tensor of shape (1, 1, H, W)
    """
    # Pad input grid
    pad_q = kernel.shape[2] // 2
    pad_r = kernel.shape[3] // 2
    padded_grid = torch.nn.functional.pad(
        dense_grid.unsqueeze(0).unsqueeze(0),
        (pad_r, pad_r, pad_q, pad_q),
        mode='constant',
        value=0
    )
    
    # Perform convolution
    result_tensor = torch.nn.functional.conv2d(padded_grid, kernel)
    result_tensor = result_tensor.squeeze()
                
    return result_tensor
