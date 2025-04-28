"""
heatmap_utils.py

Helper functions to create 2D Gaussian heatmaps from landmark coordinates.
"""

import torch
import math

def generate_gaussian_heatmap(center, shape, sigma=2):
    """
    Generate a single 2D Gaussian heatmap.

    Args:
        center (tuple): (x, y) center coordinates (normalized 0-1)
        shape (tuple): (height, width) of output heatmap
        sigma (float): standard deviation of the Gaussian
    Returns:
        heatmap (Tensor): shape [H, W]
    """
    H, W = shape
    x_range = torch.arange(W, dtype=torch.float32)
    y_range = torch.arange(H, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')

    cx = center[0] * (W - 1)
    cy = center[1] * (H - 1)

    heatmap = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    heatmap = heatmap / heatmap.max()
    return heatmap

def generate_target_heatmaps(coords, shape, sigma=2):
    """
    Generate 6 heatmaps from normalized coordinates.

    Args:
        coords (Tensor): shape [12] (x1, y1, ..., x6, y6), normalized [0-1]
        shape (tuple): (height, width) of output heatmaps
    Returns:
        heatmaps (Tensor): shape [6, H, W]
    """
    heatmaps = []
    for i in range(6):
        x = coords[i*2]
        y = coords[i*2 + 1]
        heatmap = generate_gaussian_heatmap((x, y), shape, sigma)
        heatmaps.append(heatmap)

    heatmaps = torch.stack(heatmaps, dim=0)  # shape [6, H, W]
    return heatmaps
