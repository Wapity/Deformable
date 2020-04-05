
import torch

RANDOM = True
NUMBER = 10  # Number of pairs = NUMBER**2


def dataset(random=RANDOM, batch_size):
    images = []
    data_pairs = []
    for i in range(NUMBER):
        if random:
            images.append(torch.normal(1, 0.5, size=(1, 64, 192, 192)))
            print("Image_{}_created".format(i))

    for i in range(NUMBER):
        for j in range(NUMBER):
            data_pairs.append((images[i], images[j]))

    return data_pairs


def build_affine_grid(size, device):
    B, _, H, W, D = size
    x, y, z = [torch.flatten(t) for t in [torch.meshgrid(
        *[torch.linspace(-1, 1, steps=s) for s in [H, W, D]])]]
    ones = torch.ones_like(x)
    grid = torch.stack([x, y, z, ones], dim=0)
    grid.resize_(B, 4, H * W * D)
    grid = grid.to(device)
    return grid


def affine_grid_3d(theta, grid, size):
    B, _, H, W, D = size
    return torch.transpose((theta@grid).reshape(B, 3, H, W, D), 1, 4)


def gradient_grid_3d(spatial_gds):
    _, _, H, W, D = spatial_gds.size()
    grids = [torch.cumsum(spatial_gds[:, i, ...], dim=i + 2).squeeze()
             for i in range(2)]
    return torch.stack([(grid / dim) * 2 - 1.
                        for dim, grid in zip([H, W, D], grids)], dim=4)
