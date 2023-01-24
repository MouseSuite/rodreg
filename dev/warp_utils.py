# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from monai.config.deviceconfig import USE_COMPILED
from monai.networks.layers.spatial_transforms import grid_pull
from monai.networks.utils import meshgrid_ij
from torch.nn.functional import grid_sample


def get_grid(moving_shape, target_shape, requires_grad=False):

    mesh_points = [torch.arange(0, dim) for dim in target_shape]
    ref_grid = torch.stack(meshgrid_ij(*mesh_points), dim=0)  # (spatial_dims, ...)
    #grid = torch.stack([grid] * ddf.shape[0], dim=0)  # (batch, spatial_dims, ...)
    #ref_grid = grid.to(ddf)
    ref_grid = ref_grid.float()
    ref_grid[0] *= (moving_shape[0]/target_shape[0])
    ref_grid[1] *= (moving_shape[1]/target_shape[1])
    ref_grid[2] *= (moving_shape[2]/target_shape[2]) 
    ref_grid.requires_grad = False
    return ref_grid



def apply_warp(disp_field, moving_image, target_image, interp_mode='bilinear'):
    # disp_field size should be Bx3xWxHxD
    # moving_image and target_image size should be BxCxWxHxD
    # B=batch size, C=num channels, W=width, H=height, and D=depth

    ref_grid = get_grid(moving_image.shape[2:], target_image.shape[2:])

    grid = ref_grid.to(disp_field) + disp_field
    grid = torch.permute(grid, (0, 2, 3, 4, 1))

    for i, dim in enumerate(moving_image.shape[2:]):
        grid[..., i] = grid[..., i] * 2 / (dim - 1) - 1


    spatial_dims = 3
    index_ordering: List[int] = list(range(spatial_dims - 1, -1, -1))
    grid = grid[..., index_ordering]  # z, y, x -> x, y, z

    warped_image = grid_sample(moving_image.to(disp_field), grid=grid, align_corners=True, mode=interp_mode)

    return warped_image




