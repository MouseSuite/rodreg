from monai.utils import set_determinism
from monai.networks.nets import GlobalNet
from monai.config import USE_COMPILED
from monai.networks.blocks import Warp
import torch
from torch.nn import MSELoss
from monai.transforms import LoadImage, Resize, EnsureChannelFirst, ScaleIntensityRangePercentiles
from monai.data.nifti_writer import write_nifti
from monai.losses.ssim_loss import SSIMLoss
from monai.losses import GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss
from nilearn.image import resample_to_img, resample_img, crop_img, load_img
from torch.nn.functional import grid_sample
from warp_utils import get_grid, apply_warp
from typing import List
import argparse







device = 'cuda'
moving_file = '/deneb_disk/RodentTools/data/MSA100/MSA100/MSA100.bfc.nii.gz'
fixed_file = 'F2_BC.bfc.nii.gz'  # '
output_file = 'warped_atlas.bfc.nii.gz'
label_file = '/deneb_disk/RodentTools/data/MSA100/MSA100/MSA100.label.nii.gz'
output_label_file = 'warped_atlas.label.nii.gz'
ddf_file = 'ddf.nii.gz'

# LocalNormalizedCrossCorrelationLoss() #GlobalMutualInformationLoss() # #

max_epochs = 5000
nn_input_size = 64
lr = 1e-6

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Affine registration for mouse brains')

    parser.add_argument('-m', '--moving-file', type=str, help='moving file name')
    parser.add_argument('-f', '--fixed-file', type=str, help='fixed file name')
    parser.add_argument('-o', '--output-file', type=str, help='output file name')
    parser.add_argument('-d', '--ddf-file', type=str, help='dense displacement field file name')
    parser.add_argument('--nn_input_size', type=int, default=64, help='size of the neural network input (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate (default: 1e-4)')
    parser.add_argument('-e', '--max-epochs', type=int, default=1500, help='maximum interations')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device: cuda, cpu, etc.')
    parser.add_argument('-l', '--loss', type=str, default='mse', help='loss function: mse, cc or mi')

    args = parser.parse_args()
 


























#######################

def affine_reg(fixed_file, moving_file, output_file, ddf_file, loss='cc', nn_input_size=64, lr=1e-6, max_epochs=5000, devide='cuda'):

    set_determinism(42)


       
    if loss == 'mse':
        image_loss = MSELoss()    
    elif loss == 'cc':
        image_loss = LocalNormalizedCrossCorrelationLoss()
    elif loss =='mi':
        image_loss = GlobalMutualInformationLoss()



    moving, moving_meta = LoadImage()(moving_file)
    fixed, moving_meta = LoadImage()(fixed_file)

    SZ = nn_input_size
    moving = EnsureChannelFirst()(moving)
    fixed = EnsureChannelFirst()(fixed)
    size_moving = moving[0].shape
    size_fixed = fixed[0].shape


    moving_ds = Resize(spatial_size=[SZ, SZ, SZ])(moving).to(device)
    fixed_ds = Resize(spatial_size=[SZ, SZ, SZ])(fixed).to(device)

    moving_ds = ScaleIntensityRangePercentiles(
        lower=0.5, upper=99.5, b_min=0.0, b_max=10, clip=True)(moving_ds)
    fixed_ds = ScaleIntensityRangePercentiles(
        lower=0.5, upper=99.5, b_min=0.0, b_max=10, clip=True)(fixed_ds)


    # GlobalNet is a NN with Affine head
    reg = GlobalNet(
        image_size=(SZ, SZ, SZ),
        spatial_dims=3,
        in_channels=2,  # moving and fixed
        num_channel_initial=4,
        depth=4).to(device)

    if USE_COMPILED:
        warp_layer = Warp(3, padding_mode="zeros").to(device)
    else:
        warp_layer = Warp("bilinear", padding_mode="zeros").to(device)

    reg.train()


    optimizerR = torch.optim.Adam(reg.parameters(), lr=lr)

    for epoch in range(max_epochs):

        optimizerR.zero_grad()

        input_data = torch.cat((moving_ds, fixed_ds), dim=0)
        input_data = input_data[None, ]
        ddf_ds = reg(input_data)
        image_moved = warp_layer(moving_ds[None, ], ddf_ds)

        vol_loss = image_loss(image_moved, fixed_ds[None, ])

        vol_loss.backward()
        optimizerR.step()

        print(f'epoch_loss:{vol_loss} for epoch:{epoch}')



    ddfx = Resize(spatial_size=size_fixed, mode='trilinear')(ddf_ds[:, 0])*(size_moving[0]/SZ)
    ddfy = Resize(spatial_size=size_fixed, mode='trilinear')(ddf_ds[:, 1])*(size_moving[1]/SZ)
    ddfz = Resize(spatial_size=size_fixed, mode='trilinear')(ddf_ds[:, 2])*(size_moving[2]/SZ)
    ddf = torch.cat((ddfx, ddfy, ddfz), dim=0)
    del ddf_ds, ddfx, ddfy, ddfz

    # Apply the warp
    image_movedo = apply_warp(ddf[None, ], moving[None, ], fixed[None, ])
    write_nifti(image_movedo[0, 0], output_file, affine=fixed.affine)

    write_nifti(torch.permute(ddf,[1,2,3,0]), ddf_file, affine=fixed.affine)



# Apply the warp to labels
label, meta = LoadImage()(label_file)
label = EnsureChannelFirst()(label)
warped_labels = apply_warp(ddf[None, ], label[None,], fixed[None, ], interp_mode='nearest')
write_nifti(warped_labels[0,0], output_label_file, affine=fixed.affine)


#####################
