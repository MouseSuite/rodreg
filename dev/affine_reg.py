from monai.utils import set_determinism
from monai.networks.nets import GlobalNet
from monai.config import USE_COMPILED
from monai.networks.blocks import Warp
from torch.nn import MSELoss
from monai.transforms import LoadImage, Resize, EnsureChannelFirst, ScaleIntensityRangePercentiles
from monai.data.nifti_writer import write_nifti
from monai.losses import GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss
from warp_utils import apply_warp
import argparse
import torch


def affine_reg(fixed_file, moving_file, output_file, ddf_file, loss='mse', nn_input_size=64, lr=1e-6, max_epochs=5000, device='cuda'):

    set_determinism(42)

    if loss == 'mse':
        image_loss = MSELoss()
    elif loss == 'cc':
        image_loss = LocalNormalizedCrossCorrelationLoss()
    elif loss == 'mi':
        image_loss = GlobalMutualInformationLoss()
    else:
        AssertionError

    set_determinism(42)

    moving, moving_meta = LoadImage()(moving_file)
    target, moving_meta = LoadImage()(fixed_file)

    SZ = nn_input_size
    moving = EnsureChannelFirst()(moving)
    target = EnsureChannelFirst()(target)
    size_moving = moving[0].shape
    size_target = target[0].shape

    moving_ds = Resize(spatial_size=[SZ, SZ, SZ])(moving).to(device)
    target_ds = Resize(spatial_size=[SZ, SZ, SZ])(target).to(device)

    moving_ds = ScaleIntensityRangePercentiles(
        lower=0.5, upper=99.5, b_min=0.0, b_max=10, clip=True)(moving_ds)
    target_ds = ScaleIntensityRangePercentiles(
        lower=0.5, upper=99.5, b_min=0.0, b_max=10, clip=True)(target_ds)

    # GlobalNet is a NN with Affine head
    reg = GlobalNet(
        image_size=(SZ, SZ, SZ),
        spatial_dims=3,
        in_channels=2,  # moving and fixed
        num_channel_initial=2,
        depth=2).to(device)

    if USE_COMPILED:
        warp_layer = Warp(3, padding_mode="zeros").to(device)
    else:
        warp_layer = Warp("bilinear", padding_mode="zeros").to(device)

    reg.train()

    optimizerR = torch.optim.Adam(reg.parameters(), lr=1e-6)

    for epoch in range(max_epochs):

        optimizerR.zero_grad()

        input_data = torch.cat((moving_ds, target_ds), dim=0)
        input_data = input_data[None, ]
        ddf_ds = reg(input_data)
        image_moved = warp_layer(moving_ds[None, ], ddf_ds)

        vol_loss = image_loss(image_moved, target_ds[None, ])

        vol_loss.backward()
        optimizerR.step()

        print(f'epoch_loss:{vol_loss} for epoch:{epoch}')

    ddfx = Resize(spatial_size=size_target, mode='trilinear')(
        ddf_ds[:, 0])*(size_moving[0]/SZ)
    ddfy = Resize(spatial_size=size_target, mode='trilinear')(
        ddf_ds[:, 1])*(size_moving[1]/SZ)
    ddfz = Resize(spatial_size=size_target, mode='trilinear')(
        ddf_ds[:, 2])*(size_moving[2]/SZ)
    ddf = torch.cat((ddfx, ddfy, ddfz), dim=0)
    del ddf_ds, ddfx, ddfy, ddfz

    # Apply the warp
    image_movedo = apply_warp(ddf[None, ], moving[None, ], target[None, ])
    write_nifti(image_movedo[0, 0], output_file, affine=target.affine)
    write_nifti(torch.permute(ddf, [1, 2, 3, 0]),
                ddf_file, affine=target.affine)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Affine registration for mouse brains')

    parser.add_argument('moving-file', type=str,
                        help='moving file name')
    parser.add_argument('fixed-file', type=str, help='fixed file name')
    parser.add_argument('output-file', type=str,
                        help='output file name')
    parser.add_argument('-ddf', '--ddf-file', type=str,
                        help='dense displacement field file name')
    parser.add_argument('--nn_input_size', type=int, default=64,
                        help='size of the neural network input (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('-e', '--max-epochs', type=int,
                        default=1500, help='maximum interations')
    parser.add_argument('-d', '--device', type=str,
                        default='cuda', help='device: cuda, cpu, etc.')
    parser.add_argument('-l', '--loss', type=str, default='mse',
                        help='loss function: mse, cc or mi')

    args = parser.parse_args()

    affine_reg(fixed_file=args.fixed_file, moving_file=args.moving_file, output_file=args.output_file, ddf_file=args.ddf_file,
               loss=args.loss, nn_input_size=args.nn_input_size, lr=args.lr, max_epochs=args.max_epochs, device=args.device)

    '''
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

    '''



if __name__ == "__main__":
    main()