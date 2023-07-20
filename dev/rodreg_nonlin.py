from monai.utils import set_determinism
from monai.networks.nets import GlobalNet, LocalNet, RegUNet, unet
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
from warp_utils import get_grid, apply_warp, jacobian_determinant
from typing import List
from monai.losses import BendingEnergyLoss
from deform_losses import BendingEnergyLoss as myBendingEnergyLoss
from networks import LocalNet2
from tqdm import tqdm

device = 'cuda'

moving_file = '/home/ajoshi/projects/rodreg/linwarped_aba3.nii.gz'#'/deneb_disk/RodentTools/data/MSA100/MSA100/MSA100.bfc.nii.gz'
target_file = 'test_case/F2_BC.bfc.nii.gz'  # '
output_file = 'nonlin_warped_atlas1e-2.bfc.nii.gz'
label_file = 'linwarped_aba3.nii.gz'
output_label_file = 'linwarped_aba3-2.label.nii.gz'

# LocalNormalizedCrossCorrelationLoss() ##
image_loss = LocalNormalizedCrossCorrelationLoss()# MSELoss() #GlobalMutualInformationLoss() #  #LocalNormalizedCrossCorrelationLoss() #MSELoss()# 
regularization = myBendingEnergyLoss()

max_epochs = 1200
nn_input_size = 64

reg_penalty = 1
lr = .01

#######################
set_determinism(42)

moving, moving_meta = LoadImage()(moving_file)
target, moving_meta = LoadImage()(target_file)

SZ = nn_input_size
moving = EnsureChannelFirst()(moving)
target = EnsureChannelFirst()(target)
size_moving = moving[0].shape
size_target = target[0].shape


moving_ds = Resize(spatial_size=[SZ, SZ, SZ],mode='trilinear')(moving).to(device)
target_ds = Resize(spatial_size=[SZ, SZ, SZ],mode='trilinear')(target).to(device)

moving_ds = ScaleIntensityRangePercentiles(
    lower=0.5, upper=99.5, b_min=0.0, b_max=10, clip=True)(moving_ds)
target_ds = ScaleIntensityRangePercentiles(
    lower=0.5, upper=99.5, b_min=0.0, b_max=10, clip=True)(target_ds)



reg = unet.UNet(spatial_dims=3,  # spatial dims
    in_channels=2,
    out_channels=3,# output channels (to represent 3D displacement vector field)
    channels=(16, 32, 32, 32, 32),  # channel sequence
    strides=(1, 2, 2, 4),  # convolutional strides
    dropout=0.2,
    norm="batch").to(device)

if USE_COMPILED:
    warp_layer = Warp(3, padding_mode="zeros").to(device)
else:
    warp_layer = Warp("bilinear", padding_mode="zeros").to(device)

reg.train()


optimizerR = torch.optim.Adam(reg.parameters(), lr=lr)

for epoch in tqdm(range(max_epochs)):

    optimizerR.zero_grad()

    input_data = torch.cat((moving_ds, target_ds), dim=0)
    input_data = input_data[None, ]
    ddf_ds = reg(input_data)
    image_moved = warp_layer(moving_ds[None, ], ddf_ds)

    imgloss = image_loss(image_moved, target_ds[None, ])
    regloss = reg_penalty * regularization(ddf_ds)

    vol_loss =  imgloss + regloss

    #print(f'imgloss:{imgloss},   regloss:{regloss}')

    vol_loss.backward()
    optimizerR.step()

    #print(f'epoch_loss:{vol_loss} for epoch:{epoch}')



write_nifti(image_moved[0, 0], 'moved_ds.nii.gz', affine=target_ds.affine)
write_nifti(target_ds[0], 'target_ds.nii.gz', affine=target_ds.affine)
write_nifti(moving_ds[0], 'moving_ds.nii.gz', affine=target_ds.affine)
write_nifti(torch.permute(ddf_ds[0],[1,2,3,0]),'ddf_ds.nii.gz',affine=target_ds.affine)
jdet_ds = jacobian_determinant(ddf_ds[0])
write_nifti(jdet_ds,'jdet_ds.nii.gz',affine=target_ds.affine)



ddfx = Resize(spatial_size=size_target, mode='trilinear')(ddf_ds[:, 0])*(size_moving[0]/SZ)
ddfy = Resize(spatial_size=size_target, mode='trilinear')(ddf_ds[:, 1])*(size_moving[1]/SZ)
ddfz = Resize(spatial_size=size_target, mode='trilinear')(ddf_ds[:, 2])*(size_moving[2]/SZ)
ddf = torch.cat((ddfx, ddfy, ddfz), dim=0)
del ddf_ds, ddfx, ddfy, ddfz

# Apply the warp
image_movedo = apply_warp(ddf[None, ], moving[None, ], target[None, ])
write_nifti(image_movedo[0, 0], output_file, affine=target.affine)
write_nifti(torch.permute(ddf,[1,2,3,0]),'ddf.nii.gz',affine=target.affine)


# Apply the warp to labels
label, meta = LoadImage()(label_file)
label = EnsureChannelFirst()(label)
warped_labels = apply_warp(ddf[None, ], label[None,], target[None, ], interp_mode='nearest')
write_nifti(warped_labels[0,0], output_label_file, affine=target.affine)
jdet = jacobian_determinant(ddf)
write_nifti(jdet,'jdet.nii.gz',affine=target.affine)


#####################
