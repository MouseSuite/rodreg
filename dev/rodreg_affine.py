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


device = 'cuda'

moving_file = 'linwarped_aba3.nii.gz'#'/deneb_disk/RodentTools/data/MSA100/MSA100/MSA100.bfc.nii.gz'
target_file = 'dev/test_case/F2_BC.bfc.nii.gz'  # '
output_file = 'nonlin_warped_atlas1e-1.bfc.nii.gz'
label_file = 'linwarped_aba3.nii.gz'
output_label_file = 'linwarped_aba3-1.label.nii.gz'

moving_file = '/deneb_disk/RodentTools/data/MSA100/MSA100/MSA100.bfc.nii.gz' #'distorted_M2_LCRP.bfc.nii.gz'#
target_file = 'F2_BC.bfc.nii.gz'  # '
output_file = 'linwarped_atlas.bfc.nii.gz'
label_file = '/deneb_disk/RodentTools/data/MSA100/MSA100/MSA100.label.nii.gz'
output_label_file = 'linwarped_atlas.label.nii.gz'

# LocalNormalizedCrossCorrelationLoss() #GlobalMutualInformationLoss() # #
image_loss = MSELoss() #LocalNormalizedCrossCorrelationLoss() #MSELoss()
max_epochs = 3000
nn_input_size = 64


#######################
set_determinism(42)

moving, moving_meta = LoadImage()(moving_file)
target, moving_meta = LoadImage()(target_file)

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


#####################
