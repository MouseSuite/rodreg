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

set_determinism(42)

device = 'cuda'
moving_file = 'distorted_M2_LCRP.bfc.nii.gz' # 'M2_LCRP.bfc.nii.gz' #
target_file = 'F2_BC.bfc.nii.gz' #'/home/ajoshi/MSA100/MSA100_bst.nii.gz'#
output_file = 'dreg_M2_LCRP.bfc.nii.gz'
image_loss = LocalNormalizedCrossCorrelationLoss() #MSELoss() #GlobalMutualInformationLoss() #
nn_input_size = 64
max_epochs = 5000


#######################
moving, moving_meta = LoadImage()(moving_file)
target, moving_meta = LoadImage()(target_file)

SZ = nn_input_size
movingo = EnsureChannelFirst()(moving)
targeto = EnsureChannelFirst()(target)
size_orig = movingo[0].shape

moving = Resize(spatial_size=[SZ, SZ, SZ])(movingo).to(device)
target = Resize(spatial_size=[SZ, SZ, SZ])(targeto).to(device)

moving = ScaleIntensityRangePercentiles(
    lower=2, upper=98, b_min=0.0, b_max=10, clip=True)(moving)
target = ScaleIntensityRangePercentiles(
    lower=2, upper=98, b_min=0.0, b_max=10, clip=True)(target)


# GlobalNet is a NN with Affine head
reg = GlobalNet(
    image_size=(SZ, SZ, SZ),
    spatial_dims=3,
    in_channels=2,  # moving and fixed
    num_channel_initial=16,
    depth=4).to(device)

if USE_COMPILED:
    warp_layer = Warp(3, padding_mode="zeros").to(device)
else:
    warp_layer = Warp("bilinear", padding_mode="zeros").to(device)

reg.train()


optimizerR = torch.optim.Adam(reg.parameters(), lr=1e-6)

for epoch in range(max_epochs):

    optimizerR.zero_grad()

    input_data = torch.cat((moving, target), dim=0)
    input_data = input_data[None, ]
    ddf = reg(input_data)
    image_moved = warp_layer(moving[None, ], ddf)

    vol_loss = image_loss(image_moved, target[None,])

    vol_loss.backward()
    optimizerR.step()

    print(f'epoch_loss:{vol_loss} for epoch:{epoch}')


ddfx = Resize(spatial_size=size_orig)(ddf[:, 0])*(size_orig[0]/SZ)
ddfy = Resize(spatial_size=size_orig)(ddf[:, 1])*(size_orig[1]/SZ)
ddfz = Resize(spatial_size=size_orig)(ddf[:, 2])*(size_orig[2]/SZ)
ddfo = torch.cat((ddfx, ddfy, ddfz), dim=0)
del ddf, ddfx, ddfy, ddfz
image_movedo = warp_layer(movingo[None, ].to(device), ddfo[None, ])

write_nifti(image_movedo[0, 0], output_file, affine=targeto.affine)

#####################

