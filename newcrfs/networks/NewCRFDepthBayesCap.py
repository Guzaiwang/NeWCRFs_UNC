import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .swin_transformer import SwinTransformer
from .newcrf_layers import NewCRF
from .uper_crf_head import PSP
########################################################################################################################


class NewCRFDepthBayesCap(nn.Module):
    '''
    Depth network based on NewCRFs and BayesCap uncertainty estimation
    '''
    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()
        self.newcrf = NewCRFDepth(version=version, inv_depth=inv_depth, pretrained=pretrained, 
                    frozen_stages=frozen_stages, min_depth=min_depth, max_depth=max_depth, **kwargs)
        newcrf_ckpt = torch.load('/DATA/i2r/guzw/workspace/confidence/Uncertainty/NeWCRFs_UNC/model_zoo/model_nyu.ckpt', map_location='cpu')
        new_state_dict = {}
        for k, v in newcrf_ckpt['model'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix
            else:
                new_state_dict[k] = v
        print("loaded newcrf_ckpt")
        self.newcrf.load_state_dict(new_state_dict)
        self.bayescap = BayesCap(in_channels=1, out_channels=1)
        # Freeze the parameters of self.newcrf
        for param in self.newcrf.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            depth = self.newcrf(x)
        normalized_depth = depth / 10.
        uncertainty = self.bayescap(normalized_depth.clone())
        return normalized_depth, uncertainty


class NewCRFDepthBayesCapKITTI(nn.Module):
    '''
    Depth network based on NewCRFs and BayesCap uncertainty estimation
    '''
    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()
        self.newcrf = NewCRFDepth(version=version, inv_depth=inv_depth, pretrained=pretrained, 
                    frozen_stages=frozen_stages, min_depth=min_depth, max_depth=max_depth, **kwargs)
        newcrf_ckpt = torch.load('/DATA/i2r/guzw/workspace/confidence/Uncertainty/NeWCRFs_UNC/model_zoo/model_kittieigen.ckpt', map_location='cpu')
        new_state_dict = {}
        for k, v in newcrf_ckpt['model'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix
            else:
                new_state_dict[k] = v
        print("loaded newcrf_ckpt")
        self.newcrf.load_state_dict(new_state_dict)
        self.bayescap = BayesCap(in_channels=1, out_channels=1)
        # Freeze the parameters of self.newcrf
        for param in self.newcrf.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            depth = self.newcrf(x)
        normalized_depth = depth / 80.
        uncertainty = self.bayescap(normalized_depth.clone())
        return normalized_depth, uncertainty



class NewCRFDepth(nn.Module):
    """
    Depth network based on neural window FC-CRFs architecture.
    """
    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False

        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        backbone_cfg = dict(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            frozen_stages=frozen_stages
        )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        self.decoder = PSP(**decoder_cfg)
        self.disp_head1 = DispHead(input_dim=crf_dims[0])

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(crf_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))

        self.min_depth = min_depth
        self.max_depth = max_depth

        self.init_weights(pretrained=pretrained)



    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def upsample_mask(self, disp, mask):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, 4*H, 4*W)

    def forward(self, imgs):

        feats = self.backbone(imgs)
        if self.with_neck:
            feats = self.neck(feats)

        ppm_out = self.decoder(feats)

        e3 = self.crf3(feats[3], ppm_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(feats[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(feats[1], e2)
        e1 = nn.PixelShuffle(2)(e1)
        e0 = self.crf0(feats[0], e1)

        if self.up_mode == 'mask':
            mask = self.mask_head(e0)
            d1 = self.disp_head1(e0, 1)
            d1 = self.upsample_mask(d1, mask)
        else:
            d1 = self.disp_head1(e0, 4)

        depth = d1 * self.max_depth

        return depth


class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


class DispUnpack(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128):
        super(DispUnpack, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 16, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x, output_size):
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x)) # [b, 16, h/4, w/4]
        # x = torch.reshape(x, [x.shape[0], 1, x.shape[2]*4, x.shape[3]*4])
        x = self.pixel_shuffle(x)

        return x


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

class ResidualConvBlock(nn.Module):
	"""Implements residual conv function.

	Args:
		channels (int): Number of channels in the input image.
	"""

	def __init__(self, channels: int) -> None:
		super(ResidualConvBlock, self).__init__()
		self.rcb = nn.Sequential(
			nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(channels),
			nn.PReLU(),
			nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(channels),
		)

	def forward(self, x: Tensor) -> Tensor:
		identity = x

		out = self.rcb(x)
		out = torch.add(out, identity)

		return out



#### BayesCap
class BayesCap(nn.Module):
	def __init__(self, in_channels=3, out_channels=3) -> None:
		super(BayesCap, self).__init__()
		# First conv layer.
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(
				in_channels, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
		)

		# Features trunk blocks.
		trunk = []
		for _ in range(16):
			trunk.append(ResidualConvBlock(64))
		self.trunk = nn.Sequential(*trunk)

		# Second conv layer.
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=3, stride=1, padding=1, bias=False
			),
			nn.BatchNorm2d(64),
		)

		# Output layer.
		self.conv_block3_mu = nn.Conv2d(
			64, out_channels=out_channels, 
			kernel_size=9, stride=1, padding=4
		)
		self.conv_block3_alpha = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 1, 
				kernel_size=9, stride=1, padding=4
			),
			nn.ReLU(),
		)
		self.conv_block3_beta = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 1, 
				kernel_size=9, stride=1, padding=4
			),
			nn.ReLU(),
		)

		# Initialize neural network weights.
		self._initialize_weights()

	def forward(self, x: Tensor) -> Tensor:
		return self._forward_impl(x)

	# Support torch.script function.
	def _forward_impl(self, x: Tensor) -> Tensor:
		out1 = self.conv_block1(x)
		out = self.trunk(out1)
		out2 = self.conv_block2(out)
		out = out1 + out2
		out_mu = self.conv_block3_mu(out)
		out_alpha = self.conv_block3_alpha(out)
		out_beta = self.conv_block3_beta(out)
		return out_mu, out_alpha, out_beta

	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)

