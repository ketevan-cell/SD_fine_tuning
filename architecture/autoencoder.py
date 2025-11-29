import torch
import torch.nn as nn

import numpy as np

from building_blocks import ResBlock, AttnBlock

class DiagonalGaussianDistribution(object):
	def __init__(self, parameters, deterministic=False):
    	self.parameters = parameters
    	self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
    	self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    	self.deterministic = deterministic
    	self.std = torch.exp(0.5 * self.logvar)
    	self.var = torch.exp(self.logvar)
    	if self.deterministic:
        	self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

	def sample(self):
    	x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
    	return x

	def kl(self, other=None):
    	if self.deterministic:
        	return torch.Tensor([0.])
    	else:
        	if other is None:
            	return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                   	+ self.var - 1.0 - self.logvar,
                                   	dim=[1, 2, 3])
        	else:
            	return 0.5 * torch.sum(
                	torch.pow(self.mean - other.mean, 2) / other.var
                	+ self.var / other.var - 1.0 - self.logvar + other.logvar,
                	dim=[1, 2, 3])

	def nll(self, sample, dims=[1,2,3]):
    	if self.deterministic:
        	return torch.Tensor([0.])
    	logtwopi = np.log(2.0 * np.pi)
    	return 0.5 * torch.sum(
        	logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
        	dim=dims)

	def mode(self):
    	return self.mean

class Downsample(nn.Module):
	def __init__(self):
    	super().__init__()

	def forward(self, x):
    	x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    	return x

class Encoder(nn.Module):
	def __init__(self, ch, out_ch, ch_mult=(1,2,4,4), num_res_blocks=2,
             	attn_resolutions=(4,2,1), dropout=0.0, in_channels=3,
             	resolution=256, z_channels=4, double_z=True):
    	super().__init__()

    	self.ch = ch
    	self.temb_ch = 0
    	self.num_resolutions = len(ch_mult)
    	self.num_res_blocks = num_res_blocks
    	self.resolution = resolution
    	self.in_channels = in_channels

    	# downsampling
    	self.conv_in = torch.nn.Conv2d(in_channels,
                                   	self.ch,
                                   	kernel_size=3,
                                   	stride=1,
                                   	padding=1)

    	curr_res = resolution
    	in_ch_mult = (1,)+tuple(ch_mult)
    	self.in_ch_mult = in_ch_mult
    	self.down = nn.ModuleList()
    	for i_level in range(self.num_resolutions):
        	block = nn.ModuleList()
        	attn = nn.ModuleList()
        	block_in = ch*in_ch_mult[i_level]
        	block_out = ch*ch_mult[i_level]
        	for i_block in range(self.num_res_blocks):
            	block.append(ResBlock(in_channels=block_in,
                                     	out_channels=block_out,
                                     	temb_channels=self.temb_ch,
                                     	dropout=dropout))
            	block_in = block_out
            	if curr_res in attn_resolutions:
                	attn.append(AttnBlock(block_in))
        	down = nn.Module()
        	down.block = block
        	down.attn = attn
        	if i_level != self.num_resolutions-1:
            	down.downsample = Downsample()
            	curr_res = curr_res // 2
        	self.down.append(down)

    	# middle
    	self.mid = nn.Module()
    	self.mid.block_1 = ResBlock(in_channels=block_in,
                                   	out_channels=block_in,
                                   	temb_channels=self.temb_ch,
                                   	dropout=dropout)
    	self.mid.attn_1 = AttnBlock(block_in)
    	self.mid.block_2 = ResBlock(in_channels=block_in,
                                   	out_channels=block_in,
                                   	temb_channels=self.temb_ch,
                                   	dropout=dropout)

    	# end
    	self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in)
    	self.conv_out = torch.nn.Conv2d(block_in,
                                    	2*z_channels if double_z else z_channels,
                                    	kernel_size=3,
                                    	stride=1,
                                    	padding=1)
    	self.act_out = nn.SiLU()

	def forward(self, x):
    	# timestep embedding
    	temb = None

    	# downsampling
    	hs = [self.conv_in(x)]
    	for i_level in range(self.num_resolutions):
        	for i_block in range(self.num_res_blocks):
            	h = self.down[i_level].block[i_block](hs[-1], temb)
            	if len(self.down[i_level].attn) > 0:
                	h = self.down[i_level].attn[i_block](h)
            	hs.append(h)
        	if i_level != self.num_resolutions-1:
            	hs.append(self.down[i_level].downsample(hs[-1]))

    	# middle
    	h = hs[-1]
    	h = self.mid.block_1(h, temb)
    	h = self.mid.attn_1(h)
    	h = self.mid.block_2(h, temb)

    	# end
    	h = self.norm_out(h)
    	h = self.act_out(h)
    	h = self.conv_out(h)
    	return h

class Upsample(nn.Module):
	def __init__(self):
    	super().__init__()

	def forward(self, x):
    	x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
    	return x

class Decoder(nn.Module):
	def __init__(self, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks=2,
             	attn_resolutions=[], dropout=0.0, in_channels=4,
             	resolution=32, z_channels=4):
    	super().__init__()

    	self.ch = ch
    	self.temb_ch = 0
    	self.num_resolutions = len(ch_mult)
    	self.num_res_blocks = num_res_blocks
    	self.resolution = resolution
    	self.in_channels = in_channels

    	# compute in_ch_mult, block_in and curr_res at lowest res
    	in_ch_mult = (1,)+tuple(ch_mult)
    	block_in = ch*ch_mult[self.num_resolutions-1]
    	curr_res = resolution // 2**(self.num_resolutions-1)
    	self.z_shape = (1,z_channels,curr_res,curr_res)

    	# z to block_in
    	self.conv_in = torch.nn.Conv2d(z_channels,
                                   	block_in,
                                   	kernel_size=3,
                                   	stride=1,
                                   	padding=1)

    	# middle
    	self.mid = nn.Module()
    	self.mid.block_1 = ResBlock(in_channels=block_in,
                                   	out_channels=block_in,
                                   	temb_channels=self.temb_ch,
                                   	dropout=dropout)
    	self.mid.attn_1 = AttnBlock(block_in)
    	self.mid.block_2 = ResBlock(in_channels=block_in,
                                   	out_channels=block_in,
                                   	temb_channels=self.temb_ch,
                                   	dropout=dropout)

    	# upsampling
    	self.up = nn.ModuleList()
    	for i_level in reversed(range(self.num_resolutions)):
        	block = nn.ModuleList()
        	attn = nn.ModuleList()
        	block_out = ch*ch_mult[i_level]
        	for i_block in range(self.num_res_blocks+1):
            	block.append(ResBlock(in_channels=block_in,
                                     	out_channels=block_out,
                                     	temb_channels=self.temb_ch,
                                     	dropout=dropout))
            	block_in = block_out
            	if curr_res in attn_resolutions:
                	attn.append(AttnBlock(block_in))
        	up = nn.Module()
        	up.block = block
        	up.attn = attn
        	if i_level != 0:
            	up.upsample = Upsample()
            	curr_res = curr_res * 2
        	self.up.insert(0, up) # prepend to get consistent order

    	# end
    	self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in)
    	self.conv_out = torch.nn.Conv2d(block_in,
                                    	out_ch,
                                    	kernel_size=3,
                                    	stride=1,
                                    	padding=1)
    	self.act_out = nn.SiLU()

	def forward(self, z):
    	#assert z.shape[1:] == self.z_shape[1:]
    	self.last_z_shape = z.shape

    	# timestep embedding
    	temb = None

    	# z to block_in
    	h = self.conv_in(z)

    	# middle
    	h = self.mid.block_1(h, temb)
    	h = self.mid.attn_1(h)
    	h = self.mid.block_2(h, temb)

    	# upsampling
    	for i_level in reversed(range(self.num_resolutions)):
        	for i_block in range(self.num_res_blocks+1):
            	h = self.up[i_level].block[i_block](h, temb)
            	if len(self.up[i_level].attn) > 0:
                	h = self.up[i_level].attn[i_block](h)
        	if i_level != 0:
            	h = self.up[i_level].upsample(h)

    	# end
    	if self.give_pre_end:
        	return h

    	h = self.norm_out(h)
    	h = self.act_out(h)
    	h = self.conv_out(h)
    	if self.tanh_out:
        	h = torch.tanh(h)
    	return h

class AutoencoderKL(nn.Module):
	def __init__(self,
            	z_channels=4,
            	resolution=256,
            	in_channels=3,
            	out_ch=3,
            	ch=128,
            	ch_mult=(1,2,4,4),
            	num_res_blocks=2,
            	attn_resolutions=[],
            	dropout=0.0,
            	embed_dim=4):
   	 
    	super().__init__()
    	self.encoder = Encoder(ch=ch,
                           	out_ch=out_ch,
                           	ch_mult=ch_mult,
                           	num_res_blocks=num_res_blocks,
                           	attn_resolutions=attn_resolutions,
                           	in_channels=in_channels,
                           	resolution=resolution,
                           	z_channels=z_channels,
                           	double_z=True
                           	)
    	self.decoder = Decoder(ch=ch,
                           	out_ch=out_ch,
                           	ch_mult=ch_mult,
                           	num_res_blocks=num_res_blocks,
                           	attn_resolutions=attn_resolutions,
                           	in_channels=in_channels,
                           	resolution=resolution,
                           	z_channels=z_channels,
                           	double_z=True
                           	)
    	self.quant_conv = torch.nn.Conv2d(2*z_channels, 2*embed_dim, 1)
    	self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
    	self.embed_dim = embed_dim

	def encode(self, x):
    	h = self.encoder(x)
    	moments = self.quant_conv(h)
    	posterior = DiagonalGaussianDistribution(moments)
    	return posterior

	def decode(self, z):
    	z = self.post_quant_conv(z)
    	dec = self.decoder(z)
    	return dec

	def forward(self, input, sample_posterior=True):
    	posterior = self.encode(input)
    	if sample_posterior:
        	z = posterior.sample()
    	else:
        	z = posterior.mode()
    	dec = self.decode(z)
    	return dec, posterior
