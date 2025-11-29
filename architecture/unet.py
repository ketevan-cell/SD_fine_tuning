import torch
import torch.nn as nn

import math
from einops import repeat

from unet_blocks import ResBlock, SpTBlock

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
	if not repeat_only:
    	half = dim // 2
    	freqs = torch.exp(
        	-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    	).to(device=timesteps.device)
    	args = timesteps[:, None].float() * freqs[None]
    	embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    	if dim % 2:
        	embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
	else:
    	embedding = repeat(timesteps, 'b -> b d', d=dim)
	return embedding

class UNetModel(nn.Module):

	def __init__(
    	self,
    	image_size=32,
    	in_channels=4,
    	model_channels=320,
    	out_channels=4,
    	num_res_blocks=2,
    	attention_resolutions = (1, 2, 4),
    	dropout=0,
    	channel_mult=(1, 2, 4, 4),
    	dims=2,
    	num_head_channels=64,
    	transformer_depth=1,
    	context_dim=1024,
    	num_attention_blocks=None,
	):
    	super().__init__()

    	self.image_size = image_size
    	self.in_channels = in_channels
    	self.model_channels = model_channels
    	self.out_channels = out_channels
    	self.num_res_blocks = len(channel_mult) * [num_res_blocks]

    	self.attention_resolutions = attention_resolutions
    	self.dropout = dropout
    	self.channel_mult = channel_mult

    	self.num_head_channels = num_head_channels

    	time_embed_dim = model_channels * 4
    	self.time_embed = nn.Sequential(
        	nn.Linear(model_channels, time_embed_dim),
        	nn.SiLU(),
        	nn.Linear(time_embed_dim, time_embed_dim),
    	)

    	self.input_blocks = nn.ModuleList(
        	[
            	nn.Conv2d(dims, in_channels, model_channels, 3, padding=1)
        	]
    	)
    	self._feature_size = model_channels
    	input_block_chans = [model_channels]
    	ch = model_channels
    	ds = 1
    	for level, mult in enumerate(channel_mult):
        	for nr in range(self.num_res_blocks[level]):
            	layers = [
                	ResBlock(
                    	ch,
                    	time_embed_dim,
                    	dropout,
                    	out_channels=mult * model_channels,
                    	dims=dims,
                	)
            	]
            	ch = mult * model_channels
            	if ds in attention_resolutions:
                	if num_head_channels == -1:
                    	dim_head = ch // num_heads
                	else:
                    	num_heads = ch // num_head_channels
                    	dim_head = num_head_channels

                	layers.append(
                    	SpTBlock(
                        	ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                    	)
                	)

            	self.input_blocks.append(*layers)

            	self._feature_size += ch
            	input_block_chans.append(ch)
        	if level != len(channel_mult) - 1:
            	out_ch = ch
            	self.input_blocks.append(
                    	Downsample(
                        	ch, dims=dims, out_channels=out_ch
                    	)
            	)
            	ch = out_ch
            	input_block_chans.append(ch)
            	ds *= 2
            	self._feature_size += ch

    	self.middle_block = nn.Sequential(
        	ResBlock(
            	ch,
            	time_embed_dim,
            	dropout,
            	dims=dims,
        	),
        	SpTBlock(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
        	ResBlock(
            	ch,
            	time_embed_dim,
            	dropout,
            	dims=dims,
        	),
    	)
    	self._feature_size += ch

    	self.output_blocks = nn.ModuleList([])
    	for level, mult in list(enumerate(channel_mult))[::-1]:
        	for i in range(self.num_res_blocks[level] + 1):
            	ich = input_block_chans.pop()
            	layers = [
                	ResBlock(
                    	ch + ich,
                    	time_embed_dim,
                    	dropout,
                    	out_channels=model_channels * mult,
                	)
            	]
            	ch = model_channels * mult
            	if ds in attention_resolutions:
                	num_heads = ch // num_head_channels
                	dim_head = num_head_channels

                	if num_attention_blocks == None:
                    	layers.append(
                        	SpTBlock(
                            	ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                        	)
                    	)
            	if level and i == self.num_res_blocks[level]:
                	out_ch = ch
                	layers.append(
                    	Upsample(ch, dims=dims, out_channels=out_ch)
                	)
                	ds //= 2
            	self.output_blocks.append(*layers)
            	self._feature_size += ch

    	self.out = nn.Sequential(
        	nn.GroupNorm(num_groups=32, num_channels=ch),
        	nn.SiLU(),
        	nn.Conv2d(model_channels, out_channels, 3, padding=1),
    	)

	def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
    	hs = []
    	t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    	emb = self.time_embed(t_emb)

    	h = x
    	for module in self.input_blocks:
        	h = module(h, emb, context)
        	hs.append(h)
    	h = self.middle_block(h, emb, context)
    	for module in self.output_blocks:
        	h = th.cat([h, hs.pop()], dim=1)
        	h = module(h, emb, context)
    	return self.out(h)
