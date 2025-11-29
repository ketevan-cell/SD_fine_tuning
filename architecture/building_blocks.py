import torch

class AttnBlock(torch.nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.in_channels = in_channels

		self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels)

		self.q = torch.nn.Conv2d(in_channels,
								in_channels,
								kernel_size=1,
								stride=1,
								padding=0)
		self.k = torch.nn.Conv2d(in_channels,
								in_channels,
								kernel_size=1,
								stride=1,
								padding=0)
		self.v = torch.nn.Conv2d(in_channels,
								in_channels,
								kernel_size=1,
								stride=1,
								padding=0)
		self.proj_out = torch.nn.Conv2d(in_channels,
										in_channels,
										kernel_size=1,
										stride=1,
										padding=0)

	def forward(self, x):
    	h = x
    	h = self.norm(h)

    	# Attention = Softmax(QK^T/sqrt(c)) * V
    	q = self.q(h)
    	k = self.k(h)
    	v = self.v(h)

    	batch, channel, height, width = q.shape
    	# Reshape q, k, v so that it is (batch, channel, sequence length)
    	q = q.reshape(batch, channel, height*width)
    	q = q.permute(0,2,1)
    	k = k.reshape(batch, channel, height*width)
    	# Compute attention scores Softmax(QK^T/sqrt(c))
    	w = torch.bmm(q,k)
    	w = w * (int(channel)**(-0.5))
    	w = torch.nn.functional.softmax(w, dim=2)
    	# Apply attention scores on V
    	v = v.reshape(batch, channel, height*width)
    	w = w.permute(0,2,1)
    	h = torch.bmm(v,w)
    	h = h.reshape(batch, channel, height, width)

    	h = self.proj_out(h)

    	return x+h
    
class ResnetBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, dropout):
    	super().__init__()
    	self.in_channels = in_channels
    	out_channels = in_channels if out_channels is None else out_channels
    	self.out_channels = out_channels

    	self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels)
    	self.conv1 = torch.nn.Conv2d(in_channels,
                                 	out_channels,
                                 	kernel_size=3,
                                 	stride=1,
                                 	padding=1)

    	self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels)
    	self.dropout = torch.nn.Dropout(dropout)
    	self.conv2 = torch.nn.Conv2d(out_channels,
                                 	out_channels,
                                 	kernel_size=3,
                                 	stride=1,
                                 	padding=1)
    	if self.in_channels != self.out_channels:
        	self.shortcut_proj = torch.nn.Conv2d(in_channels,
                                            	out_channels,
                                            	kernel_size=1,
                                            	stride=1,
                                            	padding=0)
    	self.act1 = torch.nn.SiLU()
    	self.act2 = torch.nn.SiLU()

	def forward(self, x):
    	# Make a copy of x for residual connection
    	h = x
    	h = self.norm1(h)
    	h = self.act1(h)
    	h = self.conv1(h)

    	h = self.norm2(h)
    	h = self.act2(h)
    	h = self.dropout(h)
    	h = self.conv2(h)
   	 
    	# Project the residual connection to have the same channel size as output
    	if self.in_channels != self.out_channels:
        	x = self.shortcut_proj(x)

    	return x+h
    
class Downsample(torch.nn.Module):
	def __init__(self):
    	super().__init__()

	def forward(self, x):
    	# Average neighboring features with a strider of 2  
    	# This reduces the spatial resolution by 2 on every axis
    	x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    	return x
    
class Upsample(torch.nn.Module):
	def __init__(self):
    	super().__init__()

	def forward(self, x):
    	# Interpolate neighboring features by a scale factor of 2  
    	# This increases the spatial resolution by 2 on every axis
    	x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
    	return x
