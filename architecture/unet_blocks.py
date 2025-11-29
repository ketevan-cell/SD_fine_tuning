import torch
from einops import rearrange, repeat

class ResBlock(torch.nn.module):
	def __init__(
    	self,
    	channels,
    	emb_channels,
    	dropout,
    	out_channels=None,

	):
    	super().__init__()
    	self.channels = channels
    	self.emb_channels = emb_channels
    	self.dropout = dropout
    	self.out_channels = out_channels or channels

    	self.in_layers = torch.nn.Sequential(
        	torch.nn.GroupNorm(num_groups=32, num_channels=channels),
        	torch.nn.SiLU(),
        	torch.nn.Conv2d(channels, self.out_channels, kernel_size=3, padding=1),
    	)

    	self.emb_layers = torch.nn.Sequential(
        	torch.nn.SiLU(),
        	torch.nn.Linear(emb_channels, self.out_channels)
        	)
    	self.out_layers = torch.nn.Sequential(
        	torch.nn.GroupNorm(num_groups=32, num_channels=self.out_channels),
        	torch.nn.SiLU(),
        	torch.nn.Dropout(p=dropout),
            	torch.nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
    	)

    	if self.out_channels == channels:
        	self.skip_connection = torch.nn.Identity()
    	else:
        	self.skip_connection = torch.nn.Conv2d(channels, self.out_channels, kernel_size=1)

	def forward(self, x, emb):
    	h = self.in_layers(x)
    	# Incorporate time-embedding
    	# UNet predicts noise at every timestep
    	# It's important to make it aware at which timestep it is now
    	emb_out = self.emb_layers(emb)
    	h = h + emb_out
    	h = self.out_layers(h)
    	return self.skip_connection(x) + h

class CrossAttention(torch.nn.Module):
	def __init__(self, query_dim, textual_feature_dim=None, heads=8, dim_head=64, dropout=0.):
    	super().__init__()
    	inner_dim = dim_head * heads

    	if textual_feature_dim == None:
        	textual_feature_dim = query_dim

    	self.scale = dim_head ** -0.5
    	self.heads = heads

    	self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
    	self.to_k = torch.nn.Linear(textual_feature_dim, inner_dim, bias=False)
    	self.to_v = torch.nn.Linear(textual_feature_dim, inner_dim, bias=False)

    	self.to_out = torch.nn.Sequential(
        	torch.nn.Linear(inner_dim, query_dim),
        	torch.nn.Dropout(dropout)
    	)

	def forward(self, x, textual_feature=None):
    	# Transform input/textual_feature to Q, K, V
    	q = self.to_q(x)
    	# Self attention if textual_feature is not passed
    	if textual_feature == None:
        	textual_feature = x
    
    	k = self.to_k(textual_feature)
    	v = self.to_v(textual_feature)
   	 
    	q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
    	# Perform multi-head self/cross attetnion
    	# Attention = scores * V = Softmax(QK^T)/sqrt(head_dim)
    	# depending on whether textual_feature is given or not
    	sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
    	sim = sim.softmax(dim=-1)
    	# Multiply the attention scores by V
    	out = torch.einsum('b i j, b j d -> b i d', sim, v)
    	out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)

    	return self.to_out(out)

class FeedForward(torch.nn.Module):
	def __init__(self, dim, dim_out=None, mult=4, dropout=0.):
    	super().__init__()
    	inner_dim = int(dim * mult)
    	if dim_out == None:
        	dim_out = dim

    	self.net = torch.nn.Sequential(
        	torch.nn.Linear(dim, inner_dim),
        	torch.nn.GELU(),
        	torch.nn.Dropout(dropout),
        	torch.nn.Linear(inner_dim, dim_out)
    	)

	def forward(self, x):
    	return self.net(x)

class SpTBlock(torch.nn.Module):
   
	def __init__(self, in_channels, n_heads, d_head, dropout=0., context_dim=None):
    	super().__init__()

    	self.in_channels = in_channels
    	inner_dim = n_heads * d_head
    	self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels)
    	self.proj_in = torch.nn.Linear(in_channels, inner_dim)

    	self.attn1 = CrossAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, context_dim=None)
    	self.ff = FeedForward(inner_dim, dropout=dropout)
    	self.attn2 = CrossAttention(query_dim=inner_dim, context_dim=context_dim, heads=n_heads, dim_head=d_head)
    	self.norm1 = torch.nn.LayerNorm(inner_dim)
    	self.norm2 = torch.nn.LayerNorm(inner_dim)
    	self.norm3 = torch.nn.LayerNorm(inner_dim)   	 
   	 
    	self.proj_out = torch.nn.Linear(in_channels, inner_dim)

	def forward(self, x, textual_feature=None):
    	batch, channel, height, width = x.shape
    	x_in = x
    	x = self.norm(x)
    	x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
    	x = self.proj_in(x)
    	# Perform self attention first followed by cross attetnion.
    	# Pass textual_feature only to cross attention
    	x = self.attn1(self.norm1(x), textual_feature=None) + x
    	x = self.attn2(self.norm2(x), textual_feature=textual_feature) + x
    	x = self.ff(self.norm3(x)) + x

    	x = self.proj_out(x)
    	x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width).contiguous()
    	return x + x_in

