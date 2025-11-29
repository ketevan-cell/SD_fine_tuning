import torch
import numpy as np

from tqdm import tqdm
from PIL import Image
from einops import rearrange
from datasets import load_from_disk

# IM-SCORE Package: https://github.com/RE-N-Y/imscore
from imscore.mps.model import MPS
from imscore.hps.model import HPSv2
from imscore.pickscore.model import PickScorer
from imscore.imreward.model import ImageReward

hpsv2 = HPSv2.from_pretrained("RE-N-Y/hpsv21")
mps = MPS.from_pretrained("RE-N-Y/mpsv1")
pickscore = PickScorer("yuvalkirstain/PickScore_v1")
imreward = ImageReward.from_pretrained("RE-N-Y/ImageReward")

score_models = [hpsv2, mps, pickscore, imreward]

def convert_to_torch_tensor(pixels):
	pixels = np.array(pixels)
	pixels = rearrange(torch.tensor(pixels), "h w c -> 1 c h w") / 255.0
	return pixels

# Change prompt to the prompt of the images you’re testing, below is a placeholder
prompt = 'Pixel art of a cat napping on a windowsill with city lights in the background.'
image_1 = Image.open(image_path_1)
image_2 = Image.open(image_path_2)

ims = [image_1, image_2]
image_tensor = [convert_to_torch_tensor(im) for im in ims]
image_tensor = torch.cat(image_tensor, dim=0).to("cuda")
for score_model_i in tqdm(score_models):
	score_model_i.to("cuda").eval()
	with torch.inference_mode():
    	scores = score_model_i.score(image_tensor, [prompt]*len(ims))
	scores = scores.exp()/scores.exp().sum()
	print(scores)

# if scores[0] > scores[1] → image_1 is better
# if scores[0] < scores[1] → image_2 is better
