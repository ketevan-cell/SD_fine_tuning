# Stable Diffusion text-to-image fine-tuning

## Architecture

The code for Stable-Diffusion architecture including all components (AutoEncoder, UNet, and text encoder) can be found in and instantiated from architecture folder.

## Running code locally with PyTorch
The `training/train.py` script shows how to fine-tune stable diffusion model on [data-is-better-together/open-image-preferences-v1-binarized](https://huggingface.co/datasets/data-is-better-together/open-image-preferences-v1-binarized)

### Installing the dependencies

Before running the scripts, make sure to install the code dependencies:

Execute the following steps in a new virtual environment:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Then in a different directory, clone my GitHub repository:
```bash
git clone https://github.com/ketevan-cell/SD_fine_tuning.git
```

## Instructions on How to Run the code

In my thesis I used Stable Diffusion 2.1 [its card](https://huggingface.co/stabilityai/stable-diffusion-2-1).


### Hardware
Since I'm using free tier Google Collab, I get access to a NVIDIA T4 GPU with 16GB VRAM. It's important to use all the training tricks that save GPU memory. To that end, I enable `gradient_checkpointing`, `mixed_precision`, and `8BitAdamW` optimizer. Further memory savings can be obtained with using LoRA adapters on UNet component of Stable Diffusion, but I resorted to full-finetuning of UNet since the dataset is relatively small.

### Reproducing Thesis model
The exact command to reproduce my model is the following:

```bash
cd training

python main.py \
--pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1" \
--dataset_name "data-is-better-together/open-image-preferences-v1-binarized" \
--train_batch_size 1 \
--gradient_accumulation_steps 64 \
--gradient_checkpointing \
--max_train_steps 250 \
--learning_rate 1e-5 \
--lr_scheduler="constant" \
--lr_warmup_steps 0 \
--output_dir "sd-finetune" \
--image_column "chosen" \
--caption_column "prompt" \
--cache_dir "cache" \
```

Once the training is finished the model will be saved in the `sd-finetune` specified in the command. Checkpoints only save the unet, so to run inference from a checkpoint, just load the unet:

## Running Sample Inference and Evaluation

### Sample Inference with Base and fine-tuned model
To load the fine-tuned model for inference and evaluation you can follow the example in `SD_inference.ipynb`

```python
import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline, DPMSolverMultistepScheduler
# Load the stable diffusion pipeline, autoecoder, textual encoder, and UNet
pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, cache_dir="./cache"
    ).to("cuda")

# noise scheduler according to which we will remove the predicted noise from the noisy latent
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to('cuda')
pipe.safety_checker = None

# this is what defines the noisy latent from which the UNet will start from
generator = torch.Generator(device='cuda')
generator = generator.manual_seed(0)

prompt = "A fox in an autumn forest."
# guidance scale is how much we want the prompt to guide the denoising process
image_base = pipe(prompt=prompt, generator=generator, guidance_scale=5).images[0]
image_base

finetuned_unet = UNet2DConditionModel.from_pretrained(
                            "KetevanK/SD_fine_tuning",
                            subfolder='unet',
                            # access token since my model is in private repository
                            token="provided_secret_hf_token",
                            torch_dtype=torch.float16,
                            cache_dir="./cache").to('cuda')

# Swapping base model UNet with my finetuned UNet
pipe.unet = finetuned_unet
# The generator seed should be fixed for both models. This ensures both UNets
# start from the same noisy latent vector
generator = torch.Generator(device='cuda')
generator = generator.manual_seed(0)

image_mine = pipe(prompt=prompt, generator=generator, guidance_scale=5).images[0]
image_mine
```

### Sample Evaluation

After generating sample from the base model and my fine-tuned model for the same chosen prompt and same generator seed, you can run evaluation using one of the aesthetic scoring models provided in imscore library.

Note that the results in my thesis are reported on 100 different prompts spanning the target genres in my fine-tuning dataset. I report results using HPSv2, MPS, PickScore, and ImageReward scoring models. Sample prompts and comparsion between base and fine-tuned model are incldued in the thesis document.

```python
import numpy as np
from einops import rearrange
# imscore (https://github.com/RE-N-Y/imscore)
# library offers a collection of aesthetic scoring models and was used in
# my quantitative evaluation
from imscore.hps.model import HPSv2
from imscore.mps.model import MPS
from imscore.pickscore.model import PickScorer
from imscore.imreward.model import ImageReward
# Since the aesthetic scorer is a fine-tuned CLIP model itself,
# loading the model through the library will download the necessary files and
# evaluate the images. Downloading the files might take a bit of time

# Uncomment one of the below lines for aesthetic scoring
scorer = HPSv2.from_pretrained("RE-N-Y/hpsv21")
# scorer = MPS.from_pretrained("RE-N-Y/mpsv1")
# scorer = ImageReward.from_pretrained("RE-N-Y/ImageReward")
# scorer = PickScorer("yuvalkirstain/PickScore_v1")

def convert_to_torch_tensor(pixels):
	pixels = np.array(pixels)
	pixels = rearrange(torch.tensor(pixels), "h w c -> 1 c h w") / 255.0
	return pixels

ims = [image_base, image_mine]
image_tensor = [convert_to_torch_tensor(im) for im in ims]
image_tensor = torch.cat(image_tensor, dim=0).to("cuda")
scorer.to("cuda").eval()

with torch.inference_mode():
	scores = scorer.score(image_tensor, [prompt]*len(ims))
scores = scores.exp()/scores.exp().sum() # softmax the scores
print(scores)

# if scores[0] > scores[1] → image_base is better
# if scores[0] < scores[1] → image_mine is better
```
