#   Run the code with the following command
#
#   python main.py \
#   --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1" \
#   --dataset_name "data-is-better-together/open-image-preferences-v1-binarized" \
#   --train_batch_size 1 \
#   --gradient_accumulation_steps 64 \
#   --gradient_checkpointing \
#   --max_train_steps 250 \
#   --learning_rate 1e-5 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps 0 \
#   --output_dir "sd-finetune" \
#   --image_column "chosen" \
#   --caption_column "prompt" \
#   --cache_dir "cache" \

import argparse
import math
import os
import shutil

import torch
import torch.nn.functional as F

import io
from PIL import Image

import bitsandbytes as bnb

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

logger = get_logger(__name__, log_level="INFO")

def parse_args():
	parser = argparse.ArgumentParser(description="Simple example of a training script.")

	parser.add_argument(
    	"--pretrained_model_name_or_path",
    	type=str,
    	default=None,
    	required=True,
    	help="Path to pretrained model or model identifier from huggingface.co/models.",
	)
	parser.add_argument(
    	"--dataset_name",
    	type=str,
    	default=None,
    	help=(
        	"The name of the Dataset (from the HuggingFace hub) to train on "
        	" dataset)."
    	),
	)
	parser.add_argument(
    	"--dataset_config_name",
    	type=str,
    	default=None,
    	help="The config of the Dataset, leave as None if there's only one config.",
	)
	parser.add_argument(
    	"--image_column", type=str, default="image", help="The column of the dataset containing an image."
	)
	parser.add_argument(
    	"--caption_column",
    	type=str,
    	default="text",
    	help="The column of the dataset containing a caption or a list of captions.",
	)
	parser.add_argument(
    	"--output_dir",
    	type=str,
    	default="sd-model-finetuned",
    	help="The output directory where the model predictions and checkpoints will be written.",
	)
	parser.add_argument(
    	"--cache_dir",
    	type=str,
    	default=None,
    	help="The directory where the downloaded models and datasets will be stored.",
	)
	parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
	parser.add_argument(
    	"--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
	)
	parser.add_argument("--num_train_epochs", type=int, default=100)
	parser.add_argument(
    	"--max_train_steps",
    	type=int,
    	default=None,
    	help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
	)
	parser.add_argument(
    	"--gradient_accumulation_steps",
    	type=int,
    	default=1,
    	help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument(
    	"--gradient_checkpointing",
    	action="store_true",
    	help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
	)
	parser.add_argument(
    	"--learning_rate",
    	type=float,
    	default=1e-4,
    	help="Initial learning rate (after the potential warmup period) to use.",
	)
	parser.add_argument(
    	"--lr_scheduler",
    	type=str,
    	default="constant",
    	help=(
        	'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        	' "constant", "constant_with_warmup"]'
    	),
	)
	parser.add_argument(
    	"--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
	)
	parser.add_argument(
    	"--logging_dir",
    	type=str,
    	default="logs",
    	help=(
        	"[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
        	" *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
    	),
	)
	parser.add_argument(
    	"--checkpointing_steps",
    	type=int,
    	default=500,
    	help=(
        	"Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
        	" training using `--resume_from_checkpoint`."
    	),
	)
	parser.add_argument(
    	"--checkpoints_total_limit",
    	type=int,
    	default=None,
    	help=("Max number of checkpoints to store."),
	)
	parser.add_argument(
    	"--resume_from_checkpoint",
    	type=str,
    	default=None,
    	help=(
        	"Whether training should be resumed from a previous checkpoint. Use a path saved by"
        	' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
    	),
	)
	parser.add_argument(
    	"--tracker_project_name",
    	type=str,
    	default="text2image-fine-tune",
    	help=(
        	"The `project_name` argument passed to Accelerator.init_trackers for"
        	" more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
    	),
	)

	args = parser.parse_args()

	return args


def main():
	args = parse_args()

	accelerator = Accelerator(
    	gradient_accumulation_steps=args.gradient_accumulation_steps,
    	mixed_precision='fp16',
	)

	# If passed along, set the training seed now.
	if args.seed is not None:
    	set_seed(args.seed)

	# Handle the repository creation
	if accelerator.is_main_process:
    	if args.output_dir is not None:
        	os.makedirs(args.output_dir, exist_ok=True)

	# Load scheduler, tokenizer and models.
	noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
	tokenizer = CLIPTokenizer.from_pretrained(
    	args.pretrained_model_name_or_path, subfolder="tokenizer",
	)

	text_encoder = CLIPTextModel.from_pretrained(
    	args.pretrained_model_name_or_path, subfolder="text_encoder",
	)
	vae = AutoencoderKL.from_pretrained(
    	args.pretrained_model_name_or_path, subfolder="vae",
	)

	unet = UNet2DConditionModel.from_pretrained(
    	args.pretrained_model_name_or_path, subfolder="unet",
	)

	# Freeze vae and text_encoder and set unet to trainable
	vae.requires_grad_(False)
	text_encoder.requires_grad_(False)

	unet.train()
	unet.enable_xformers_memory_efficient_attention()
	unet.enable_gradient_checkpointing()

	# Initialize the optimizer
	optimizer = bnb.optim.AdamW8bit(
    	unet.parameters(),
    	lr=args.learning_rate,
    	betas=(0.9, 0.999),
    	weight_decay=1e-2,
    	eps=1e-08,
	)

	dataset = load_dataset(
    	args.dataset_name,
    	args.dataset_config_name,
    	cache_dir=args.cache_dir,
	)

	image_column = args.image_column

	caption_column = args.caption_column

	# Preprocessing the datasets.
	# I need to tokenize input captions and transform the images.
	def tokenize_captions(examples, is_train=True):
    	captions = []
    	for caption in examples[caption_column]:
        	captions.append(caption)
   	 
    	inputs = tokenizer(
        	captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    	)

    	return inputs.input_ids

	# Data preprocessing transformations
	train_transforms = transforms.Compose(
    	[
        	transforms.Resize(768, interpolation=transforms.InterpolationMode.LANCZOS),  # Use dynamic interpolation method
        	transforms.CenterCrop(768),
        	transforms.RandomHorizontalFlip(),
        	transforms.ToTensor(),
        	transforms.Normalize([0.5], [0.5]),
    	]
	)

	def preprocess_train(examples):
    	images = [Image.open(io.BytesIO(image["bytes"])) for image in examples[image_column]]
    	images = [image.convert("RGB") for image in images]
    	examples["pixel_values"] = [train_transforms(image) for image in images]
    	examples["input_ids"] = tokenize_captions(examples)
    	return examples

	dataset["train"] = dataset["train"].shuffle(seed=args.seed)
	# Set the training transforms
	train_dataset = dataset["train"].with_transform(preprocess_train)

	def collate_fn(examples):
    	pixel_values = torch.stack([example["pixel_values"] for example in examples])
    	pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    	input_ids = torch.stack([example["input_ids"] for example in examples])
    	return {"pixel_values": pixel_values, "input_ids": input_ids}

	# DataLoaders creation:
	train_dataloader = torch.utils.data.DataLoader(
    	train_dataset,
    	shuffle=True,
    	collate_fn=collate_fn,
    	batch_size=args.train_batch_size,
    	num_workers=0,
	)

	lr_scheduler = get_scheduler(
    	args.lr_scheduler,
    	optimizer=optimizer,
    	num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    	num_training_steps=args.max_train_steps * accelerator.num_processes,
	)

	# Prepare everything with my `accelerator`.
	unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    	unet, optimizer, train_dataloader, lr_scheduler
	)

	# Move text_encode and vae to gpu and cast to weight_dtype
	text_encoder.to(accelerator.device, dtype=torch.float16)
	vae.to(accelerator.device, dtype=torch.float16)

	# I need to recalculate my total training steps as the size of the training dataloader may have changed.
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

	# Afterwards I recalculate my number of training epochs
	args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

	# I need to initialize the trackers I use, and also store my configuration.
	# The trackers initializes automatically on the main process.
	if accelerator.is_main_process:
    	tracker_config = dict(vars(args))
    	accelerator.init_trackers(args.tracker_project_name, tracker_config)

	# Train
	total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

	logger.info("***** Running training *****")
	logger.info(f"  Num examples = {len(train_dataset)}")
	logger.info(f"  Num Epochs = {args.num_train_epochs}")
	logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
	logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
	logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
	logger.info(f"  Total optimization steps = {args.max_train_steps}")
	global_step = 0
	first_epoch = 0

	# Potentially load in the weights and states from a previous save
	if args.resume_from_checkpoint:
    	if args.resume_from_checkpoint != "latest":
        	path = os.path.basename(args.resume_from_checkpoint)
    	else:
        	# Get the most recent checkpoint
        	dirs = os.listdir(args.output_dir)
        	dirs = [d for d in dirs if d.startswith("checkpoint")]
        	dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        	path = dirs[-1] if len(dirs) > 0 else None

    	if path is None:
        	accelerator.print(
            	f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        	)
        	args.resume_from_checkpoint = None
        	initial_global_step = 0
    	else:
        	accelerator.print(f"Resuming from checkpoint {path}")
        	accelerator.load_state(os.path.join(args.output_dir, path))
        	global_step = int(path.split("-")[1])

        	initial_global_step = global_step
        	first_epoch = global_step // num_update_steps_per_epoch

	else:
    	initial_global_step = 0

	progress_bar = tqdm(
    	range(0, args.max_train_steps),
    	initial=initial_global_step,
    	desc="Steps",
    	# Only show the progress bar once on each machine.
    	disable=not accelerator.is_local_main_process,
	)

	for epoch in range(first_epoch, args.num_train_epochs):
    	train_loss = 0.0
    	for step, batch in enumerate(train_dataloader):
        	with accelerator.accumulate(unet):
            	# Convert images to latent space
            	latents = vae.encode(batch["pixel_values"].to(torch.float16)).latent_dist.sample()
            	latents = latents * vae.config.scaling_factor

            	# Sample noise that I'll add to the latents
            	noise = torch.randn_like(latents)
            	bsz = latents.shape[0]
            	# Sample a random timestep for each image
            	timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            	timesteps = timesteps.long()

            	# Add noise to the latents according to the noise magnitude at each timestep
            	# (this is the forward diffusion process)
            	noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            	# Get the text embedding for conditioning
            	encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

            	target = noise_scheduler.get_velocity(latents, noise, timesteps)

            	# Predict the noise residual and compute loss
            	model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
            	loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            	# Gather the losses across all processes for logging (if I use distributed training).
            	avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            	train_loss += avg_loss.item() / args.gradient_accumulation_steps

            	# Backpropagate
            	accelerator.backward(loss)
            	if accelerator.sync_gradients:
                	accelerator.clip_grad_norm_(unet.parameters(), 1.0)
            	optimizer.step()
            	lr_scheduler.step()
            	optimizer.zero_grad()

        	# Checks if the accelerator has performed an optimization step behind the scenes
        	if accelerator.sync_gradients:
            	progress_bar.update(1)
            	global_step += 1
            	accelerator.log({"train_loss": train_loss}, step=global_step)
            	train_loss = 0.0

            	if global_step % args.checkpointing_steps == 0:
                	if accelerator.is_main_process:
                    	# _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    	if args.checkpoints_total_limit is not None:
                        	checkpoints = os.listdir(args.output_dir)
                        	checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        	checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        	# before I save the new checkpoint, I need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        	if len(checkpoints) >= args.checkpoints_total_limit:
                            	num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            	removing_checkpoints = checkpoints[0:num_to_remove]

                            	logger.info(
                                	f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            	)
                            	logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            	for removing_checkpoint in removing_checkpoints:
                                	removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                	shutil.rmtree(removing_checkpoint)

                    	save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    	accelerator.save_state(save_path)
                    	logger.info(f"Saved state to {save_path}")

        	logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        	progress_bar.set_postfix(**logs)

        	if global_step >= args.max_train_steps:
            	break

	# Create the pipeline using the trained modules and save it.
	accelerator.wait_for_everyone()

	if accelerator.is_main_process:
    	pipeline = StableDiffusionPipeline.from_pretrained(
        	args.pretrained_model_name_or_path,
        	text_encoder=text_encoder,
        	vae=vae,
        	unet=unet,
        	revision=args.revision,
        	variant=args.variant,
    	)
    	pipeline.save_pretrained(args.output_dir)

	accelerator.end_training()


if __name__ == "__main__":
	main()
