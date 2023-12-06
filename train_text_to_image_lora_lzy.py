# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import json
import logging
import math
import os
import re
import random
from pathlib import Path
from typing import Optional
from PIL import Image
import torch.nn.functional as F
import cv2

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from EMAclass import ExponentialMovingAverage
import copy


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def save_model_card(repo_name, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_name}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


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
        "--dataset_folder",
        type=str,
        default="./",
        help="Path to dataset folder.",
    )
    parser.add_argument(
        "--regularization_annotation",
        type=str,
        default=None,
        help="Path to regularization file.",
    )
    parser.add_argument(
        "--negative_prefix", type=str, default="Weird image. ", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--self_negative_prefix", type=str, default="Bad face. ", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_dropout", type=float, default=0.0, help="drop out rate of prompt for classifier free guidance."
    )
    parser.add_argument(
        "--validation_prompt_file", type=str, default=None, help="A file with prompts that is sampled during training for inference."
    )
    parser.add_argument(
        "--evaluation_prompts", type=str, default="validation_prompts_1000.json", help="A file with prompts that is sampled during training for evaluation."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_iters",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X iters. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help=(
            "the rank of lora residule"
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_dir",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
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
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
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
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
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
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--ema_update_steps",
        type=int,
        default=50,
        help=(
            "used for ema update"
        ),
    )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    args = parse_args()
    os.makedirs(args.output_dir,exist_ok = True)
    project_dir = os.path.join(args.output_dir, args.logging_dir) 
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=project_dir,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    # unet_original = UNet2DConditionModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    # )

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    # unet_original.requires_grad_(False)
    vae.requires_grad_(False)

    text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    # unet_original.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            # unet_original.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=args.lora_rank,
        )
    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    # lora_layers_ema = copy.deepcopy(lora_layers).to(unet.device) 
    # lora_layers_ema = ExponentialMovingAverage(lora_layers,device=unet.device)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    # class ImageTextFaceDataset(torch.utils.data.Dataset):
    #     def __init__(self, dataset_folder, regularization_file, tokenizer, image_transforms, 
    #                 mask_transforms, args):
    #         #dataset_folder = '/data/lzy/LAION-FACE-100k/split_00000'
    #         self.tokenizer = tokenizer
    #         self.image_transforms = image_transforms
    #         self.mask_transforms = mask_transforms
    #         self.args = args

    #         self.images = []
    #         self.captions = []
    #         self.masks = []
    #         self.sub_folders = []
    #         self.weights = []
            
    #         for i in range(10):
    #             self.sub_folders.append(f'0000{i}')
            
    #         #face data
    #         for sub_folder in self.sub_folders:
    #             json_list = sorted(os.listdir(os.path.join(dataset_folder,sub_folder,'jsons')))
    #             image_list = sorted(os.listdir(os.path.join(dataset_folder,sub_folder,'images')))
    #             for json_file,image_file in zip(json_list,image_list):
    #                 assert json_file[0:-4] == image_file[0:-3]
    #                 json_path = os.path.join(dataset_folder,sub_folder,'jsons',json_file)
    #                 with open(json_path, "r") as f:
    #                     info = json.load(f)
    #                     if info['caption'] is None:
    #                         self.captions.append('')
    #                     else:
    #                         self.captions.append(info['caption'])
    #                     self.images.append(os.path.join(os.path.join(dataset_folder,sub_folder,'images',image_file)))
    #                     self.masks.append(os.path.join(os.path.join(dataset_folder,sub_folder,'masks',image_file)))
    #                     self.weights.append(1)

    #                     if info['caption'] is None:
    #                         self.captions.append('')
    #                     else:
    #                         self.captions.append(info['caption'])
    #                     self.images.append(os.path.join(os.path.join(dataset_folder,sub_folder,'bad_images',image_file)))
    #                     self.masks.append(os.path.join(os.path.join(dataset_folder,sub_folder,'masks',image_file)))
    #                     self.weights.append(-1)
    #                     # self.good_images.append(os.path.join(os.path.join(dataset_folder,sub_folder,'images',image_file)))
    #                     # self.bad_images.append(os.path.join(os.path.join(dataset_folder,sub_folder,'bad_images',image_file)))
    #                     # self.masks.append(os.path.join(os.path.join(dataset_folder,sub_folder,'masks',image_file)))
    #         # counter = 0
    #         if regularization_file is not None:
    #             with open(regularization_file, 'r') as f:
    #                 for key, row in enumerate(f):
    #                     data = json.loads(row)
    #                     image_path = data['file_name']
    #                     # filename = os.path.basename(original_image_path)
    #                     # directory = os.path.dirname(original_image_path)
    #                     # image_path_number = int(re.findall(r'(\d+)\.jpg', filename)[0])
    #                     # sub_folder = image_path_number // 1000

    #                     # modified_image_path = os.path.join(directory,f"{sub_folder:04d}",filename)
    #                     # # 使用字符串的split方法将文件路径按照"/"进行分割
    #                     # path_parts = modified_image_path.split("/")
    #                     # # 找到要去掉的一级目录的索引
    #                     # index_to_remove = path_parts.index("regularization_images")
    #                     # # 使用列表切片操作获取去掉一级目录后的路径部分
    #                     # new_path_parts = path_parts[:index_to_remove] + path_parts[index_to_remove + 1:]
    #                     # # 使用join方法将路径部分重新组合成新的文件路径
    #                     # new_file_path = "/".join(new_path_parts)
    #                     self.images.append(image_path)
    #                     self.weights.append(1)
    #                     prompt = data['caption']
    #                     if prompt is None:
    #                         prompt = ''
    #                     self.captions.append(prompt)
    #                     #加入一个全1的mask 不要黑色的
    #                     self.masks.append('/data/lzy/improved_aesthetics_6.5plus1/mask_image.jpg')
    #                     # counter += 1
    #                     # if counter == 1000:
    #                     #     break
        
    #     def __len__(self):
    #         assert len(self.images) == len(self.captions)
    #         assert len(self.images) == len(self.weights)
    #         return len(self.images)
        
    #     def __getitem__(self, idx):
    #         image = Image.open(self.images[idx]).convert('RGB')
    #         mask = Image.open(self.masks[idx]).convert('RGB')
    #         if self.image_transforms is not None:
    #             image = self.image_transforms(image)
    #             #binarize
    #             mask = self.mask_transforms(mask)
    #         if random.random() < self.args.prompt_dropout:
    #             prompt = ""
    #         else:
    #             prompt = self.captions[idx]
    #         inputs = self.tokenizer(prompt, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
    #         return dict(
    #             pixel_values=image,
    #             mask_values =mask,
    #             weight_value = torch.tensor([self.weights[idx]]),
    #             input_ids=inputs.input_ids,
    #         )
    
    # class ImageTextDataset(torch.utils.data.Dataset): 
    #     def __init__(self, dataset_folder, regularization_annotation, tokenizer, image_transforms, 
    #                 mask_transforms, args):
    #         #dataset_folder = '/data/lzy/LAION-FACE-100k/split_00000'
    #         self.tokenizer = tokenizer
    #         self.image_transforms = image_transforms
    #         self.mask_transforms = mask_transforms
    #         self.args = args

    #         self.face_images = []
    #         self.bad_face_images = []
    #         self.face_masks = []
    #         self.face_captions = []

    #         self.aes_images = []
    #         self.aes_captions = []
    #         self.sub_folders = []

    #         for i in range(10):
    #             self.sub_folders.append(f'0000{i}')
            
    #         #face data
    #         for sub_folder in self.sub_folders:
    #             json_list = sorted(os.listdir(os.path.join(dataset_folder,sub_folder,'jsons')))
    #             image_list = sorted(os.listdir(os.path.join(dataset_folder,sub_folder,'images')))
    #             for json_file,image_file in zip(json_list,image_list):
    #                 assert json_file[0:-4] == image_file[0:-3]
    #                 json_path = os.path.join(dataset_folder,sub_folder,'jsons',json_file)
    #                 with open(json_path, "r") as f:
    #                     info = json.load(f)
    #                     if info['caption'] is None:
    #                         self.face_captions.append('')
    #                     else:
    #                         self.face_captions.append(info['caption'])
    #                     self.face_masks.append(os.path.join(os.path.join(dataset_folder,sub_folder,'masks',image_file)))
    #                     self.face_images.append(os.path.join(os.path.join(dataset_folder,sub_folder,'images',image_file)))
    #                     # self.masks.append(os.path.join(os.path.join(dataset_folder,sub_folder,'masks',image_file)))
    #                     self.bad_face_images.append(os.path.join(os.path.join(dataset_folder,sub_folder,'bad_images',image_file)))
    #         if regularization_annotation is not None:            
    #             with open(regularization_annotation, 'r') as f:
    #                 for key, row in enumerate(f):
    #                     data = json.loads(row)
    #                     image_path = data['file_name']
    #                     self.aes_images.append(image_path)
    #                     prompt = data['caption']
    #                     if prompt is None:
    #                         prompt = ''
    #                     self.aes_captions.append(prompt)
  
    #     def __len__(self):
    #         assert len(self.face_images) == len(self.aes_images) 
    #         assert len(self.face_images) == len(self.bad_face_images)
    #         assert len(self.face_images) == len(self.face_masks)
    #         return len(self.aes_images)
        
    #     def __getitem__(self, idx):
    #         face_image = Image.open(self.face_images[idx]).convert('RGB')
    #         bad_face_image = Image.open(self.bad_face_images[idx]).convert('RGB')
    #         aes_image = Image.open(self.aes_images[idx]).convert('RGB')
    #         mask = Image.open(self.face_masks[idx]).convert('RGB')
    #         # mask = Image.open('/data/lzy/LAION-FACE-100k/split_00000/00004/masks/000048961.jpg').convert('RGB')
    #         if self.image_transforms is not None:
    #             face_image = self.image_transforms(face_image)
    #             bad_face_image = self.image_transforms(bad_face_image)
    #             aes_image = self.image_transforms(aes_image)
    #             face_mask = self.mask_transforms(mask)
            
    #         if random.random() < self.args.prompt_dropout:
    #             face_prompt = ""
    #         else:
    #             face_prompt = self.face_captions[idx]
    #         if random.random() < self.args.prompt_dropout:
    #             aes_prompt = ""
    #         else:
    #             aes_prompt = self.aes_captions[idx]
    #         face_inputs = self.tokenizer(face_prompt, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
    #         bad_face_inputs = self.tokenizer("", padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
    #         aes_inputs = self.tokenizer(aes_prompt, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
    #         return dict(
    #             face_pixel_values=face_image,
    #             bad_face_pixel_values=bad_face_image,
    #             aes_pixel_values=aes_image,
    #             face_mask_values=face_mask,
    #             face_input_ids=face_inputs.input_ids,
    #             bad_face_input_ids=bad_face_inputs.input_ids,
    #             aes_input_ids=aes_inputs.input_ids,
    #         )


    class ImageTextDataset(torch.utils.data.Dataset):
        def __init__(self, tokenizer, image_transforms,mask_transforms,args):
            #dataset_folder = '/data/lzy/LAION-FACE-100k/split_00000'
            self.tokenizer = tokenizer
            self.image_transforms = image_transforms
            self.mask_transforms = mask_transforms
            self.args = args

            self.face_images = []
            self.bad_face_images = []
            self.captions = []
            self.masks = []
            annotation_path = '/data/lzy/coco/annotations/people_captions_train2014_1.json'
            with open(annotation_path, 'r') as file:
                data = json.load(file)
            #face data
            image_list = sorted(os.listdir('/data/lzy/coco/resize_1/'))
            for image_name in image_list:
                pattern = r"_(\d+)\."      
                match = re.search(pattern, image_name)
                image_id = int(match.group(1))    
                caption = [item for item in data if item['image_id'] == image_id][0]['caption']
                self.captions.append(caption)
                self.face_images.append(os.path.join('/data/lzy/coco/resize_1/',image_name))
                for i in range(10):
                    bad_image_name = image_name[0:-4] + f'_{i}' + image_name[-4:]
                    self.bad_face_images.append(os.path.join('/data/lzy/coco/bad_people_images_1/',bad_image_name))
                self.masks.append(os.path.join('/data/lzy/coco/masks_1/',image_name))
            # sorted_pairs = sorted(zip(self.face_images,self.captions), key = lambda x: x[0])
            # self.face_images,self.captions = zip(*sorted_pairs)
            

        def __len__(self):
            assert len(self.face_images) * 10 == len(self.bad_face_images)
            assert len(self.masks) == len(self.face_images)
            return len(self.bad_face_images)
        
        def __getitem__(self, idx):
            face_image = Image.open(self.face_images[idx//10]).convert('RGB')
            bad_face_image = Image.open(self.bad_face_images[idx]).convert('RGB')
            mask_image = Image.open(self.face_images[idx//10]).convert('RGB')
            if self.image_transforms is not None:
                face_image = self.image_transforms(face_image)
                bad_face_image = self.image_transforms(bad_face_image)
            if self.mask_transforms is not None:
                mask_image = self.mask_transforms(mask_image)
            
            if random.random() < self.args.prompt_dropout:
                face_prompt = ""
            else:
                face_prompt = self.captions[idx//10]
            face_inputs = self.tokenizer(face_prompt, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
            # bad_face_inputs = self.tokenizer(args.self_negative_prefix + face_prompt, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
            return dict(
                face_pixel_values=face_image,
                bad_face_pixel_values=bad_face_image,
                mask_pixel_values = mask_image,
                face_input_ids=face_inputs.input_ids,
                # bad_face_input_ids = bad_face_inputs.input_ids
            )


    # Preprocessing the datasets.
    train_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_mask_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ]
    )

    with accelerator.main_process_first():
        # train_dataset = ImageTextDataset(args.dataset_folder, args.regularization_annotation, tokenizer, train_image_transforms, train_mask_transforms, args)
        train_dataset = ImageTextDataset(tokenizer, train_image_transforms, train_mask_transforms,args)

    def collate_fn(examples):
        pixel_values = torch.cat([torch.stack([example["face_pixel_values"], example['bad_face_pixel_values']]) for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        mask_pixel_values = torch.cat([torch.stack([example["mask_pixel_values"]]) for example in examples])
        mask_pixel_values = mask_pixel_values.to(memory_format=torch.contiguous_format).float()
        # input_ids = torch.cat([torch.cat([example["face_input_ids"],example["bad_face_input_ids"]]) for example in examples])
        input_ids = torch.cat([torch.cat([example["face_input_ids"],example["face_input_ids"]]) for example in examples])
        return {"pixel_values": pixel_values,"mask_pixel_values":mask_pixel_values, "input_ids": input_ids}
        # pixel_values = torch.cat([torch.stack([example["face_pixel_values"]]) for example in examples])
        # input_ids = torch.cat([torch.cat([example["face_input_ids"]]) for example in examples])
        # return {"pixel_values": pixel_values,"input_ids": input_ids}
        # return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("train text2image LoRA", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
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
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    validation_prompts = []
    if args.validation_prompt_file is not None:
        with open(args.validation_prompt_file, 'r') as f:
            validation_prompts = f.readlines()
        validation_prompts = [t.strip() for t in validation_prompts]
    # if accelerator.is_main_process:
    #     pipeline1 = StableDiffusionPipeline.from_pretrained(
    #                         "models/sd1.4",
    #                         revision=args.revision,
    #                         torch_dtype=weight_dtype,
    #                         requires_safety_checker=False,
    #                         safety_checker=None,
    #                     )
    #     pipeline1 = pipeline1.to(accelerator.device)
    #     pipeline1.set_progress_bar_config(disable=True)
    #     model_weight = torch.load('models/normalization_only.bin', map_location='cpu')
    #     unet1 = pipeline1.unet
    #     lora_attn_procs = {}
    #     lora_rank = list(set([v.size(0) for k, v in model_weight.items() if k.endswith("down.weight")]))
    #     assert len(lora_rank) == 1
    #     lora_rank = lora_rank[0]
    #     for name in unet1.attn_processors.keys():
    #         cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    #         if name.startswith("mid_block"):
    #             hidden_size = unet1.config.block_out_channels[-1]
    #         elif name.startswith("up_blocks"):
    #             block_id = int(name[len("up_blocks.")])
    #             hidden_size = list(reversed(unet1.config.block_out_channels))[block_id]
    #         elif name.startswith("down_blocks"):
    #             block_id = int(name[len("down_blocks.")])
    #             hidden_size = unet1.config.block_out_channels[block_id]

    #         lora_attn_procs[name] = LoRACrossAttnProcessor(
    #             hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
    #         ).to(pipeline1.device)
    #     unet1.set_attn_processor(lora_attn_procs)
    #     unet1.load_state_dict(model_weight, strict=False)

        # for pt_id, validation_prompt in enumerate(validation_prompts):
        #         # run inference
        #     norm_generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        #     norm_images = []
        #     for _ in range(args.num_validation_images):
        #         norm_images.append(
        #             pipeline1(validation_prompt, num_inference_steps=50, generator=norm_generator, negative_prompt=args.negative_prefix).images[0]
        #         )

        #     for tracker in accelerator.trackers:
        #         if tracker.name == "tensorboard":
        #             norm_np_images = np.stack([np.asarray(img) for img in norm_images])
        #             tracker.writer.add_images(f"validation_{pt_id}_norm_only", norm_np_images, global_step, dataformats="NHWC")
        #         if tracker.name == "wandb":
        #             tracker.log(
        #                 {
        #                     "validation_norm_only": [
        #                         wandb.Image(image, caption=f"{i}: {'-' + args.negative_prefix + validation_prompt}")
        #                         for i, image in enumerate(norm_images)
        #                     ],
        #                 }
        #             )

        # del pipeline1
    # if accelerator.is_main_process:
        # pipeline1 = StableDiffusionPipeline.from_pretrained(
        #                     "models/sd1.4",
        #                     revision=args.revision,
        #                     torch_dtype=weight_dtype,
        #                     requires_safety_checker=False,
        #                     safety_checker=None,
        #                 )
        # pipeline1 = pipeline1.to(accelerator.device)
        # pipeline1.set_progress_bar_config(disable=True)
        # model_weight = torch.load('models/adapted_model.bin', map_location='cpu')
        # unet1 = pipeline1.unet
        # lora_attn_procs = {}
        # lora_rank = list(set([v.size(0) for k, v in model_weight.items() if k.endswith("down.weight")]))
        # assert len(lora_rank) == 1
        # lora_rank = lora_rank[0]
        # for name in unet1.attn_processors.keys():
        #     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        #     if name.startswith("mid_block"):
        #         hidden_size = unet1.config.block_out_channels[-1]
        #     elif name.startswith("up_blocks"):
        #         block_id = int(name[len("up_blocks.")])
        #         hidden_size = list(reversed(unet1.config.block_out_channels))[block_id]
        #     elif name.startswith("down_blocks"):
        #         block_id = int(name[len("down_blocks.")])
        #         hidden_size = unet1.config.block_out_channels[block_id]

        #     lora_attn_procs[name] = LoRACrossAttnProcessor(
        #         hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
        #     ).to(pipeline1.device)
        # unet1.set_attn_processor(lora_attn_procs)
        # unet1.load_state_dict(model_weight, strict=False)

        # for pt_id, validation_prompt in enumerate(validation_prompts):
        #         # run inference
        #     pos_generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        #     pos_images = []
        #     for _ in range(args.num_validation_images):
        #         pos_images.append(
        #             pipeline1(validation_prompt, num_inference_steps=50, generator=pos_generator, negative_prompt=args.negative_prefix).images[0]
        #         )

        #     for tracker in accelerator.trackers:
        #         if tracker.name == "tensorboard":
        #             pos_np_images = np.stack([np.asarray(img) for img in pos_images])
        #             tracker.writer.add_images(f"validation_{pt_id}_positive", pos_np_images, global_step, dataformats="NHWC")
        #         if tracker.name == "wandb":
        #             tracker.log(
        #                 {
        #                     "validation_positive": [
        #                         wandb.Image(image, caption=f"{i}: {'-' + args.negative_prefix + validation_prompt}")
        #                         for i, image in enumerate(pos_images)
        #                     ],
        #                 }
        #             )

        # del pipeline1
    torch.cuda.empty_cache()
    first_val = False
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                
                # Convert images to latent space
                # images = batch["pixel_values"] * 255
                # for i in range(images.shape[0]):
                #     image = images[i].cpu().numpy().transpose((1, 2, 0))
                #     image = Image.fromarray(np.uint8(image))
                #     image.save(f"tmp/normal{i}.jpg")
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                mask = batch["mask_pixel_values"][:,0:1]
                mask = F.interpolate(mask, size=(latents.shape[2],latents.shape[3]), mode='nearest')
                mask = torch.where(mask < 0.5, torch.tensor(0), torch.tensor(1)).float()
                latents = latents * vae.config.scaling_factor
        
                # Sample noise that we'll add to the latents
                # For the same noise by broadcast
                bsz = latents.shape[0]
                noises = []
                for i in range(bsz//2):
                    noise = torch.randn_like(latents[0][None,]).repeat(2,1,1,1)
                    noises.append(noise)
                noise = torch.cat(noises, dim = 0)
                # noise = torch.randn_like(latents)


                # noise = noise.repeat(bsz//3,1,1,1)
                # Sample a random timestep for each image
                # timesteps = torch.randint(0, int(noise_scheduler.num_train_timesteps), (bsz,), device=latents.device)
                timesteps = torch.randint(0, int(noise_scheduler.num_train_timesteps),(1,),device=latents.device).expand(bsz)
                timesteps = timesteps.long()
   

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # reverse_image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0] * 255
                # reverse_image = reverse_image.cpu().numpy().astype(np.uint8)
                # for i in range(reverse_image.shape[0]):
                #     image = Image.fromarray(reverse_image[i].transpose(1, 2, 0))
                #     image.save(f"tmp/reverse{i}.jpg",mode='RGB')
                # exit()

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                # model_pred_original = unet_original(noisy_latents, timesteps, encoder_hidden_states).sample
                model_pred = model_pred.float()
                # model_pred_original = model_pred_original.float()
                target = target.float()
                loss_reconstruct = F.mse_loss(mask[[i for i in range(bsz//2)]] * model_pred[[2*i for i in range(bsz//2)]], mask[[i for i in range(bsz//2)]] * target[[2*i for i in range(bsz//2)]], reduction="mean")
                # loss_reconstruct = F.mse_loss(model_pred[[2*i for i in range(bsz//2)]], target[[2*i for i in range(bsz//2)]], reduction="mean")

                #To calculate the guidance loss
                # def weight_function(x):
                #     """
                #     30-40 need more weight
                #     40-50 can be less
                #     <30 least
                #     """
                #     if x <= 200:
                #         weight = (x - 1) / 199  
                #     elif x <= 400:
                #         weight = 1  
                #     else:
                #         weight = (1000 - x) / 600  
                #     return weight
                def cal_loss_adv_one_step():
                    """
                    The guidance loss
                    """
                    alphas_cumprod = noise_scheduler.alphas_cumprod.flatten().to(device = model_pred.device, dtype = model_pred.dtype)
                    snr = alphas_cumprod[timesteps[0]] ** 2/(1 - alphas_cumprod[timesteps[0]] ** 2)
                    # loss_weight = weight_function(timesteps[0])
                    sqrt_alpha_prod = alphas_cumprod[timesteps[0]] ** 0.5
                    sqrt_alpha_prod_prev = alphas_cumprod[timesteps[0] - 1] ** 0.5
                    sqrt_one_minus_alpha_prod = ((1 - alphas_cumprod[timesteps[0]]) ** 0.5)
                    sqrt_one_minus_alpha_prod_prev = ((1 - alphas_cumprod[timesteps[0] - 1]) ** 0.5)
                    while len(sqrt_alpha_prod.shape) < len(model_pred.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                        sqrt_alpha_prod_prev = sqrt_alpha_prod_prev.unsqueeze(-1)
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                        sqrt_one_minus_alpha_prod_prev = sqrt_one_minus_alpha_prod_prev.unsqueeze(-1)
                    correct_one_step_target = (1/sqrt_alpha_prod) * (noisy_latents[[2*i for i in range(bsz//2)]] - sqrt_one_minus_alpha_prod * model_pred[[2*i for i in range(bsz//2)]])
                    # correct_one_step_target = (1/sqrt_alpha_prod) * (noisy_latents[[2*i for i in range(bsz//2)]] - sqrt_one_minus_alpha_prod * noise[[2*i for i in range(bsz//2)]])
                    false_one_step_target = (1/sqrt_alpha_prod) * (noisy_latents[[2*i+1 for i in range(bsz//2)]] - sqrt_one_minus_alpha_prod * model_pred[[2*i+1 for i in range(bsz//2)]])
                    # correct_denoise_target = sqrt_alpha_prod_prev * correct_one_step_target + sqrt_one_minus_alpha_prod_prev * model_pred[[2*i for i in range(bsz//2)]]
                    correct_denoise_target = sqrt_alpha_prod_prev * correct_one_step_target + sqrt_one_minus_alpha_prod_prev * noise[[2*i for i in range(bsz//2)]]
                    false_denoise_target = sqrt_alpha_prod_prev * false_one_step_target + sqrt_one_minus_alpha_prod_prev * model_pred[[2*i+1 for i in range(bsz//2)]]
                    # correct_denoise_target = ((sqrt_alpha_prod_prev/sqrt_alpha_prod) * noisy_latents[[2*i for i in range(bsz//2)]]) + \
                    #                         (sqrt_one_minus_alpha_prod_prev - (sqrt_alpha_prod_prev/sqrt_alpha_prod) * sqrt_one_minus_alpha_prod) * model_pred[[2*i for i in range(bsz//2)]]
                    # false_denoise_target = ((sqrt_alpha_prod_prev/sqrt_alpha_prod) * noisy_latents[[2*i+1 for i in range(bsz//2)]]) + \
                    #                         (sqrt_one_minus_alpha_prod_prev - (sqrt_alpha_prod_prev/sqrt_alpha_prod) * sqrt_one_minus_alpha_prod) * model_pred[[2*i+1 for i in range(bsz//2)]]
                    # assert noisy_latents.shape[0] == 2 * mask.shape[0]
                    
                    loss_adv = snr * F.mse_loss(mask[[i for i in range(bsz//2)]] * correct_denoise_target.float().detach(),mask[[i for i in range(bsz//2)]] * false_denoise_target.float(), reduction="mean")
                    # loss_adv =  snr * F.mse_loss(correct_denoise_target.float().detach(), false_denoise_target.float(), reduction="mean")
                    return loss_adv

                    
                loss_adv = cal_loss_adv_one_step()
                # loss_adv = F.mse_loss(mask * model_pred[[3*i+1 for i in range(bsz//3)]], mask * model_pred.detach()[[3*i for i in range(bsz//3)]], reduction="mean")
                # loss_aes = F.mse_loss(model_pred[[3*i+2 for i in range(bsz//3)]], target[[3*i+2 for i in range(bsz//3)]], reduction="mean")
                # gamma = 1.5
                # loss = loss_reconstruct.mean() - loss_adv.mean() + loss_aes.mean()
                # loss = loss_reconstruct.mean()
                # loss = loss_reconstruct.mean() + 5 * loss_adv 
                loss = loss_reconstruct + loss_adv 
                # Gather the losses across all processes for logging (if we use distributed training).
                # loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # if step

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                # accelerator.log({"loss_positive": loss_positive.mean()}, step=global_step)
                # accelerator.log({"loss_negative": loss_negative.mean()}, step=global_step)
                # accelerator.log({"loss_aes": loss_aes.mean()}, step=global_step)
                accelerator.log({"loss_reconstruct": loss_reconstruct.mean()}, step=global_step)
                accelerator.log({"loss_adv": loss_adv.mean()}, step=global_step)
                # accelerator.log({"loss_one_step": loss_one_step.mean()}, step=global_step)
                # accelerator.log({"loss_adv": loss_adv.mean()}, step=global_step)
                # accelerator.log({"loss_aes": loss_aes.mean()}, step=global_step)
                train_loss = 0.0
            

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
            
            #add ema
            # print(step,args.gradient_accumulation_steps,global_step,args.ema_update_steps)
            # if step % args.gradient_accumulation_steps == 0 and global_step % args.ema_update_steps == 0:
            #     decay_rate = 0.5
            #     for ema_param, train_param in zip(lora_layers_ema.parameters(), lora_layers.parameters()):
            #         ema_param.data = decay_rate * ema_param.data + (1 - decay_rate) * train_param.data
            #     lora_layers.load_state_dict(lora_layers_ema.state_dict())
                # lora_layers_ema.update_parameters(lora_layers)
                # lora_layers.load_state_dict(lora_layers_ema.state_dict())
                # print('ema establish')
            
            if accelerator.is_main_process:
                if (accelerator.sync_gradients and args.validation_prompt_file is not None and (global_step) % args.validation_iters == 0) or (global_step == 0 and not first_val):
                    if global_step == 0 and not first_val:
                        first_val = True
                    torch.cuda.empty_cache()
                    logger.info(
                        f"Running validation... \n Generating {args.num_validation_images} images"
                    )
                    # create pipeline
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        revision=args.revision,
                        torch_dtype=weight_dtype,
                        requires_safety_checker=False,
                        safety_checker=None,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                
                    
                    # log example images for visualization
                    for pt_id, validation_prompt in enumerate(validation_prompts):
                        # run inference
                        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                        images = []
                        for _ in range(args.num_validation_images):
                            images.append(
                                # pipeline(validation_prompt, num_inference_steps=50, generator=generator,negative_prompt = args.self_negative_prefix).images[0]
                                pipeline(validation_prompt, num_inference_steps=50, generator=generator).images[0]
                            )

                        # neg_generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                        # neg_images = []
                        # for _ in range(args.num_validation_images):
                        #     neg_images.append(
                        #         pipeline1(args.negative_prefix + validation_prompt, num_inference_steps=50, generator=neg_generator).images[0]
                        #     )

                        # pos_generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                        # pos_images = []
                        # for _ in range(args.num_validation_images):
                        #     pos_images.append(
                        #         pipeline1(validation_prompt, num_inference_steps=50, generator=pos_generator, negative_prompt=args.negative_prefix).images[0]
                        #     )

                        for tracker in accelerator.trackers:
                            if tracker.name == "tensorboard":
                                np_images = np.stack([np.asarray(img) for img in images])
                                tracker.writer.add_images(f"validation_{pt_id}_neutral", np_images, global_step, dataformats="NHWC")
                                # neg_np_images = np.stack([np.asarray(img) for img in neg_images])
                                # tracker.writer.add_images(f"validation_{pt_id}_negative", neg_np_images, global_step, dataformats="NHWC")
                                # pos_np_images = np.stack([np.asarray(img) for img in pos_images])
                                # tracker.writer.add_images(f"validation_{pt_id}_positive", pos_np_images, global_step, dataformats="NHWC")
                            if tracker.name == "wandb":
                                tracker.log(
                                    {
                                        "validation_neutral": [
                                            wandb.Image(image, caption=f"{i}: {validation_prompt}")
                                            for i, image in enumerate(images)
                                        ],
                                        # "validation_negative": [
                                        #     wandb.Image(image, caption=f"{i}: {args.negative_prefix + validation_prompt}")
                                        #     for i, image in enumerate(neg_images)
                                        # ],
                                        # "validation_positive": [
                                        #     wandb.Image(image, caption=f"{i}: {'-' + args.negative_prefix + validation_prompt}")
                                        #     for i, image in enumerate(pos_images)
                                        # ],
                                    }
                                )

                    del pipeline
                    torch.cuda.empty_cache()


    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(args.output_dir)

    # skip final inference
    accelerator.end_training()
    return 


if __name__ == "__main__":
    main()
