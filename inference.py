#credits to lopho

import os
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import numpy as np
import random

from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers import DDPMScheduler, UNetPseudo3DConditionModel, AutoencoderKL
from diffusers import StableDiffusionVideoInpaintPipeline
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import time
from einops import rearrange
import imageio
from lion_pytorch import Lion

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    pretrained_model_name_or_path = '/workspace/disk/models/latest'
    seed = 22
    frames_length = 22

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='fp16',
        log_with='tensorboard',
        logging_dir='logs'
    )

    unet = UNetPseudo3DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder='unet'
    )
    unet.enable_gradient_checkpointing()
    unet.enable_xformers_memory_efficient_attention()

    noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")
    unet = accelerator.prepare(unet)
    with accelerator.autocast():
        tmp_pipe = StableDiffusionVideoInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            torch_dtype=torch.float32
        ).to("cuda")
        init_image = Image.open("/workspace/TempoFunk/dancer.png").convert("RGB").resize((512, 512))
        mask_image = Image.new("L", (512,512), 0).convert("RGB")
        for i in range(100):
            outputs = tmp_pipe('"A man dancing in the middle of the scene, coreography, beautiful, motion, adequate lighting, professional"',
                                negative_prompt='"Disgusting, Bad lighting, Too bright, disco lights, Bad quality, Shaky"',
                                image=init_image, 
                                mask_image=mask_image, 
                                num_inference_steps=100, 
                                guidance_scale=12.0, 
                                frames_length=frames_length).images
            imageio.mimsave(f"{str(i)}_.gif", outputs, fps=frames_length)

if __name__ == "__main__":
    main()
