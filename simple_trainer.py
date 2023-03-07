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
from model.unet_pseudo3d_condition import UNetPseudo3DConditionModel
from model.pipeline_stable_diffusion_video_inpaint import StableDiffusionVideoInpaintPipeline
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import time
from einops import rearrange
import imageio
from lion_pytorch import Lion

class LatentDataset(Dataset):
    def __init__(self, folders_path, texts_path, frames, verbose: bool, device):
        self.folders_path = folders_path
        self.texts_path = texts_path
        self.foldersmap = os.listdir(folders_path)
        self.verbose = verbose
        self.device = device
        blocksmap = []

        for folder in sorted(self.foldersmap):
            tmp = []
            frame_index = 0
            #frames/001/001.png, 002.png, etc.
            #texts/001.pt
            expected_text = os.path.join(self.texts_path, f'{folder}.pt')
            if os.path.exists(expected_text) is False:
                raise OSError(f"LatentDataset: Text tensor not found at {str(expected_text)}")
            for latentframe in sorted(os.listdir(os.path.join(self.folders_path, folder))):
                tmp.append(os.path.join(self.folders_path, folder, latentframe))
                frame_index = frame_index + 1
                if frame_index >= frames:
                    blocksmap.append({"video": deepcopy(tmp), "text": expected_text})
                    frame_index = 0
                    tmp = []

        self.blocksmap = blocksmap

    def __len__(self):
        return len(self.blocksmap)
    
    def __getitem__(self, idx):
        blocklist_pointers = self.blocksmap[idx]
        blocklist = []
        for block in blocklist_pointers['video']:
            tensor = torch.load(block)['mean']
            #if self.verbose:
            #    print("Loaded Block:", tensor.shape)
            blocklist.append(tensor)
        video_embed = torch.stack(blocklist)
        stage_1 = deepcopy(video_embed.shape)
        video_embed = rearrange(video_embed, 'f c h w -> c f h w')
        stage_2 = deepcopy(video_embed.shape)
        video_embed.to(self.device)
        text_embed = torch.load(blocklist_pointers['text'])
        text_embed = text_embed.to(self.device)
        text_embed = torch.squeeze(text_embed)
        if self.verbose is True:
            print(
                f'''
                Sample Loaded Block: {tensor.shape}
                Stacked Video Embed: {stage_1}
                Rearranged Video Embed: {stage_2}
                Text Embed: {text_embed.shape}
                '''
            )
            torch.save(video_embed, 'video.pt')
        return text_embed, video_embed

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(epochs: int = 18):
    pretrained_model_name_or_path = '/workspace/disk/models/latest'
    seed = 22
    #learning_rate = 1e-4
    learning_rate = 0.000033333333333333335
    gradient_accumulation_steps = 1
    batch_size = 10
    frames_length = 22
    project_name = "TempoFunk"
    training_name = f"v8-17-BS10-INTERPO-LOCA_lr{str(learning_rate)}"
    lr_warmup_steps = 0
    unfreeze_all = False
    enable_wandb = True
    enable_validation = True
    enable_inference = True
    dataloader_verbose = False
    save_steps = 300
    infer_step = 300
    start_step = 2700 #<- usually 0
    val_step = 12
    save_path = f"/workspace/disk/models/{training_name}/"
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16',
        log_with='tensorboard',
        logging_dir='logs'
    )
    world_size = accelerator.num_processes
    print(f"There are {world_size} processes.")
    set_seed(seed)

    if accelerator.is_local_main_process:
        if enable_wandb:
            import wandb
            run = wandb.init(project=project_name, name=training_name, mode="online")
            #TODO: turn config into a proper dict?
            run.config.lr = learning_rate
            run.config.gradient_accumululation_steps = gradient_accumulation_steps
            run.config.batch_size = batch_size
            run.config.frames_length = frames_length
            run.config.lr_warmup_steps = lr_warmup_steps
            run.config.unfreeze_all = unfreeze_all
            run.config.enable_validation = enable_validation
            run.config.enable_inference = enable_inference
            run.config.save_steps = save_steps
            run.config.world_size = world_size
            run.config.infer_srep = infer_step
            run.config.val_step = val_step
            run.config.pretrained_model_name_or_path = pretrained_model_name_or_path
            run.config.seed = seed
    train_dataset = LatentDataset(
        "/workspace/disk/dataset/processed/train/frames",
        "/workspace/disk/dataset/processed/train/text",
        frames_length,
        dataloader_verbose,
        accelerator.device
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = LatentDataset(
        "/workspace/disk/dataset/processed/val/frames",
        "/workspace/disk/dataset/processed/val/text",
        frames_length,
        dataloader_verbose,
        accelerator.device
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    os.makedirs(save_path, exist_ok=True)

    unet = UNetPseudo3DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder='unet'
    )
    unet.enable_gradient_checkpointing()
    unet.enable_xformers_memory_efficient_attention()

    unet.train()
    if not unfreeze_all:
        unet.requires_grad_(False)
        for name, param in unet.named_parameters():
            if 'temporal_conv' in name:
                param.requires_grad_(True)
        for block in unet.down_blocks:
            if hasattr(block, 'attentions') and block.attentions is not None:
                for attn_block in block.attentions:
                    for transformer_block in attn_block.transformer_blocks:
                        transformer_block.requires_grad_(False)
                        transformer_block.attn_temporal.requires_grad_(True)
                        transformer_block.norm_temporal.requires_grad_(True)
        for block in [unet.mid_block,]:
            if hasattr(block, 'attentions') and block.attentions is not None:
                for attn_block in block.attentions:
                    for transformer_block in attn_block.transformer_blocks:
                        transformer_block.requires_grad_(False)
                        transformer_block.attn_temporal.requires_grad_(True)
                        transformer_block.norm_temporal.requires_grad_(True)
        for block in unet.up_blocks:
            if hasattr(block, 'attentions') and block.attentions is not None:
                for attn_block in block.attentions:
                    for transformer_block in attn_block.transformer_blocks:
                        transformer_block.requires_grad_(False)
                        transformer_block.attn_temporal.requires_grad_(True)
                        transformer_block.norm_temporal.requires_grad_(True)
    params_to_optimize = (
        filter(lambda p: p.requires_grad, unet.parameters()) 
    )
    optimizer_class = Lion
    optimizer = optimizer_class(
        params_to_optimize,
        lr = learning_rate,
        weight_decay = 0.06
    )

    noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

    num_update_steps_per_epoch = math.ceil(train_dataloader.__len__() / gradient_accumulation_steps / world_size)
    max_train_steps = epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        'constant_with_warmup',
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    unet, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader, val_dataloader
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("video_diffusion")

    progress_bar = tqdm(range(max_train_steps))
    global_step = 0 + start_step
    tqdm.write(str(max_train_steps))
    tqdm.write(str(train_dataloader.__len__()))
    tqdm.write(str(epochs))

    print("starting...")

    def get_loss(text_embed, video_embed):
        latents = video_embed * 0.18215
        hint_latents = latents[:,:,0::21,:,:]
        input_latents = latents[:,:,1:,:,:]
        noise = torch.randn_like(input_latents)
        bsz = input_latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=input_latents.device)
        timesteps = timesteps.long()
        noisy_latents = noise_scheduler.add_noise(input_latents, noise, timesteps)
        hint_latents_expand = hint_latents.repeat_interleave(21,2)
        hint_latents_expand = hint_latents_expand[:,:,:frames_length-1,:,:]
        masks_input = torch.zeros([noisy_latents.shape[0], 1, noisy_latents.shape[2], noisy_latents.shape[3], noisy_latents.shape[4]]).to(accelerator.device)
        latent_model_input = torch.cat([noisy_latents, masks_input, hint_latents_expand], dim=1).to(accelerator.device)

        encoder_hidden_states = text_embed

        with accelerator.autocast():
            noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss

    saved_step = 0
    for epoch in range(epochs):
        for text_embed, video_embed in train_dataloader:
            b_start = time.perf_counter()
            with accelerator.accumulate(unet):
                if dataloader_verbose:
                    print("Recieved text_embed", text_embed.shape)
                    print("Recieved video_embed", video_embed.shape)
                loss = get_loss(text_embed, video_embed)
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            logs = {"loss": accelerator.gather(loss.repeat(batch_size)).mean().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            b_end = time.perf_counter()

            if (global_step % infer_step) == 0 and enable_inference:
                set_seed(accelerator.process_index)
                with accelerator.autocast():
                    tmp_pipe = StableDiffusionVideoInpaintPipeline.from_pretrained(
                        pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        torch_dtype=torch.float32
                    ).to(accelerator.device)
                    init_image = Image.open("/workspace/TempoFunk/dancer.png").convert("RGB").resize((512, 512))
                    mask_image = Image.new("L", (512,512), 0).convert("RGB")
                    outputs = tmp_pipe("Dancing Coreography", 
                                        image=init_image, 
                                        mask_image=mask_image, 
                                        num_inference_steps=100, 
                                        guidance_scale=12.0, 
                                        frames_length=frames_length).images
                    imageio.mimsave(f"{save_path}/{str(global_step)}_{accelerator.process_index}.gif", outputs, fps=frames_length)
                set_seed(seed)
            if (global_step % val_step) == 0 and enable_validation is True:
                with torch.no_grad():
                    val_loss = torch.tensor(0., device=accelerator.device)
                    for text_embed, video_embed in val_dataloader:
                        val_pred_loss = get_loss(text_embed, video_embed)
                        val_loss = val_loss +  val_pred_loss / len(val_dataloader)
                    val_loss = accelerator.gather(val_loss).mean().item()
                    print("VALIDATION LOSS:", val_loss)
                    if accelerator.is_local_main_process and enable_wandb is True:
                        run.log({'val_loss': val_loss}, step=global_step)
            if accelerator.is_local_main_process:
                if enable_wandb:
                    if accelerator.is_local_main_process:
                        seconds_per_step = b_end - b_start
                        steps_per_second = 1 / seconds_per_step
                        rank_tensors_per_second = batch_size * steps_per_second
                        world_images_per_second = rank_tensors_per_second * world_size
                        tensors_seen = global_step * batch_size * world_size
                        wandblogs = {
                            "loss": logs["loss"] / gradient_accumulation_steps,
                            "lr": logs["lr"],
                            "train_step": global_step,
                            "epoch": epoch,
                            "rank.t/s": rank_tensors_per_second,
                            "world.t/s": world_images_per_second,
                            "tensors_seen": tensors_seen
                        }
                        run.log(wandblogs, step=global_step)
                if (global_step % save_steps) == 0 and (saved_step != global_step) and accelerator.is_local_main_process:
                    saved_unet = accelerator.unwrap_model(unet, keep_fp32_wrapper = False)
                    save_at = f'{save_path}/{str(global_step)}.pt'
                    torch.save(saved_unet.state_dict(), save_at)
            # if global_step >= max_train_steps:
            #     break
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        saved_unet = accelerator.unwrap_model(unet, keep_fp32_wrapper = False)
        save_at = f'{save_path}/{str(global_step)}.pt'
        torch.save(saved_unet.state_dict(), save_at)
    accelerator.end_training()

if __name__ == "__main__":
    main()
