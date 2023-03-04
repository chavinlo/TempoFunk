#credits to lopho

import os
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import numpy as np
import random

from accelerate import Accelerator
from diffusers import DDPMScheduler, UNetPseudo3DConditionModel, AutoencoderKL
from diffusers import StableDiffusionVideoInpaintPipeline
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm.auto import tqdm
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import time
import json
from einops import rearrange
import imageio
from lion_pytorch import Lion

def pil_to_torch(image, device = 'cpu'):
    return (2 * (pil_to_tensor(image).to(dtype=torch.float32, device=device)) / 255) - 1

def torch_to_pil(x):
    return to_pil_image((x + 1) / 2)

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

#unused
def encode_latents(path, outpath, model):
    files: list[str] = os.listdir(path)
    files.sort()
    os.makedirs(outpath, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(model, subfolder='vae').to('cuda:0')
    for f in tqdm(files):
        im = Image.open(os.path.join(path, f))
        im = ImageOps.fit(im, (512, 512), centering = (0.5, 0.5))
        with torch.inference_mode():
            m = pil_to_torch(im, 'cuda:0').unsqueeze(0)
            m = vae.encode(m).latent_dist
            torch.save({ 'mean': m.mean.squeeze().cpu(), 'std': m.std.squeeze().cpu() }, os.path.join(outpath, os.path.splitext(f)[0] + '.pt'))

def split_list(input_list, chunk_no):
    chunk_size = len(input_list) // chunk_no
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def custom_encode_latents(frames_path, text_path, outpath, model, gpus: int):
    from threading import Thread
    # gpus = 4
    folders: list[str] = sorted(os.listdir(frames_path))
    folders.sort()
    os.makedirs(outpath, exist_ok=True)
    model_list = []
    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder='tokenizer')
    for i in range(gpus):
        vae = AutoencoderKL.from_pretrained(model, subfolder='vae').to(f'cuda:{i}')
        text_encoder = CLIPTextModel.from_pretrained(model, subfolder='text_encoder').to(f'cuda:{i}')
        model_list.append({
            "gpu_id": i,
            "vae_obj": vae,
            "enc_obj": text_encoder
        })
    folders_chunks = split_list(folders, gpus)
    print(len(folders_chunks))
    print(len(folders_chunks[0]))

    def sepa_engine(chunk: list, model_dict: dict, frames_path: str, text_path: str, outpath: str,):
        #text file must have the same name as folder
        #/frames/0001/001.png, 002.png, etc
        #/texts/0001.txt
        vae = model_dict['vae_obj']
        enc = model_dict['enc_obj']
        gpu_id = model_dict['gpu_id']
        out_vid = os.path.join(outpath, 'frames')
        out_txt = os.path.join(outpath, 'text')
        os.makedirs(out_vid, exist_ok=True)
        os.makedirs(out_txt, exist_ok=True)
        for folder in tqdm(chunk):
            predicted_text_path = os.path.join(text_path, f'{folder}.txt')
            if os.path.exists(predicted_text_path) is False:
                raise OSError(f"LatentDataset: Text file not found at {str(predicted_text_path)}")
            video_label = open(predicted_text_path, "r").read()
            with torch.inference_mode():
                tokens = tokenizer(
                    [ video_label ],
                    truncation = True,
                    return_overflowing_tokens = False,
                    padding = 'max_length',
                    return_tensors = 'pt'
                ).input_ids.to(f'cuda:{gpu_id}')
                text_embed = enc(input_ids = tokens).last_hidden_state
                torch.save(text_embed, os.path.join(out_txt, f'{folder}.pt'))
            raw_folder_path = os.path.join(frames_path, folder)
            for img in sorted(os.listdir(raw_folder_path)):
                im = Image.open(os.path.join(raw_folder_path, img))
                im = ImageOps.fit(im, (512, 512), centering = (0.5, 0.5))
                with torch.inference_mode():
                    m = pil_to_torch(im, f'cuda:{gpu_id}').unsqueeze(0)
                    m = vae.encode(m).latent_dist
                    os.makedirs(os.path.join(out_vid, folder), exist_ok=True)
                    torch.save({ 'mean': m.mean.squeeze().cpu(), 'std': m.std.squeeze().cpu() }, os.path.join(out_vid, folder, os.path.splitext(img)[0] + '.pt'))

    for i in range(gpus):
        print("eh?")
        processThread = Thread(target=sepa_engine, args=(folders_chunks[i], model_list[i], frames_path, text_path, outpath,))
        processThread.start()

#TODO: fix device and del unused tokenizer/text encoder just like encode_single_prompt 
def encode_prompts(prompts, outpath, model):
    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder='text_encoder').to('cuda:0')
    for i, prompt in enumerate(prompts):
        with torch.inference_mode():
            tokens = tokenizer(
                    [ prompt ],
                    truncation = True,
                    return_overflowing_tokens = False,
                    padding = 'max_length',
                    return_tensors = 'pt'
            ).input_ids.to('cuda:0')
            y = text_encoder(input_ids = tokens).last_hidden_state
            torch.save(y.cpu(), os.path.join(outpath, str(i).zfill(4) + '.pt' ))

def encode_single_prompt(prompt, model, device):
    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder='text_encoder').to(device)
    with torch.inference_mode():
        tokens = tokenizer(
                [ prompt ],
                truncation = True,
                return_overflowing_tokens = False,
                padding = 'max_length',
                return_tensors = 'pt'
        ).input_ids.to(device)
        y = text_encoder(input_ids = tokens).last_hidden_state
    del tokenizer
    del text_encoder
    return y.cuda()

#TODO: fix device and del unused tokenizer/text encoder just like encode_single_prompt 
def load_dataset(latent_path, prompt_path, batch_size, frames):
    files: list[str] = os.listdir(latent_path)
    files.sort()
    assert len(files) >= batch_size * frames
    # just make one batch for testing
    files = files[:batch_size * frames]
    prompt: torch.Tensor = torch.load(prompt_path)
    prompt = prompt.pin_memory('cuda:0')
    latents: list[torch.Tensor] = []
    for f in tqdm(files):
        l: dict[str, torch.Tensor] = torch.load(os.path.join(latent_path, f))
        latents.append(l['mean'].squeeze())
    latents: torch.Tensor = torch.stack(latents).unsqueeze(0)
    latents = rearrange(latents, 'b f c h w -> b c f h w').unsqueeze(0)
    return latents.to('cuda:0'), prompt.to('cuda:0')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(epochs: int = 10):
    pretrained_model_name_or_path = '/workspace/TempoFunk/models/latest'
    seed = 22
    #learning_rate = 1e-4
    learning_rate = 0.000033333333333333335
    gradient_accumulation_steps = 1
    batch_size = 4
    frames_length = 55
    project_name = "TempoFunk"
    training_name = f"v8-11LOCA_lr{str(learning_rate)}"
    lr_warmup_steps = 0
    unfreeze_all = False
    enable_wandb = True
    enable_validation = True
    enable_inference = True
    dataloader_verbose = False
    save_steps = 300
    infer_step = 300
    start_step = 7800 #<- usually 0
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
        "/workspace/disk/webvid/processed/train/frames",
        "/workspace/disk/webvid/processed/train/text",
        frames_length,
        dataloader_verbose,
        accelerator.device
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = LatentDataset(
        "/workspace/disk/webvid/processed/val/frames",
        "/workspace/disk/webvid/processed/val/text",
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

    saved_step = 0
    for epoch in range(epochs):
        for text_embed, video_embed in train_dataloader:
            b_start = time.perf_counter()
            with accelerator.accumulate(unet):
                if dataloader_verbose:
                    print("Recieved text_embed", text_embed.shape)
                    print("Recieved video_embed", video_embed.shape)
                latents = video_embed * 0.18215
                hint_latent = latents[:,:,:1,:,:]
                input_latents = latents[:,:,1:,:,:]
                hint_latent = hint_latent
                input_latents = input_latents
                noise = torch.randn_like(input_latents)
                bsz = input_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=input_latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(input_latents, noise, timesteps)
                encoder_hidden_states = text_embed
                mask = torch.zeros([noisy_latents.shape[0], 1, noisy_latents.shape[2], noisy_latents.shape[3], noisy_latents.shape[4]]).to(accelerator.device)
                latent_model_input = torch.cat([noisy_latents, mask, hint_latent.expand(-1,-1,noisy_latents.shape[2],-1,-1)], dim=1).to(accelerator.device)
                with accelerator.autocast():
                    noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
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
                        latents = video_embed * 0.18215
                        hint_latent = latents[:,:,:1,:,:]
                        input_latents = latents[:,:,1:,:,:]
                        hint_latent = hint_latent
                        input_latents = input_latents
                        noise = torch.randn_like(input_latents)
                        bsz = input_latents.shape[0]
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=input_latents.device)
                        timesteps = timesteps.long()
                        noisy_latents = noise_scheduler.add_noise(input_latents, noise, timesteps)
                        encoder_hidden_states = text_embed
                        mask = torch.zeros([noisy_latents.shape[0], 1, noisy_latents.shape[2], noisy_latents.shape[3], noisy_latents.shape[4]]).to(accelerator.device)
                        latent_model_input = torch.cat([noisy_latents, mask, hint_latent.expand(-1,-1,noisy_latents.shape[2],-1,-1)], dim=1).to(accelerator.device)
                        with accelerator.autocast():
                            noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
                        val_pred_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
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
