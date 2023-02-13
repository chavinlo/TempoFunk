import os
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from diffusers import DDPMScheduler, UNetPseudo3DConditionModel, AutoencoderKL
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

def pil_to_torch(image, device = 'cpu'):
    return (2 * (pil_to_tensor(image).to(dtype=torch.float32, device=device)) / 255) - 1

def torch_to_pil(x):
    return to_pil_image((x + 1) / 2)

class LatentDataset(Dataset):
    def __init__(self, folders_path, text_embed, frames, verbose: bool):
        self.text_embed = text_embed
        self.folders_path = folders_path
        self.foldersmap = sorted(os.listdir(folders_path))
        self.verbose = verbose
        blocksmap = []

        for folder in self.foldersmap:
            tmp = []
            frame_index = 0
            for latentframe in sorted(os.listdir(os.path.join(self.folders_path, folder))):
                tmp.append(os.path.join(self.folders_path, folder, latentframe))
                frame_index = frame_index + 1
                if frame_index >= frames:
                    blocksmap.append(deepcopy(tmp))
                    frame_index = 0
                    tmp = []
        self.blocksmap = blocksmap

        json.dump(blocksmap, open("blocksmap.json","w"))

    def __len__(self):
        return len(self.blocksmap)

    def __getitem__(self, idx):
        blocklist_pointers = self.blocksmap[idx]
        blocklist = []
        text_embed = self.text_embed
        if self.verbose:
            print("Blocklist length:", len(blocklist_pointers))
        for block in blocklist_pointers:
            tensor = torch.load(block)['mean']
            #if self.verbose:
            #    print("Loaded Block:", tensor.shape)
            blocklist.append(tensor)
        video_embed = torch.stack(blocklist)
        if self.verbose:
            print("Stacked Blocklist:", video_embed.shape)
        video_embed = rearrange(video_embed, 'f c h w -> c f h w')
        if self.verbose:
            print("Rearranged Video Embed:", video_embed.shape)
        video_embed.to('cuda:0')
        if self.verbose:
            print("Saving sample video_embed")
            torch.save(video_embed, "/workspace/TempoFunk/video_embed.pt")
        return text_embed, video_embed

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

def custom_encode_latents(path, outpath, model, gpus: int):
    from threading import Thread
    # gpus = 4
    folders: list[str] = os.listdir(path)
    folders.sort()
    os.makedirs(outpath, exist_ok=True)
    vae_list = []
    for i in range(gpus):
        vae = AutoencoderKL.from_pretrained(model, subfolder='vae').to(f'cuda:{i}')
        vae_list.append({
            "gpu_id": i,
            "vae_obj": vae
        })
    folders_chunks = split_list(folders, gpus)
    print(len(folders_chunks))
    print(len(folders_chunks[0]))

    def sepa_engine(chunk: list, vae_dict: dict, og_path: str, outpath: str,):
        vae = vae_dict['vae_obj']
        gpu_id = vae_dict['gpu_id']
        for folder in tqdm(chunk):
            raw_folder_path = os.path.join(og_path, folder)
            for img in os.listdir(raw_folder_path):
                im = Image.open(os.path.join(raw_folder_path, img))
                im = ImageOps.fit(im, (512, 512), centering = (0.5, 0.5))
                with torch.inference_mode():
                    m = pil_to_torch(im, f'cuda:{gpu_id}').unsqueeze(0)
                    m = vae.encode(m).latent_dist
                    os.makedirs(os.path.join(outpath, folder), exist_ok=True)
                    torch.save({ 'mean': m.mean.squeeze().cpu(), 'std': m.std.squeeze().cpu() }, os.path.join(outpath, folder, os.path.splitext(img)[0] + '.pt'))

    for i in range(gpus):
        print("eh?")
        processThread = Thread(target=sepa_engine, args=(folders_chunks[i], vae_list[i], path, outpath,))
        processThread.start()

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

def encode_single_prompt(prompt, model):
    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder='text_encoder').to('cuda:0')
    with torch.inference_mode():
        tokens = tokenizer(
                [ prompt ],
                truncation = True,
                return_overflowing_tokens = False,
                padding = 'max_length',
                return_tensors = 'pt'
        ).input_ids.to('cuda:0')
        y = text_encoder(input_ids = tokens).last_hidden_state
        return y.cuda()

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

def main(epochs: int = 10):
    pretrained_model_name_or_path = '/workspace/TempoFunk/models/MaSDV'
    learning_rate = 5e-6
    gradient_accumulation_steps = 1
    batch_size = 9
    frames_length = 24
    representing_prompt = "Dancing Coreography"
    save_path = "models/"
    txt_embed = encode_single_prompt(representing_prompt, "runwayml/stable-diffusion-v1-5")
    txt_embed = txt_embed.squeeze(0)
    dataloader_verbose = False
    train_dataset = LatentDataset(
        "/workspace/TempoFunk/data/tiktok/latents",
        txt_embed,
        frames_length,
        dataloader_verbose
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    lr_warmup_steps = 300
    unfreeze_all = False
    enable_wandb = True
    save_steps = 300
    world_size = 4

    os.makedirs(save_path, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='fp16',
        log_with='tensorboard',
        logging_dir='logs'
    )

    if accelerator.is_local_main_process:
        if enable_wandb:
            import wandb
            run = wandb.init(project="video_train", name="fin_bs7_fp16_4xA100", mode="online")

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
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,
        lr = learning_rate
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

    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("video_diffusion")

    progress_bar = tqdm(range(max_train_steps))
    global_step = 0
    tqdm.write(str(max_train_steps))
    tqdm.write(str(train_dataloader.__len__()))
    tqdm.write(str(epochs))

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
                        "t_r/s": rank_tensors_per_second,
                        "t_w/s": world_images_per_second,
                        "tensors_seen": tensors_seen
                    }
                    run.log(wandblogs, step=global_step)

            if accelerator.is_local_main_process:
                if (global_step % save_steps) == 0 and (saved_step != global_step) and accelerator.is_local_main_process:
                    saved_unet = accelerator.unwrap_model(unet, keep_fp32_wrapper = False)
                    save_at = f'{save_path}/v6-new/{str(global_step)}.pt'
                    torch.save(saved_unet.state_dict(), save_at)
            if global_step >= max_train_steps:
                break
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        saved_unet = accelerator.unwrap_model(unet, keep_fp32_wrapper = False)
        save_at = f'{save_path}/v6-new/{str(global_step)}.pt'
        torch.save(saved_unet.state_dict(), save_at)
    accelerator.end_training()

if __name__ == "__main__":
    main()