import os
import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm.auto import tqdm
from PIL import Image, ImageOps

from torchvision.transforms.functional import pil_to_tensor, to_pil_image
def torch_to_pil(x):
    return to_pil_image((x + 1) / 2)

def pil_to_torch(image, device = 'cpu'):
    return (2 * (pil_to_tensor(image).to(dtype=torch.float32, device=device)) / 255) - 1

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