from diffusers import StableDiffusionVideoPipeline
import imageio
import os
from PIL import Image
import torch

output_dir = "samples"
gpus = 3
os.makedirs(output_dir, exist_ok=True)
model_id = f"/workspace/disk/models/latest"

def infe_engine(gpu_index: int):
    pipe = StableDiffusionVideoPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        )
    #pipe.enable_sequential_cpu_offload()
    print("USING GPU:", gpu_index)
    pipe.to(f"cuda:{gpu_index}")

    # notice this is only the video prompt, must be cloudscape, because I only trained on that, do not mention the cat (if it fades away, add cat prompt)
    prompt = '"A man dancing in the middle of the scene, coreography, beautiful, motion, adequate lighting, professional"'
    negative_prompt='"Disgusting, Bad lighting, Too bright, disco lights, Bad quality, Shaky"'


    counter_i = 0
    mult = 1
    mult_2 = 1
    for i in range(100):
        if i % 2:
            mult = mult + 1
        if mult > 8:
            mult = 0
            mult_2 = mult_2 + 1
        infer_steps = 75+(mult*25)
        infer_cfg = 9.0+(1*mult_2)
        images = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=infer_steps, guidance_scale=infer_cfg, frames_length=22).images
        counter_j = 0
        imageio.mimsave(os.path.join(output_dir, f'out_{i}_{gpu_index}_steps{infer_steps}_cfg{infer_cfg}.gif'), images, fps = 22)
        counter_i += 1

from threading import Thread

for gpu_index in range(gpus):
    print("STATING FOR GPU:", gpu_index)
    processThread = Thread(target=infe_engine, args=(gpu_index,))
    processThread.start()
