from diffusers import StableDiffusionVideoInpaintPipeline, UNetPseudo3DConditionModel
import torch
import imageio
import os
from PIL import Image

output_dir = "samples"
gpus = 3
step_point = 6600
os.makedirs(output_dir, exist_ok=True)
model_id = f"/workspace/TempoFunk/models/{step_point}"


def infe_engine(gpu_index: int):
    pipe = StableDiffusionVideoInpaintPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        )
    #pipe.enable_sequential_cpu_offload()
    print("USING GPU:", gpu_index)
    pipe.to(f"cuda:{gpu_index}")

    # notice this is only the video prompt, must be cloudscape, because I only trained on that, do not mention the cat (if it fades away, add cat prompt)
    prompt = "Dance, dancing woman, slow performance, beautiful, colorful, man standing in the middle of the picture, moving his hands lightly"
    neg_prompt = "festival, black, background, disco, night, ballet, horrible, disgusting, nonsense, disturbed, corrupted, disruption, disembodied, bad anatomy"
    # provide first frame, generated from elsewhere
    # prompt = "a portrait of a cat, sitting on top of a tall building under sunset clouds"
    init_image = Image.open("/workspace/TempoFunk/dancer.png").convert("RGB").resize((512, 512))
    # provide first frame as a whole, or you could use custom mask, it is also supported to do inpaint while making video
    mask_image = Image.new("L", (512,512), 0).convert("RGB")

    counter_i = 0
    mult = 1
    mult_2 = 1
    for i in range(100):
        if i % 2:
            mult = mult + 1
        if mult > 8:
            mult = 0
            mult_2 = mult_2 + 1
        infer_steps = 50+(mult*25)
        infer_cfg = 9.0+(1*mult_2)
        images = pipe(prompt, negative_prompt=neg_prompt, image=init_image, mask_image=mask_image, num_inference_steps=infer_steps, guidance_scale=infer_cfg, frames_length=110).images
        counter_j = 0
        imageio.mimsave(os.path.join(output_dir, f'{step_point}_{i}_{gpu_index}_steps{infer_steps}_cfg{infer_cfg}.gif'), images, fps = 24)
        counter_i += 1

from threading import Thread

for gpu_index in range(gpus):
    print("STATING FOR GPU:", gpu_index)
    processThread = Thread(target=infe_engine, args=(gpu_index,))
    processThread.start()
