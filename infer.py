from diffusers import StableDiffusionVideoInpaintPipeline, UNetPseudo3DConditionModel
import torch
import imageio

from PIL import Image

model_id = "/workspace/TempoFunk/models/latest_timelapse"
pipe = StableDiffusionVideoInpaintPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    )
#pipe.enable_sequential_cpu_offload()
pipe.to("cuda:3")

# notice this is only the video prompt, must be cloudscape, because I only trained on that, do not mention the cat (if it fades away, add cat prompt)
prompts = ["Dancing Coreography"]

# provide first frame, generated from elsewhere
# prompt = "a portrait of a cat, sitting on top of a tall building under sunset clouds"
init_image = Image.open("/workspace/TempoFunk/dancer.png").convert("RGB").resize((512, 512))
# provide first frame as a whole, or you could use custom mask, it is also supported to do inpaint while making video
mask_image = Image.new("L", (512,512), 0).convert("RGB")

counter_i = 0
for p in prompts:
    for i in range(100):
        images = pipe(p, image=init_image, mask_image=mask_image, num_inference_steps=100, guidance_scale=12.0, frames_length=110).images
        counter_j = 0
        imageio.mimsave('output_' + str(counter_i) + ".gif", images, fps = 24)
        counter_i += 1