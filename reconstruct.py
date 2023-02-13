import diffusers
import imageio
import numpy as np
from PIL import Image
import torch
import einops
vae = diffusers.StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
    ).to("cuda").vae

tmp_tensor = torch.load("/workspace/TempoFunk/video_embed.pt").to("cuda")
print(tmp_tensor.shape)

tmp_tensor = tmp_tensor.unsqueeze(0)
tmp_tensor = einops.rearrange(tmp_tensor, 'b c f h w -> b f c h w')

frame_list = []
frame_count = 24

for frame_index in range(frame_count):
    a_frame = tmp_tensor[:, frame_index, :, :, :]
    latents = a_frame
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    frame_list.append(pil_images[0])
    print(frame_index, "/", frame_count)

index = 0
# for img in frame_list:
#     index += 1
#     img.save(f'/workspace/diffused-video-trainer/outimg/{index}_out.png')

imageio.mimsave('animation.gif', [np.array(img) for img in frame_list], 'GIF', fps=24)