
from diffusers import AutoencoderKL 
import numpy as np
from PIL import Image
import torch
import cv2
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to("cuda")

def convert_images_to_video(image_list, output_name, fps):
    img_size = image_list[0].size
    size = (img_size[0], img_size[1])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    out = cv2.VideoWriter(output_name, fourcc, fps, size)
    for i in range(len(image_list)):
        img_np = cv2.cvtColor(np.array(image_list[i]), cv2.COLOR_RGB2BGR)
        out.write(img_np)
    out.release()

tmp_tensor = np.load("/home/ubuntu/10357517.npy")
tmp_tensor = torch.from_numpy(tmp_tensor).to("cuda")
print(tmp_tensor.shape)

tmp_tensor = tmp_tensor.unsqueeze(0)
print(tmp_tensor.shape)
#tmp_tensor = einops.rearrange(tmp_tensor, 'b c f h w -> b f c h w')
print(tmp_tensor.shape)

frame_list = []
frame_count = 570

for frame_index in range(frame_count):
    latents = tmp_tensor[:, frame_index, :, :, :]
    latents = (1 / vae.config.scaling_factor) * latents
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    #print(image)
    frame_list.append(Image.fromarray(image[0]))
    print(frame_index, "/", frame_count)

index = 0
# for img in frame_list:
#     index += 1
#     img.save(f'/workspace/diffused-video-trainer/outimg/{index}_out.png')

convert_images_to_video(frame_list, "animation.mp4", 24)
