from main import custom_encode_latents

custom_encode_latents(
    frames_path="/workspace/disk/dataset/semi/frames",
    text_path="/workspace/disk/dataset/semi/text",
    outpath="/workspace/disk/dataset/processed",
    model="runwayml/stable-diffusion-v1-5",
    gpus=3
)
