from simple_trainer import custom_encode_latents

custom_encode_latents(
    frames_path="/workspace/disk/webvid/semi/frames",
    text_path="/workspace/disk/webvid/semi/text",
    outpath="/workspace/disk/webvid/processed",
    model="runwayml/stable-diffusion-v1-5",
    gpus=3
)
