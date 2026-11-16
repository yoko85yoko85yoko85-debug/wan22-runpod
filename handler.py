import base64
import os
import tempfile

import runpod
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

vae = AutoencoderKLWan.from_pretrained(
    MODEL,
    subfolder="vae",
    torch_dtype=torch.float32,
)

pipe = WanPipeline.from_pretrained(
    MODEL,
    vae=vae,
    torch_dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()


def handler(job):
    params = job.get("input", {})
    prompt = str(params.get("prompt", "")).strip()

    if not prompt or len(prompt) > 4000:
        return {"error": "Prompt must contain 1–4000 characters."}

    width = int(params.get("width", 1280))
    height = int(params.get("height", 704))

    if (width, height) not in [(1280, 704), (704, 1280)]:
        return {"error": "Unsupported resolution."}

    frames = min(121, max(17, int(params.get("num_frames", 121))))
    frames = ((frames - 1) // 4) * 4 + 1

    steps = min(50, max(10, int(params.get("steps", 30))))
    seed = int(params.get("seed", 42))

    with torch.inference_mode():
        video = pipe(
            prompt=prompt,
            negative_prompt=str(params.get("negative_prompt", "")),
            width=width,
            height=height,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=5.0,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).frames[0]

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "video.mp4")
        export_to_video(video, path, fps=24)

        with open(path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("ascii")

    return {
        "video_base64": encoded,
        "mime_type": "video/mp4",
        "seed": seed,
    }


runpod.serverless.start({"handler": handler})
