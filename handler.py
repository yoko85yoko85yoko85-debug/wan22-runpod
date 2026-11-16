import base64
import io
import os
import tempfile
import warnings

import runpod
import torch
from PIL import Image, ImageOps
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
Image.MAX_IMAGE_PIXELS = 20_000_000
vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling(
    tile_sample_min_height=256, tile_sample_min_width=256,
    tile_sample_stride_height=192, tile_sample_stride_width=192,
)


def handler(job):
    p = job.get("input", {})
    encoded = p.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        return {"error": "I2V worker: image_base64 is required. Update the worker before using the I2V app."}
    if len(encoded) > 4_000_000:
        return {"error": "Image exceeds the 3 MB decoded limit."}
    try:
        raw = base64.b64decode(encoded, validate=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.format not in ("JPEG", "PNG", "WEBP"):
                    return {"error": "Use a JPEG, PNG or WebP image."}
                image = ImageOps.exif_transpose(opened).convert("RGB")
        width = int(p.get("width", 1280))
        height = int(p.get("height", 704))
        if (width, height) not in ((1280, 704), (704, 1280)):
            return {"error": "Unsupported resolution."}
        prompt = str(p.get("prompt", "")).strip()
        if not 1 <= len(prompt) <= 4000:
            return {"error": "Describe the desired motion in 1–4000 characters."}
        frames = int(p.get("num_frames", 49))
        steps = int(p.get("steps", 10))
        seed = int(p.get("seed", 42))
        if frames not in (17, 49, 121) or not 10 <= steps <= 50 or not 0 <= seed <= 2147483647:
            return {"error": "Invalid frames, steps or seed."}
    except Exception:
        return {"error": "Invalid image or generation settings."}

    # Fit the entire reference image inside the output; add black padding, never crop.
    image = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS, color="black")
    with torch.inference_mode():
        video = pipe(
            image=image, prompt=prompt, negative_prompt=str(p.get("negative_prompt", ""))[:4000],
            width=width, height=height, num_frames=frames, num_inference_steps=steps,
            guidance_scale=5.0, generator=torch.Generator("cpu").manual_seed(seed),
        ).frames[0]
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "video.mp4")
        export_to_video(video, path, fps=24)
        with open(path, "rb") as file:
            video_base64 = base64.b64encode(file.read()).decode("ascii")
    return {"video_base64": video_base64, "mime_type": "video/mp4", "seed": seed, "mode": "i2v"}


runpod.serverless.start({"handler": handler})
