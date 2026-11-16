import base64
import hashlib
import threading
import io
import os
import tempfile
import warnings

import runpod
import torch
from PIL import Image, ImageOps
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
Image.MAX_IMAGE_PIXELS = 20_000_000
vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling(
    tile_sample_min_height=256, tile_sample_min_width=256,
    tile_sample_stride_height=192, tile_sample_stride_width=192,
)


def turbo_parameter_name(name):
    name = name.removeprefix('diffusion_model.')
    if name.startswith('blocks.'):
        name = name.replace('.self_attn.', '.attn1.').replace('.cross_attn.', '.attn2.')
        for old, new in [('q','to_q'),('k','to_k'),('v','to_v'),('o','to_out.0')]:
            name = name.replace('.attn1.'+old, '.attn1.'+new).replace('.attn2.'+old, '.attn2.'+new)
        name = name.replace('.ffn.0', '.ffn.net.0.proj').replace('.ffn.2', '.ffn.net.2')
        name = name.replace('.norm3', '.norm2')
    else:
        names = {'head.head':'proj_out', 'patch_embedding':'patch_embedding',
                 'text_embedding.0':'condition_embedder.text_embedder.linear_1',
                 'text_embedding.2':'condition_embedder.text_embedder.linear_2',
                 'time_embedding.0':'condition_embedder.time_embedder.linear_1',
                 'time_embedding.2':'condition_embedder.time_embedder.linear_2',
                 'time_projection.1':'condition_embedder.time_proj'}
        name = names.get(name, name)
    return name


TURBO_REPO = "Kijai/WanVideo_comfy"
TURBO_REVISION = "8260d429d19fd7a72304cad059160b95d843913f"
TURBO_FILE = "LoRAs/Wan22-Turbo/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors"
TURBO_SHA256 = "0ace5244e3d1256f884662c261b017249796cf5b95f05d5ed93cc02a478967b8"
turbo_loaded = False
turbo_failed = False
turbo_diffs = {}
# A pipeline is mutable; serialize adapter changes and inference within this process.
pipeline_lock = threading.Lock()


def base_parameter(path):
    module_name, attribute = path.rsplit(".", 1)
    module = pipe.transformer.get_submodule(module_name)
    if hasattr(module, "get_base_layer"):
        module = module.get_base_layer()
    return getattr(module, attribute)


def load_turbo():
    global turbo_loaded, turbo_diffs
    if turbo_loaded:
        return
    path = hf_hub_download(repo_id=TURBO_REPO, filename=TURBO_FILE, revision=TURBO_REVISION)
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != TURBO_SHA256:
        raise ValueError("Turbo LoRA checksum mismatch")
    weights = load_file(path, device="cpu")
    adapters = {}
    diffs = {}
    for key, tensor in weights.items():
        if key.endswith(".lora_down.weight"):
            name = turbo_parameter_name(key[:-len(".lora_down.weight")])
            adapters["transformer." + name + ".lora_A.weight"] = tensor
        elif key.endswith(".lora_up.weight"):
            name = turbo_parameter_name(key[:-len(".lora_up.weight")])
            adapters["transformer." + name + ".lora_B.weight"] = tensor
        elif key.endswith((".diff", ".diff_b")):
            suffix = ".diff_b" if key.endswith(".diff_b") else ".diff"
            target = turbo_parameter_name(key[:-len(suffix)]) + (".bias" if suffix == ".diff_b" else ".weight")
            parameter = base_parameter(target)
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise ValueError("Turbo correction shape mismatch: " + target)
            diffs[target] = (parameter.detach().cpu().clone(), tensor.float())
        else:
            raise ValueError("Unsupported Turbo tensor: " + key)
    for key, tensor in adapters.items():
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key[:-len(".lora_A.weight")]
        up = adapters.get(prefix + ".lora_B.weight")
        target = base_parameter(prefix.removeprefix("transformer.") + ".weight")
        if up is None or tensor.ndim != 2 or up.ndim != 2 or tensor.shape[0] != up.shape[1] or (up.shape[0], tensor.shape[1]) != tuple(target.shape):
            raise ValueError("Turbo adapter shape mismatch: " + prefix)
    # Keep norm/bias deltas explicitly: native format conversion may discard them.
    pipe.load_lora_weights(adapters, adapter_name="turbo")
    turbo_diffs = diffs
    turbo_loaded = True


def configure_turbo(enabled, strength):
    global turbo_failed
    if turbo_failed:
        raise RuntimeError("A previous Turbo load failed. Restart this worker to clear partial state.")
    if enabled:
        try:
            load_turbo()
        except Exception:
            turbo_failed = True
            raise
        pipe.enable_lora()
        pipe.set_adapters(["turbo"], adapter_weights=[strength])
    elif turbo_loaded:
        pipe.disable_lora()
    if turbo_loaded:
        with torch.no_grad():
            for name, (original, delta) in turbo_diffs.items():
                parameter = base_parameter(name)
                value = original.float() + delta * (strength if enabled else 0.0)
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))



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
        if (width, height) not in ((640, 352), (352, 640), (1280, 704), (704, 1280)):
            return {"error": "Unsupported resolution."}
        prompt = str(p.get("prompt", "")).strip()
        if not 1 <= len(prompt) <= 4000:
            return {"error": "Describe the desired motion in 1–4000 characters."}
        frames = int(p.get("num_frames", 49))
        turbo = p.get("turbo", False)
        strength = float(p.get("turbo_strength", 1.0))
        if not isinstance(turbo, bool) or not 0.0 <= strength <= 1.0 or (turbo and strength == 0.0):
            return {"error": "Invalid Turbo setting."}
        steps = int(p.get("steps", 4 if turbo else 10))
        seed = int(p.get("seed", 42))
        if frames not in (17, 49, 121) or not (steps in (4, 6, 8) if turbo else 10 <= steps <= 50) or not 0 <= seed <= 2147483647:
            return {"error": "Invalid frames, steps or seed."}
    except Exception:
        return {"error": "Invalid image or generation settings."}

    # Fit the entire reference image inside the output; add black padding, never crop.
    image = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS, color="black")
    with pipeline_lock, torch.inference_mode():
        try:
            configure_turbo(turbo, strength)
        except Exception as error:
            # Do not continue with a partially loaded adapter/correction state.
            raise RuntimeError("Turbo initialization failed; restart this worker before retrying: " + str(error)) from error
        video = pipe(
            image=image, prompt=prompt, negative_prompt=str(p.get("negative_prompt", ""))[:4000],
            width=width, height=height, num_frames=frames, num_inference_steps=steps,
            guidance_scale=1.0 if turbo else 5.0, generator=torch.Generator("cpu").manual_seed(seed),
        ).frames[0]
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "video.mp4")
        export_to_video(video, path, fps=24)
        with open(path, "rb") as file:
            video_base64 = base64.b64encode(file.read()).decode("ascii")
    return {"video_base64": video_base64, "mime_type": "video/mp4", "seed": seed, "mode": "i2v", "turbo": turbo, "turbo_strength": strength if turbo else 0.0, "steps": steps}


runpod.serverless.start({"handler": handler})
