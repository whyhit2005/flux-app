import torch
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from pathlib import Path
import shutil
import sys
import logging
logging.basicConfig(level=logging.INFO)

model_id = "black-forest-labs/FLUX.1-dev" #you can also use `black-forest-labs/FLUX.1-dev`
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
# pipe.enable_sequential_cpu_offload()
# pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()
# pipe.to(torch.float16) # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once

work_dir = Path('flux-loras/wangxj-30/000-chorsks/')
lora_model_dir = work_dir / 'models/checkpoint-1000'
lora_model_dir = Path("trained-flux-lora")
pipe.load_lora_weights(
        str(lora_model_dir), 
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="custom")
pipe.fuse_lora(lora_scale=1.0)
logging.info(f"lora {lora_model_dir} loaded")

prompts = []
with open("prompts/dog1.txt") as pf:
    for line in pf.readlines():
        prompts.append(line.strip())

# outdir = Path("outputs/boy")
outdir = work_dir / 'samples'
outdir = Path("outputs/dog")
if outdir.exists():
    shutil.rmtree(str(outdir))
outdir.mkdir(parents=True, exist_ok=True)
generator = torch.Generator("cuda").manual_seed(42)
for i, prompt in enumerate(prompts):
    logging.info(f'{i} {prompt}')
    images = pipe(
        prompt,
        height = 1024,
        width = 1024,
        guidance_scale=7.0,
        num_images_per_prompt = 3,
        output_type="pil",
        num_inference_steps=40, #use a larger number if you are using [dev]
        generator=generator
    ).images
    for j, image in enumerate(images):
        outfile = outdir.joinpath(f"p{i:02d}-{j:02d}.png")
        image.save(outfile)

del pipe
if torch.cuda.is_available():
    torch.cuda.empty_cache()
