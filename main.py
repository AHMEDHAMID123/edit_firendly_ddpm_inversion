from huggingface_hub.hf_api import HfFolder
from PIL import Image
import torch
from diffusers import StableDiffusion3Pipeline
import os
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import utils
from utils import ddpm_config, DDPM_inversion, visualize_results
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    arguemnts = utils.load_yml_file(args.config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("DDPM inversion")
    config = ddpm_config(**arguemnts)
    if config.seed:
        seed = config.seed
        torch.manual_seed(seed)
    image = Image.open(config.image_path)

    diffusion_model = DDPM_inversion(config=config, device=device)
    image = Image.open(config.image_path)
    xts, zts = diffusion_model.encode_image(
        image=image,
        num_inference_steps=config.num_inference_steps,
        prompt=config.prompt,
        guidance_scale=config.guidance_scale,
    )

    edited_x = diffusion_model.decode_image(
        xts,
        zts,
        num_inference_steps=config.num_inference_steps,
        prompt=config.edit_prompt,
        guidance_scale=config.guidance_scale,
    )
    to_pil_image(edited_x).save(os.path.join(config.base_dir, "ddpm_inversion.png"))
    torch.manual_seed(seed)
    generated_image = diffusion_model.generate_image(config.edit_prompt)
    generated_image.save(
        os.path.join(config.base_dir, "generated_image_with_prompt.png")
    )
    images = {
        "org image": image,
        "edited image": edited_x,
        "generated image": generated_image,
    }
    utils.visualize_results(images, fp=config.base_dir, font_size=config.font_size)
    return None


if __name__ == "__main__":
    main()
