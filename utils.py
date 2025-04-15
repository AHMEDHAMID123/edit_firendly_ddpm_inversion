from torchvision.transforms import transforms, ToPILImage
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import yaml
from functools import partial
import os

from PIL import Image, ImageDraw, ImageFont

from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline


class ddpm_config(BaseModel):
    """
    Configuration class for DDPM inversion.

    Attributes:
        base_dir (str): Base directory for saving outputs.
        sd_model (str): Path to the Stable Diffusion model.
        guidance_scale (float): Scale for classifier-free guidance.
        num_inference_steps (int): Number of inference steps.
        image_path (str): Path to the input image.
        seed (int): Random seed for reproducibility.
        dtype (str): Data type for computations ('half' or 'float').
        prompt (str): Text prompt for image generation.
        edit_prompt (str): Text prompt for editing.
        font_size (int): Font size for label images.
    """

    base_dir: str
    sd_model: str
    guidance_scale: float
    num_inference_steps: int
    image_path: str
    seed: int
    dtype: str
    image_path: str
    prompt: str
    edit_prompt: str
    font_size: int


class DDPM_inversion:
    """
    Class for DDPM inversion and image generation.

    Args:
        config (ddpm_config): Configuration object.
        device (torch.device): Device to run computations on.
    """

    def __init__(self, config, device):
        """
        Initialize the DDPM inversion pipeline.

        Args:
            config (ddpm_config): Configuration object.
            device (torch.device): Device to run computations on.
        """
        self.dtype = torch.float16 if config.dtype == "half" else torch.float32
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            config.sd_model,
            torch_dtype=self.dtype,
            text_encoder_3=None,
            tokenizer_3=None,
        ).to(device)
        self.pipeline.enable_model_cpu_offload()
        self.config = config
        self.device = device
        self.vae_decoder = self.pipeline.vae.decode
        self.scaling_factor = self.pipeline.vae.config.scaling_factor
        self.shift_factor = self.pipeline.vae.config.shift_factor
        self.vae_encoder = self.pipeline.vae.encode

    @torch.no_grad()
    def latent_encoder(self, img):
        """
        Encode an image into latent space.

        Args:
            img (PIL.Image.Image or torch.Tensor): Input image.

        Returns:
            torch.Tensor: Latent representation of the image.
        """
        if isinstance(img, Image.Image):
            transform = transforms.ToTensor()
            im_tensor = transform(img)
        else:
            im_tensor = img
        # preprocessing in sd3
        im_tensor = 2.0 * im_tensor - 1.0
        latent_image = self.vae_encoder(
            im_tensor.unsqueeze(0).to(self.dtype).to(self.device)
        )
        latent_model_input = latent_image.latent_dist.sample()
        latent_model_input = (
            latent_model_input - self.shift_factor
        ) * self.scaling_factor
        return latent_model_input

    @torch.no_grad()
    def latent_decoder(
        self,
        latent: torch.tensor,
        img: bool = False,
    ):
        """
        Decode a latent representation back into an image.

        Args:
            latent (torch.Tensor): Latent representation.
            img (bool): Whether to return a PIL image.

        Returns:
            torch.Tensor or PIL.Image.Image: Decoded image.
        """
        latent = (latent / self.scaling_factor) + self.shift_factor

        decoded = self.vae_decoder(latent)
        decoded = (decoded.sample / 2.0 + 0.5).clamp(0, 1)[0]
        if img:
            to_pil = ToPILImage()
            img = to_pil(decoded.squeeze(0))
            return img
        return decoded

    def prepare_time_steps(self, num_inference_steps=10, end_time=1000, shift=3):
        """
        Prepare time steps for the DDPM process.

        Args:
            num_inference_steps (int): Number of inference steps.
            end_time (int): End time for the process.
            shift (int): Shift factor for time step scaling.

        Returns:
            torch.Tensor: Time steps.
        """
        num_training_steps = 1000
        timesteps = torch.linspace(1, end_time, num_training_steps).flip(0)
        sigmas = timesteps / num_training_steps

        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        sigma_min = sigmas[-1].item()
        sigma_max = sigmas[0].item()
        timesteps = torch.linspace(
            num_training_steps * sigma_max,
            num_training_steps * sigma_min,
            num_inference_steps,
        )
        sigmas = timesteps / num_training_steps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        return sigmas

    def rescale_time_step(self, timestep):
        """
        Rescale a time step to the DDPM range.

        Args:
            timestep (float): Time step to rescale.

        Returns:
            float: Rescaled time step.
        """
        x_min = 0.0
        x_max = 1.0
        y_min = 0.0
        y_max = 1000.0
        scaled = (timestep - x_min) * (y_max - y_min) / (x_max - x_min) + y_min
        return scaled

    @torch.no_grad()
    def forward_process(self, image, num_inference_steps):
        """
        Perform the forward process of DDPM.

        Args:
            image (PIL.Image.Image or torch.Tensor): Input image.
            num_inference_steps (int): Number of inference steps.

        Returns:
            torch.Tensor: Forward process results.
        """
        if isinstance(image, torch.Tensor):
            if image.dim() < 4:
                x_0 = image.unsqueeze(0)
        else:
            x_0 = self.latent_encoder(image)
        x_ts = x_0.expand(num_inference_steps, -1, -1, -1).to(x_0.device)
        timesteps = self.prepare_time_steps(num_inference_steps)[:-1].to(x_0.device)
        timesteps = timesteps.view(num_inference_steps, 1, 1, 1).to(x_0.dtype)
        x_ts = (1 - timesteps) * x_ts + timesteps * torch.randn_like(x_ts).to(
            x_0.device
        ).to(x_0.dtype)
        return torch.cat([x_ts, x_0], dim=0).unsqueeze(1)  # xT, .................. X_0

    @torch.no_grad()
    def encode_image(
        self,
        image,
        num_inference_steps: int = 50,
        eta: float = 1.0,
        prompt: str = " ",
        negative_prompt: str = " ",
        guidance_scale: float = 3.5,
    ):
        """
        Encode an image into DDPM latent space.

        Args:
            image (PIL.Image.Image or torch.Tensor): Input image.
            num_inference_steps (int): Number of inference steps.
            eta (float): Noise scale factor.
            prompt (str): Text prompt for guidance.
            negative_prompt (str): Negative text prompt for guidance.
            guidance_scale (float): Scale for classifier-free guidance.

        Returns:
            tuple: Latent representations (xts, zs).
        """
        latents = self.forward_process(
            image,
            num_inference_steps,
        )
        timesteps = (
            self.prepare_time_steps(num_inference_steps=num_inference_steps)[:-1]
            .to(latents.dtype)
            .to(latents.device)
        )
        do_classifier_free_guidance = True if guidance_scale > 0.0 else False
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        xts = latents.clone()
        x_t = latents[0]
        zs = torch.zeros(
            (num_inference_steps, *x_t.shape), device=x_t.device, dtype=x_t.dtype
        )
        x_t = torch.cat([x_t] * 2) if do_classifier_free_guidance else x_t
        for i, t in enumerate(timesteps):  # 1,............, 0
            x_t = latents[i]
            x_t = torch.cat([x_t] * 2) if do_classifier_free_guidance else x_t
            timestep = self.rescale_time_step(t).expand(x_t.shape[0])
            noise_pred = self.pipeline.transformer(
                hidden_states=x_t,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            prev_sigma = timesteps[i + 1] if i < num_inference_steps - 1 else 0  # t-1
            sigma = t

            dt = prev_sigma - sigma
            predicted_x0 = x_t
            direction_pointing_to_xt = (dt) * noise_pred
            mu_xt = predicted_x0 + direction_pointing_to_xt
            prev_xt = (
                torch.cat([latents[i + 1]] * 2)
                if do_classifier_free_guidance
                else latents[i + 1]
            )
            z_t = prev_xt - mu_xt
            x_t = mu_xt + ((z_t))
            if do_classifier_free_guidance:
                z_in, _ = z_t.chunk(2)
                x_in, _ = x_t.chunk(2)
                zs[i] = z_in
                xts[i + 1] = x_in
            else:
                zs[i] = z_t
                xts[i + 1] = x_t

        zs[-1] = torch.zeros_like(zs[-1])
        return xts, zs

    def generate_image(self, prompt):
        """
        Generate an image from a text prompt.

        Args:
            prompt (str): Text prompt for image generation.

        Returns:
            PIL.Image.Image: Generated image.
        """
        return self.pipeline(prompt).images[0]

    @torch.no_grad()
    def decode_image(
        self,
        xts,
        zts,
        num_inference_steps: int = 50,
        strength: float = 1.0,
        eta: float = 1.0,
        prompt: str = " ",
        negative_prompt: str = " ",
        guidance_scale: float = 0.0,
    ):
        """
        Decode latent representations back into an image.

        Args:
            xts (torch.Tensor): Latent representations.
            zts (torch.Tensor): Variance noises.
            num_inference_steps (int): Number of inference steps.
            strength (float): Strength of the decoding process.
            eta (float): Noise scale factor.
            prompt (str): Text prompt for guidance.
            negative_prompt (str): Negative text prompt for guidance.
            guidance_scale (float): Scale for classifier-free guidance.

        Returns:
            PIL.Image.Image: Decoded image.
        """
        start_step = num_inference_steps - int(num_inference_steps * strength)
        timesteps = self.prepare_time_steps(num_inference_steps=num_inference_steps)
        do_classifier_free_guidance = True if guidance_scale > 0.0 else False
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        variance_noises = zts[(start_step):]
        latents = xts[(start_step):]
        x_t = latents[start_step]
        x_t = torch.cat([x_t] * 2) if do_classifier_free_guidance else x_t
        for i, t in enumerate(timesteps[(start_step):-1]):  # 1,....., , 0
            # print(i, t)
            timestep = self.rescale_time_step(t).expand(x_t.shape[0])
            noise_pred = self.pipeline.transformer(
                hidden_states=x_t,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            prev_sigma = timesteps[i + 1] if i < num_inference_steps - 1 else 0  # t-1
            sigma = t

            dt = prev_sigma - sigma
            predicted_x0 = x_t
            direction_pointing_to_xt = (dt) * noise_pred
            x_t = predicted_x0 + direction_pointing_to_xt
            variance_noise = (
                variance_noises[i]
                if i < num_inference_steps - start_step - 1
                else torch.zeros_like(x_t)
            )
            sigma_z = variance_noise
            x_t = x_t + sigma_z

        if do_classifier_free_guidance:
            x_t, _ = x_t.chunk(2)
        return self.latent_decoder(x_t)


def create_label_image(text, image_size, font_size):
    """
    Create a tensor containing a text label image.

    Args:
        text (str): Text to display.
        image_size (tuple): Tuple (C, H, W) of the target image size.
        font_size (int): Size of the font.

    Returns:
        torch.Tensor: Tensor of shape (1, C, H, W).
    """
    transform = transforms.ToTensor()
    C, H, W = image_size
    img = Image.new("RGB" if C == 3 else "L", (W, H), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        # text_bbox = draw.textbbox((0, 0), text, font=font)
        # text_w = text_bbox[2] - text_bbox[0]
        # text_h = text_bbox[3] - text_bbox[1]
        # x = (W - text_w) / 2
        # y = (H - text_h) / 2
    except IOError:
        font = ImageFont.load_default(font_size)
    w = draw.textlength(text, font=font)
    h = font_size
    x = (W - w) / 2
    y = (H - h) / 2

    draw.text((x, y), text, fill="black" if C == 3 else 0, font=font)

    label_tensor = transform(img)
    return label_tensor.unsqueeze(0)


def load_yml_file(config_path: str):
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            raise


def create_image_grid(images, transform, grid_size=None):
    """
    Create and display a grid of images with no empty spots.

    Args:
        images (list): A list of images to include in the grid.
        transform (callable): A transformation function to apply to each image.
        grid_size (tuple, optional): Number of rows and columns in the grid (rows, cols).

    Returns:
        None
    """
    transformed_images = [transform(img) for img in images]

    # Dynamically determine grid size if not provided
    num_images = len(transformed_images)
    if grid_size is None:
        rows = int(num_images**0.5)
        cols = (num_images + rows - 1) // rows
    else:
        rows, cols = grid_size

    # Create the grid
    grid = make_grid(transformed_images, nrow=cols, padding=2)

    # Convert the grid to a PIL image and display
    grid_image = to_pil_image(grid)
    plt.figure(figsize=(cols * 2, rows * 2))
    plt.imshow(grid_image)
    plt.axis("off")
    plt.show()


def normalize_tensor(tensor):
    """
    Normalize a tensor to have zero mean and unit variance.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    return (tensor - mean) / std


def visualize_results(images, fp, font_size):
    """
    Visualize and save a collage of images with labels.

    Args:
        images (dict): Dictionary of images with their names as keys.
        fp (str): File path to save the collage.
        font_size (int): Font size for labels.

    Returns:
        None
    """
    for image in images:
        if isinstance(images[image], Image.Image):
            transform = transforms.ToTensor()
            images[image] = transform(images[image]).unsqueeze(0)
        if (images[image], torch.Tensor):
            if images[image].dim() < 4:
                images[image] = images[image].unsqueeze(0)
    components = [(images[i].to("cpu"), i) for i in images]

    rows = []
    for tensor, name in components:
        B, C, H, W = tensor.shape
        label = create_label_image(name, (C, H, W), font_size=font_size)
        label = label.to(tensor.device)
        row = torch.cat([label, tensor], dim=0)  # Shape: (B+1, C, H, W)
        rows.append(row)

    save_tensor = torch.cat(rows, dim=0)
    torchvision.utils.save_image(
        save_tensor, fp=os.path.join(fp, "collage.png"), nrow=B + 1
    )
