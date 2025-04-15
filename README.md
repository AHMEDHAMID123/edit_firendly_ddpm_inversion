# Denoising Diffusion Implicit Models

This repository contains an implementation of An Edit Friendly DDPM Noise Space: Inversion and Manipulations for image generation and manipulation. It edits the paper implementation to be compitable with flow matching models. It uses the Stable Diffusion3/3.5 models which are a rectified flow formulations which assumes that flow between the data and the noise follows a straight line trajectory, this line is actully the solution for the optimal transport map between the noise and data distributions, and provides tools for encoding, decoding, and generating images using diffusion-based techniques.


### Usage

1. **Set Up Configuration**:
   Create a YAML configuration file or use the `ddpm_config` class to define the parameters for the pipeline.

2. **Initialize the Pipeline**:
   ```python
   from utils import ddpm_config, DDPM_inversion
   import torch

   config = ddpm_config(
       base_dir="output",
       sd_model="path/to/stable-diffusion-model",
       guidance_scale=7.5,
       num_inference_steps=50,
       image_path="path/to/image.png",
       seed=42,
       dtype="float",
       prompt="A beautiful landscape",
       edit_prompt="",
       font_size=20,
   )
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ddpm = DDPM_inversion(config, device)
   ```

3. **Generate an Image**:
   ```python
   generated_image = ddpm.generate_image(prompt="A futuristic cityscape")
   generated_image.show()
   ```

4. **Encode and Decode an Image**:
   ```python
   from PIL import Image

   input_image = Image.open("path/to/image.png")
   latents = ddpm.latent_encoder(input_image)
   decoded_image = ddpm.latent_decoder(latents, img=True)
   decoded_image.show()
   ```

5. **Visualize Results**:
   ```python
   images = {"Original": input_image, "Decoded": decoded_image}
   ddpm.visualize_results(images, fp="output", font_size=20)
   ```

## Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [An Edit Friendly DDPM Noise Space: Inversion and Manipulations] (https://arxiv.org/abs/2304.06140)