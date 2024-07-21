# noise.py demonstrates how to generate noise images from a given image using the DDPMSampler.

from ddpm import DDPMSampler
from PIL import Image
import torch
import numpy as np
import math
from pathlib import Path

generator = torch.Generator()
generator.manual_seed(0)

ddpm_sampler = DDPMSampler(generator)

# How many noise levels to generate
noise_levels = [0, 10, 50, 75, 100, 250, 500, 750]

img = Image.open("../img/dog.jpg")
img_tensor = torch.tensor(np.array(img))
img_tensor = ((img_tensor / 255.0) * 2.0) - 1.0
# Create a batch by repeating the same image many times
batch = img_tensor.repeat(len(noise_levels), 1, 1, 1)

ts = torch.tensor(noise_levels, dtype=torch.int, device=batch.device)
noise_imgs = []
epsilons = torch.randn(batch.shape, device=batch.device)
# Generate a noisifed version of the image for each noise level
for i in range(len(ts)):
    a_hat = ddpm_sampler.alphas_cumprod[ts[i]]
    noise_imgs.append(
        (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
    )

noise_imgs = torch.stack(noise_imgs, dim=0)
noise_imgs = (noise_imgs.clamp(-1, 1) + 1) / 2
noise_imgs = (noise_imgs * 255).type(torch.uint8)

# Create the output directory if it doesn't exist
output_dir = Path("../img/results/noise")
output_dir.mkdir(parents=True, exist_ok=True)

# Save the output images
for idx, noise_img in enumerate(noise_imgs):
    output_file = output_dir / f"noise_level_{noise_levels[idx]}.png"
    display_img = Image.fromarray(noise_img.squeeze(0).numpy(), 'RGB')
    display_img.save(output_file)
    print(f"Saved noise image with noise level {noise_levels[idx]} to {output_file}")

# Display the last noise image in the notebook
display_img.show()