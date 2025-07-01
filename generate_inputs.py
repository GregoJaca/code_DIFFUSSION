# generate_inputs.py
"""Generates and saves random Gaussian noise tensors."""
import os
import torch
import argparse
from config import generation_config, system_config

def generate_and_save_noise(num_samples: int, image_size: int, channels: int, output_dir: str):
    """
    Generates N random noise tensors and saves them to disk.
    
    Args:
        num_samples (int): The number of noise tensors to generate.
        image_size (int): The height and width of the image.
        channels (int): The number of image channels.
        output_dir (str): The directory to save the .pt files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_samples} noise inputs...")
    
    for i in range(num_samples):
        noise = torch.randn(
            (1, channels, image_size, image_size),
            generator=torch.Generator().manual_seed(system_config.SEED + i)
        )
        output_path = os.path.join(output_dir, f"noise_{i:04d}.pt")
        torch.save(noise, output_path)
        
    print(f"Successfully saved {num_samples} noise files to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate initial noise for diffusion models.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=generation_config.NUM_SAMPLES,
        help="Number of noise tensors to generate."
    )
    args = parser.parse_args()

    generate_and_save_noise(
        num_samples=args.num_samples,
        image_size=generation_config.IMAGE_SIZE,
        channels=generation_config.IN_CHANNELS,
        output_dir=system_config.INPUT_DIR,
    )