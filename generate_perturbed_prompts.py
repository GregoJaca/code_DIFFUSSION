# generate_perturbed_inputs.py
"""
Generates a base noise tensor and a set of N perturbed tensors.
Each perturbation has a random direction but a constant magnitude (epsilon).
"""
import os
import torch
import argparse

# Import base configurations for consistency
from config import generation_config, system_config

def generate_perturbed_set(
    base_seed: int,
    perturb_seed: int,
    num_perturbations: int,
    epsilon: float,
    image_size: int,
    channels: int,
    output_dir: str,
):
    """
    Generates and saves a base noise tensor and N perturbed versions.

    Args:
        base_seed (int): Seed for the initial base noise tensor.
        perturb_seed (int): Seed for generating the random perturbation directions.
        num_perturbations (int): How many perturbed samples to create from the base.
        epsilon (float): The constant magnitude (L2 norm) of the perturbation.
        image_size (int): The height and width of the image.
        channels (int): The number of image channels.
        output_dir (str): The directory to save the .pt files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Generating Perturbation Set ---")
    print(f"Base Seed: {base_seed}, Perturbation Seed: {perturb_seed}")
    print(f"Number of Perturbations: {num_perturbations}, Epsilon: {epsilon}")

    # 1. Generate the base noise tensor using the base_seed
    base_generator = torch.Generator().manual_seed(base_seed)
    base_noise = torch.randn(
        (1, channels, image_size, image_size),
        generator=base_generator
    )

    # Save the base noise tensor
    base_filename = f"set_{base_seed:04d}_base.pt"
    base_path = os.path.join(output_dir, base_filename)
    torch.save(base_noise, base_path)
    print(f"\nSaved base noise: {base_path}")

    # 2. Generate N perturbed tensors using the perturb_seed
    perturb_generator = torch.Generator().manual_seed(perturb_seed)
    print(f"Generating {num_perturbations} perturbed samples...")

    for i in range(num_perturbations):
        # Create a random direction tensor. Its shape must match the base noise.
        random_direction = torch.randn(
            base_noise.shape,
            generator=perturb_generator
        )

        # Calculate the L2 norm of the direction tensor.
        # Add a small value to prevent division by zero, though it's highly unlikely.
        direction_norm = torch.linalg.norm(random_direction)
        
        # Normalize the direction tensor to have a magnitude of 1 (a unit vector)
        # and scale it by epsilon. This is the final perturbation vector.
        perturbation = random_direction / (direction_norm + 1e-9) * epsilon

        # Add the perturbation to the base noise to get the final tensor
        perturbed_noise = base_noise + perturbation

        # Save the perturbed tensor
        pert_filename = f"set_{base_seed:04d}_pert_{i:03d}.pt"
        pert_path = os.path.join(output_dir, pert_filename)
        torch.save(perturbed_noise, pert_path)

    print(f"\nSuccessfully saved {num_perturbations} perturbed noise files to '{output_dir}'.")
    print("--- Generation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a base noise tensor and N perturbed versions of it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Seed for generating the initial 'base' noise tensor."
    )
    parser.add_argument(
        "--perturb_seed",
        type=int,
        default=69,
        help="Seed for generating the random 'directions' of the perturbations."
    )
    parser.add_argument(
        "--num_perturbations",
        type=int,
        default=10,
        help="Number of perturbed samples to generate from the single base noise."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.7071,
        help="The constant magnitude (L2 norm) of the perturbation vector."
    )

    args = parser.parse_args()

    generate_perturbed_set(
        base_seed=args.base_seed,
        perturb_seed=args.perturb_seed,
        num_perturbations=args.num_perturbations,
        epsilon=args.epsilon,
        image_size=generation_config.IMAGE_SIZE,
        channels=generation_config.IN_CHANNELS,
        output_dir=system_config.INPUT_DIR,
    )