import os
import torch
import argparse

# Import base configurations for consistency
from config import generation_config, system_config, default_config

def generate_plane_set(
    base_seed: int,
    direction_seeds: tuple[int, int],
    num_prompts_per_direction: tuple[int, int],
    len_per_direction: tuple[float, float],
    center_coords: tuple[float, float],
    image_size: int,
    channels: int,
    output_dir: str,
):
    """
    Generates and saves a plane of noise tensors.

    Args:
        base_seed (int): Seed for the initial base noise tensor.
        direction_seeds (tuple of int): Seeds for the two direction vectors.
        num_prompts_per_direction (tuple of int): Number of prompts along each direction.
        len_per_direction (tuple of float): Length of the plane along each direction.
        image_size (int): The height and width of the image.
        channels (int): The number of image channels.
        output_dir (str): The directory to save the .pt files.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Generating Plane Set ---")
    print(f"Base Seed: {base_seed}")
    print(f"Direction Seeds: {direction_seeds}")
    print(f"Number of Prompts per Direction: {num_prompts_per_direction}")
    print(f"Length per Direction: {len_per_direction}")
    print(f"Center Coordinates: {center_coords}")

    # 1. Generate the base noise tensor using the base_seed
    base_generator = torch.Generator().manual_seed(base_seed)
    base_noise = torch.randn(
        (1, channels, image_size, image_size),
        generator=base_generator
    )

    # Save the base noise tensor only if it's not redundant with a plane sample
    # It's redundant if both num_prompts_per_direction are odd, as this means
    # the (0,0) point of the plane will be generated, which is the base noise.
    # if num_prompts_per_direction[0] % 2 == 1 and num_prompts_per_direction[1] % 2 == 1:
    #     print(f"\nSkipping saving base noise as it's redundant with a plane sample (num_prompts_per_direction is {num_prompts_per_direction}).")
    # else:
    #     base_filename = f"plane_{base_seed:04d}_base.pt"
    #     base_filename = f"plane_{base_seed:04d}_{direction_seeds[0]:03d}_{direction_seeds[1]:03d}_x000_y000.pt"

    #     base_path = os.path.join(output_dir, base_filename)
    #     torch.save(base_noise, base_path)
    #     print(f"\nSaved base noise: {base_path}")

    # 2. Generate two direction vectors
    direction_generator_0 = torch.Generator().manual_seed(direction_seeds[0])
    direction_vector_0 = torch.randn(
        base_noise.shape,
        generator=direction_generator_0
    )
    direction_vector_0 = direction_vector_0 / torch.linalg.norm(direction_vector_0) # Normalize

    direction_generator_1 = torch.Generator().manual_seed(direction_seeds[1])
    direction_vector_1 = torch.randn(
        base_noise.shape,
        generator=direction_generator_1
    )
    direction_vector_1 = direction_vector_1 / torch.linalg.norm(direction_vector_1) # Normalize

    # Add the center coordinates to the base noise
    base_noise = base_noise + (center_coords[0] * direction_vector_0) + (center_coords[1] * direction_vector_1)

    print(f"Generating {num_prompts_per_direction[0]}x{num_prompts_per_direction[1]} plane samples...")

    # 3. Create the plane by doing a linspace with these two directions centered at base_noise
    linspace_0 = torch.linspace(
        -len_per_direction[0] / 2,
        len_per_direction[0] / 2,
        num_prompts_per_direction[0]
    )
    linspace_1 = torch.linspace(
        -len_per_direction[1] / 2,
        len_per_direction[1] / 2,
        num_prompts_per_direction[1]
    )

    for i, val_0 in enumerate(linspace_0):
        for j, val_1 in enumerate(linspace_1):
            plane_noise = base_noise + (val_0 * direction_vector_0) + (val_1 * direction_vector_1)

            filename = f"plane_{base_seed:04d}_{direction_seeds[0]:03d}_{direction_seeds[1]:03d}_x{i:03d}_y{j:03d}.pt"
            file_path = os.path.join(output_dir, filename)
            torch.save(plane_noise, file_path)
            print(f"  Saved plane sample: {filename}")

    print(f"Successfully saved plane noise files to '{output_dir}'.")

    # Save linspace values
    linspace_data = {
        "linspace_0": linspace_0,
        "linspace_1": linspace_1
    }
    # linspace_path = os.path.join(output_dir, "linspace_values.pt")
    linspace_path = "linspace_values.pt"
    torch.save(linspace_data, linspace_path)
    print(f"Saved linspace values to {linspace_path}")

    print("--- Generation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a plane of noise tensors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=default_config.BASE_SEED,
        help="Seed for generating the initial 'base' noise tensor."
    )
    parser.add_argument(
        "--direction_seeds",
        type=int,
        nargs=2,
        default=default_config.DIRECTION_SEEDS,
        help="Two seeds for generating the two direction vectors."
    )
    parser.add_argument(
        "--num_prompts_per_direction",
        type=int,
        nargs=2,
        default=default_config.NUM_PROMPTS_PER_DIRECTION,
        help="Number of prompts along each direction (e.g., 5 5 for a 5x5 grid)."
    )
    parser.add_argument(
        "--len_per_direction",
        type=float,
        nargs=2,
        default=default_config.LEN_PER_DIRECTION,
        help="Length of the plane along each direction (e.g., 2.0 2.0)."
    )
    parser.add_argument(
        "--center_coords",
        type=float,
        nargs=2,
        default=default_config.CENTER_COORDS,
        help="Center coordinates (x0, y0) of the plane."
    )

    args = parser.parse_args()

    generate_plane_set(
        base_seed=args.base_seed,
        direction_seeds=tuple(args.direction_seeds),
        num_prompts_per_direction=tuple(args.num_prompts_per_direction),
        len_per_direction=tuple(args.len_per_direction),
        center_coords=tuple(args.center_coords),
        image_size=generation_config.IMAGE_SIZE,
        channels=generation_config.IN_CHANNELS,
        output_dir=system_config.INPUT_DIR,
    )