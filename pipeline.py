import argparse
import subprocess
import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import system_config, default_config

def run_command(command, description):
    print(f"\n--- {description} ---")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"Error: {description} failed.")
        print(f"Stdout: {process.stdout}")
        print(f"Stderr: {process.stderr}")
        exit(1)
    else:
        print(f"Stdout: {process.stdout}")
        print(f"Stderr: {process.stderr}")

def visualize_classifications(
    num_prompts_per_direction: tuple[int, int],
    output_dir: str,
    results_dir: str,
    colors: list[tuple[int, int, int]],
    linspace_0: torch.Tensor,
    linspace_1: torch.Tensor,
):
    print("\n--- Visualizing Classifications ---")
    classifications_plane_path = os.path.join(output_dir, "classifications_plane.pt")
    if not os.path.exists(classifications_plane_path):
        print(f"Error: Classification plane tensor not found at {classifications_plane_path}")
        exit(1)

    classifications_plane = torch.load(classifications_plane_path)

    os.makedirs(results_dir, exist_ok=True)
    output_tensor_path = os.path.join(results_dir, "classification_plane.pt")
    torch.save(classifications_plane, output_tensor_path)
    print(f"Classification tensor saved to {output_tensor_path}")

    num_x, num_y = num_prompts_per_direction

    # Convert RGB colors to 0-1 range for matplotlib
    norm_colors = [(r/255, g/255, b/255) for r, g, b in colors]
    cmap = ListedColormap(norm_colors)

    plt.figure(figsize=(max(num_x, 12), max(num_y, 12)))
    plt.imshow(classifications_plane.T, cmap=cmap, origin='lower',
               extent=[linspace_0.min(), linspace_0.max(), linspace_1.min(), linspace_1.max()],
               interpolation='nearest')

    plt.xticks(linspace_0.tolist(), rotation=45, ha="right")
    plt.yticks(linspace_1.tolist())

    plt.xlabel("Direction 0", fontsize=14)
    plt.ylabel("Direction 1", fontsize=14)
    plt.title("Classification Plane Visualization", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Create a colorbar with discrete values
    cbar = plt.colorbar(ticks=range(len(colors)))
    cbar.set_label("Classification", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    output_image_path = os.path.join(results_dir, "classification_plane_visualization.png")
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close() # Close the plot to free memory
    print(f"Visualization saved to {output_image_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline for generating, processing, classifying, and visualizing noise tensors.",
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
        "--generate_png",
        action="store_true",
        help="Generate PNG images from tensors (optional)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=system_config.OUTPUT_DIR,
        help="The base output directory."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save visualization results."
    )

    args = parser.parse_args()

    # Step 1: Generate plane prompts
    generate_plane_cmd = (
        f"python generate_plane_prompts.py "
        f"--base_seed {args.base_seed} "
        f"--direction_seeds {args.direction_seeds[0]} {args.direction_seeds[1]} "
        f"--num_prompts_per_direction {args.num_prompts_per_direction[0]} {args.num_prompts_per_direction[1]} "
        f"--len_per_direction {args.len_per_direction[0]} {args.len_per_direction[1]} "
        # f"--output_dir {system_config.INPUT_DIR}"
    )
    run_command(generate_plane_cmd, "Generating Plane Prompts")

    # Step 2: Run main.py to generate image .pt files
    main_cmd = f"python main.py --no-extract_states"
    run_command(main_cmd, "Running Main.py (Image Generation)")

    # Step 3: Optional - Generate PNG images
    if args.generate_png:
        tensor_to_image_cmd = f"python tensor_to_image.py --folder-path \"{system_config.OUTPUT_DIR}/final_tensors\""
        run_command(tensor_to_image_cmd, "Generating PNG Images")

    # Step 4: Classify generated tensors
    classify_cmd = (
        f"python classify.py "
        f"--tensors_path {system_config.OUTPUT_DIR}/final_tensors "
        f"--num_prompts_per_direction {args.num_prompts_per_direction[0]} {args.num_prompts_per_direction[1]}"
    )
    run_command(classify_cmd, "Classifying Tensors")

    # Step 5: Visualize classifications
    # linspace_path = os.path.join(system_config.INPUT_DIR, "linspace_values.pt")
    linspace_path = "linspace_values.pt"
    if not os.path.exists(linspace_path):
        print(f"Error: Linspace values not found at {linspace_path}")
        exit(1)
    linspace_data = torch.load(linspace_path)

    visualize_classifications(
        num_prompts_per_direction=tuple(args.num_prompts_per_direction),
        output_dir=f"{system_config.OUTPUT_DIR}/final_tensors",
        results_dir= f"{system_config.OUTPUT_DIR}/images",# args.results_dir,
        colors=system_config.CLASSIFICATION_COLORS,
        linspace_0=linspace_data["linspace_0"],
        linspace_1=linspace_data["linspace_1"],
    )

if __name__ == "__main__":
    main()
