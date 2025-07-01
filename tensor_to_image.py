
# tensor_to_image.py
"""Converts a saved tensor sequence into a visual image."""
import os
import torch
import argparse
from torchvision.utils import save_image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tensor_to_image(tensor_path: str, step: int, output_dir: str):
    """
    Loads a tensor, selects a specific step, and saves it as a PNG image.

    Args:
        tensor_path (str): The path to the input tensor .pt file.
        step (int): The generation step to visualize. -1 for the last step.
        output_dir (str): The directory to save the output image.
    """
    if not os.path.exists(tensor_path):
        logging.error(f"Tensor file not found at: {tensor_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the full sequence tensor
    try:
        sequence_tensor = torch.load(tensor_path, map_location=torch.device('cpu'))
        logging.info(f"Loaded tensor with shape: {sequence_tensor.shape}")
    except Exception as e:
        logging.error(f"Failed to load or process tensor file: {e}")
        return

    # Validate tensor dimensions (expected: [steps, channels, height, width])
    if sequence_tensor.dim() != 4:
        logging.error(
            f"Unexpected tensor dimensions. Expected 4 (steps, C, H, W), "
            f"but got {sequence_tensor.dim()}. A different script might be needed for this tensor format."
        )
        return

    num_steps = sequence_tensor.shape[0]

    # Determine the target step
    if step == -1:
        target_step_idx = num_steps - 1
    elif 0 <= step < num_steps:
        target_step_idx = step
    else:
        logging.error(f"Invalid step '{step}'. Must be between 0 and {num_steps - 1}.")
        return

    # Select the image tensor for the chosen step
    image_tensor = sequence_tensor[target_step_idx]

    # Normalize the tensor from [-1, 1] to [0, 1] for saving
    # The DDPM pipeline output is in this range.
    image_tensor = (image_tensor + 1) / 2.0
    image_tensor = image_tensor.clamp(0, 1) # Ensure values are in [0, 1]

    # Generate a sensible output filename
    base_filename = os.path.splitext(os.path.basename(tensor_path))[0]
    output_filename = f"{base_filename}_step_{target_step_idx:04d}.png"
    output_path = os.path.join(output_dir, output_filename)

    # Save the image
    try:
        save_image(image_tensor, output_path, format='png')
        logging.info(f"Successfully saved image to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save image: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a tensor sequence to an image.")
    parser.add_argument(
        "--tensor-path",
        type=str,
        required=True,
        help="Path to the input tensor file (e.g., 'outputs/sequences/sample_0000_sequence.pt')."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=-1,
        help="The generation step to use for the image. Defaults to the last step (-1)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("outputs", "images", "sequences"),
        help="Directory to save the output image."
    )
    # The 'quality' flag from the prompt is best handled by choosing a file format.
    # PNG is lossless, so it represents the highest quality.
    # A quality flag is typically for lossy formats like JPEG.

    args = parser.parse_args()

    tensor_to_image(args.tensor_path, args.step, args.output_dir)
