
# tensor_to_image.py
"""Converts a saved tensor sequence into a visual image."""
import os
import torch
import argparse
from torchvision.utils import save_image
import logging
from config import default_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tensor_to_image(tensor_path: str, step: int, output_base_dir: str):
    """
    Loads a tensor, selects a specific step if it's a sequence, and saves it as a PNG image.
    Handles both full sequence tensors and final step tensors.

    Args:
        tensor_path (str): The path to the input tensor .pt file.
        step (int): The generation step to visualize for sequence tensors. -1 for the last step.
                    This argument is ignored for final tensors.
        output_base_dir (str): The base directory to save the output images.
                               Images will be saved in subdirectories like 'sequences'.
    """
    if not os.path.exists(tensor_path):
        logging.error(f"Tensor file not found at: {tensor_path}")
        return

    # Load the tensor
    try:
        loaded_tensor = torch.load(tensor_path, map_location=torch.device('cpu'))
        logging.info(f"Loaded tensor with shape: {loaded_tensor.shape}")
    except Exception as e:
        logging.error(f"Failed to load or process tensor file: {e}")
        return

    base_filename = os.path.splitext(os.path.basename(tensor_path))[0]
    image_tensor = None
    output_filename = ""
    output_subdir = os.path.join(output_base_dir, "sequences") # Both sequence and final tensors go here

    # Determine if it's a sequence tensor (4D) or a final tensor (3D)
    if loaded_tensor.dim() == 4:
        # Handle sequence tensor
        num_steps = loaded_tensor.shape[0]

        # Determine the target step
        if step == -1:
            target_step_idx = num_steps - 1
        elif 0 <= step < num_steps:
            target_step_idx = step
        else:
            logging.error(f"Invalid step '{step}'. Must be between 0 and {num_steps - 1}.")
            return

        image_tensor = loaded_tensor[target_step_idx]
        output_filename = f"{base_filename}_step_{target_step_idx:04d}.png"

    elif loaded_tensor.dim() == 3:
        # Handle final tensor
        image_tensor = loaded_tensor
        # Remove _final from base_filename and append _step_final
        output_filename = f"{base_filename.replace('_final', '')}_step_final.png"

    else:
        logging.error(
            f"Unexpected tensor dimensions. Expected 3 (C, H, W) or 4 (steps, C, H, W), "
            f"but got {loaded_tensor.dim()}. A different script might be needed for this tensor format."
        )
        return

    if image_tensor is None:
        logging.error("Could not extract image tensor.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_subdir, exist_ok=True)

    logging.debug(f"Before normalization: Min={image_tensor.min():.4f}, Max={image_tensor.max():.4f}, Mean={image_tensor.mean():.4f}, Std={image_tensor.std():.4f}")

    # Normalize the tensor from [-1, 1] to [0, 1] for saving
    image_tensor = (image_tensor + 1) / 2.0
    image_tensor = image_tensor.clamp(0, 1) # Ensure values are in [0, 1]

    logging.debug(f"After normalization and clamping: Min={image_tensor.min():.4f}, Max={image_tensor.max():.4f}, Mean={image_tensor.mean():.4f}, Std={image_tensor.std():.4f}")

    output_path = os.path.join(output_subdir, output_filename)

    # Save the image
    try:
        save_image(image_tensor, output_path, format='png')
        logging.info(f"Successfully saved image to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save image: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert tensor(s) to image(s).")
    
    # Create a mutually exclusive group for tensor-path and folder-path
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--tensor-path",
        type=str,
        help="Path to a single input tensor file (e.g., 'outputs/sequences/sample_0000_sequence.pt' or 'outputs/final_tensors/sample_0001_final.pt')."
    )
    group.add_argument(
        "--folder-path",
        type=str,
        help="Path to a folder containing tensor files to process."
    )

    parser.add_argument(
        "--step",
        type=int,
        default=default_config.IMAGE_STEP,
        help="The generation step to use for sequence tensors. Defaults to the last step (-1). Ignored for final tensors."
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default=default_config.OUTPUT_IMAGE_DIR,
        help="Base directory to save the output images. Subdirectories 'sequences' will be created."
    )

    args = parser.parse_args()

    if args.tensor_path:
        tensor_to_image(args.tensor_path, args.step, args.output_base_dir)
    elif args.folder_path:
        if not os.path.isdir(args.folder_path):
            logging.error(f"Folder not found at: {args.folder_path}")
        else:
            for filename in os.listdir(args.folder_path):
                if filename.endswith(".pt"):
                    full_path = os.path.join(args.folder_path, filename)
                    logging.info(f"Processing tensor from folder: {full_path}")
                    tensor_to_image(full_path, args.step, args.output_base_dir)
                else:
                    logging.info(f"Skipping non-.pt file in folder: {filename}")

