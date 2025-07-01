# main.py
"""Main execution script for the Diffusion Model Hidden State Extraction System."""
import os
import torch
import argparse
import time
import logging
from tqdm import tqdm
from typing import List

import random
import numpy as np

from config import system_config, model_config, inference_config, default_config
from extraction_system import StateExtractor

# Setup logging (will be configured based on args later)

def setup_directories():
    """Create all necessary output directories."""
    os.makedirs(os.path.join(system_config.OUTPUT_DIR, "sequences"), exist_ok=True)
    os.makedirs(os.path.join(system_config.OUTPUT_DIR, "layer1"), exist_ok=True)
    os.makedirs(os.path.join(system_config.OUTPUT_DIR, "last_layer"), exist_ok=True)
    os.makedirs(os.path.join(system_config.OUTPUT_DIR, "final_tensors"), exist_ok=True)
    logging.info("Output directories created.")

def save_tensors(
    input_file_bases: List[str],
    sequence_batch: torch.Tensor,
    states_batch: dict,
):
    """Saves the extracted tensors for a batch with clear naming."""
    base_path = system_config.OUTPUT_DIR
    
    # Unbind batch dimension and save each sample individually
    for i, file_base in enumerate(input_file_bases):
        # Save full sequence
        seq_path = os.path.join(base_path, "sequences", f"{file_base}_sequence.pt")
        torch.save(sequence_batch[i], seq_path)
        
        # Save layer 1 trajectory
        l1_path = os.path.join(base_path, "layer1", f"{file_base}_trajectory.pt")
        torch.save(states_batch["layer1"][i], l1_path)

        # Save last layer trajectory
        ll_path = os.path.join(base_path, "last_layer", f"{file_base}_trajectory.pt")
        torch.save(states_batch["last_layer"][i], ll_path)

def main(args):
    """Main function to run the extraction process."""
    # Configure logging based on debug argument
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    import random
    import numpy as np
    random.seed(system_config.SEED)
    np.random.seed(system_config.SEED)
    if system_config.DEVICE == "cuda":
        if not torch.cuda.is_available():
            logging.warning("CUDA not available. Falling back to CPU.")
            system_config.DEVICE = "cpu"
        # The CUDA determinism settings are now applied per-batch in the loop
        # else:
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False
        #     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        #     torch.use_deterministic_algorithms(True)
    
    if system_config.DEVICE == "cpu" and torch.cuda.is_available():
        logging.info("CUDA is available but device is set to CPU. Using CPU.")
        logging.warning("CUDA not available. Falling back to CPU.")
        system_config.DEVICE = "cpu"
    logging.info(f"Using device: {system_config.DEVICE}")

    setup_directories()

    # Load extractor
    extractor = StateExtractor(
        model_id=model_config.MODEL_ID,
        device=system_config.DEVICE,
        first_layer_name=model_config.FIRST_LAYER_NAME,
        last_layer_name=model_config.LAST_LAYER_NAME
    )

    # Find input files
    input_files = sorted([os.path.join(system_config.INPUT_DIR, f) for f in os.listdir(system_config.INPUT_DIR) if f.endswith(".pt")])
    if not input_files:
        logging.error(f"No input noise files found in '{system_config.INPUT_DIR}'.")
        logging.error("Please run 'python generate_inputs.py' first.")
        return

    # Process in batches
    num_batches = (len(input_files) + args.batch_size - 1) // args.batch_size
    
    total_time = 0
    
    with tqdm(total=len(input_files), desc="Extracting States") as pbar:
        for i in range(num_batches):
            start_idx = i * args.batch_size
            end_idx = min((i + 1) * args.batch_size, len(input_files))
            
            batch_files = input_files[start_idx:end_idx]
            if not batch_files:
                continue


            # Load batch of noise tensors
            noise_batch = torch.cat([torch.load(f) for f in batch_files], dim=0)
            input_file_bases = [os.path.splitext(os.path.basename(f))[0] for f in batch_files]

            logging.debug(f"Pre-extractor: noise_batch.shape = {noise_batch.shape}, dtype = {noise_batch.dtype}, device = {noise_batch.device}")
            logging.debug(f"Pre-extractor: noise_batch_mean={noise_batch.mean():.6f}, noise_batch_std={noise_batch.std():.6f}")

            # Reset seeds for each batch to ensure determinism
            random.seed(system_config.SEED)
            np.random.seed(system_config.SEED)
            torch.manual_seed(system_config.SEED)
            if system_config.DEVICE == "cuda":
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.use_deterministic_algorithms(True)

            # Move input batch to the correct device (GPU if available)
            noise_batch = noise_batch.to(system_config.DEVICE)

            # Debug: Print input tensor details before model call (after device transfer)
            logging.debug(f"Batch {i+1}: noise_batch.shape = {noise_batch.shape}, dtype = {noise_batch.dtype}, device = {noise_batch.device}")
            logging.debug(f"Batch {i+1}: Any NaNs in noise_batch? {torch.isnan(noise_batch).any().item()}")
            logging.debug(f"Batch {i+1}: Any Infs in noise_batch? {torch.isinf(noise_batch).any().item()}")

            logging.info(f"Processing batch {i+1}/{num_batches}...")

            start_mem = torch.cuda.memory_allocated(system_config.DEVICE) if system_config.DEVICE == "cuda" else 0
            batch_start_time = time.time()

            # Run extraction or generation
            if args.extract_states:
                full_sequence, extracted_states = extractor.run_extraction(
                    initial_noise=noise_batch,
                    num_steps=args.num_steps,
                    flatten_output=args.flatten
                )
                save_tensors(input_file_bases, full_sequence, extracted_states)
                logging.info(f"Saved results for batch {i+1}.")
            else:
                final_images = extractor.generate_final_image(
                    initial_noise=noise_batch,
                    num_steps=args.num_steps
                )
                # Save final images
                for j, file_base in enumerate(input_file_bases):
                    img_path = os.path.join(system_config.OUTPUT_DIR, "final_tensors", f"{file_base}.pt")
                    torch.save(final_images[j], img_path)
                    logging.debug(f"saved {file_base}_final_tensor.pt")
                logging.info(f"Saved final images for batch {i+1}.")

            pbar.update(len(batch_files))

    logging.info("="*50)
    logging.info("Extraction process complete.")
    logging.info(f"Total samples processed: {len(input_files)}")
    logging.info(f"Total time elapsed: {total_time:.2f} seconds.")
    logging.info(f"Average time per sample: {total_time / len(input_files):.2f} seconds.")
    logging.info(f"Outputs saved to '{system_config.OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hidden states from a diffusion model.")
    parser.add_argument("--batch_size", type=int, default=inference_config.BATCH_SIZE, help="Batch size for inference.")
    parser.add_argument("--num_steps", type=int, default=inference_config.NUM_INFERENCE_STEPS, help="Number of denoising steps.")
    parser.add_argument("--flatten", action=argparse.BooleanOptionalAction, default=inference_config.FLATTEN_OUTPUT, help="Flatten spatial dimensions of hidden states.")
    parser.add_argument("--extract_states", action=argparse.BooleanOptionalAction, default=inference_config.EXTRACT_HIDDEN_STATES, help="Extract hidden states.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=default_config.DEBUG, help="Enable debug logging.")
    
    cli_args = parser.parse_args()
    main(cli_args)