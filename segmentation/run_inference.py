"""
Cellpose Inference - Config-Based
==================================
Runs Cellpose on tiled images using parameters from config.yaml.
"""

import argparse
import yaml
import os
import pickle as pkl
from pathlib import Path

import numpy as np
from cellpose import models


def run_inference(config_path: str):
    """Run Cellpose inference from config file."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    output_cfg = config['output']
    input_cfg = config['input']
    cellpose_cfg = config['cellpose']
    resources_cfg = config['resources']['cellpose_inference']
    
    tiles_dir = os.path.join(output_cfg['base_dir'], "tiles")
    cellpose_predictions_dir = os.path.join(output_cfg['base_dir'], "cellpose_predictions")
    os.makedirs(cellpose_predictions_dir)
    
    print("=" * 60)
    print("CELLPOSE INFERENCE")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Use GPU: {cellpose_cfg['use_gpu']}")
    print(f"Tiles directory: {tiles_dir}")
    print(f"Output directory: {cellpose_predictions_dir}")
    print("=" * 60)
    
    # Initialize model
    print("\nLoading Cellpose model...")
    model = models.CellposeModel(gpu=cellpose_cfg['use_gpu'])
    
    # Get all tile files
    tile_files = sorted(Path(tiles_dir).glob("tile_*.npy"))
    print(f"Found {len(tile_files)} tiles to process")
    
    # Process each tile
    for i, tile_file in enumerate(tile_files, 1):
        tile_name = tile_file.stem
        
        print(f"\n[{i}/{len(tile_files)}] Processing {tile_name}...")
        
        # Load tile
        tile = np.load(tile_file)
        
        # Run Cellpose
        masks, flows, styles = model.eval(
            tile, # cellpose4 is invariant of channel order, could just predict
        )
        
        # Save predictions
        mask_filename = os.path.join(cellpose_predictions_dir, f"{tile_name}_mask.npy")
        flows_filename = os.path.join(cellpose_predictions_dir, f"{tile_name}_flows.pkl")
        
        np.save(mask_filename, masks)
        with open(flows_filename, "wb") as file:
            pkl.dump(flows, file)
        
        print(f"  Saved mask: {mask_filename}")
        print(f"  Saved flows: {flows_filename}")
        print(f"  Found {len(np.unique(masks)) - 1} cells")
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Processed {len(tile_files)} tiles")
    print(f"Results saved to: {cellpose_predictions_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cellpose inference on tiles")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    run_inference(args.config)
