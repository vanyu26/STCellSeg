"""
Tiling Script - Config-Based
=============================
Loads parameters from config.yaml and runs tiling pipeline.
Currently supports xenium data, could update for other data formats
"""
import os 
import sys
import json
import yaml
import argparse
from pathlib import Path

# Import from same directory
import multichannel_downsampling as mcd
import tiles_io_helper


def run_tiling(config_path: str):
    """Run tiling pipeline from config file."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    input_cfg = config['input']
    image_cfg = config['image']
    tiling_cfg = config['tiling']
    output_cfg = config['output']

    print("=" * 60)
    print("XENIUM TILING PIPELINE")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Mpp: {image_cfg['mpp']}")
    print(f"Tile size: {tiling_cfg['tile_size']}")
    print(f"Overlap: {tiling_cfg['overlap']}")
    print("=" * 60)
    
    # Run tiling
    tiles, reader = mcd.load_and_tile(
        channel_paths=[os.path.join(input_cfg['data_dir'], "morphology_focus", i) for i in input_cfg.get('morphology_images')],
        channel_names=input_cfg.get('channel_names'),
        mpp=image_cfg['mpp'],
        tile_size=tiling_cfg['tile_size'],
        overlap=tiling_cfg['overlap'],
        )
    
    print(f"\nGenerated {len(tiles)} tiles")
    print(f"Image shape at mpp {image_cfg['mpp']}: {reader.image_shape}")
    
    # Save tiles and metadata separately
    out_dir = output_cfg['base_dir']
    
    tiles_io_helper.save_tiles_for_pipeline(
        tiles=tiles,
        tiles_output_dir=os.path.join(out_dir, "tiles"),
        metadata_output_dir=os.path.join(out_dir, "tiles_metadata"),
        prefix="tile"
    )
    
    # Save tiling info for downstream steps
    tiling_info = {
        'image_shape': reader.image_shape,
        'tile_size': tiling_cfg['tile_size'],
        'overlap': tiling_cfg['overlap'],
        'num_tiles': len(tiles),
        'mpp': image_cfg['mpp'],
    }
    
    info_path = os.path.join(out_dir, "tiles_metadata", 'tiling_info.json')
    with open(info_path, 'w') as f:
        json.dump(tiling_info, f, indent=2)
    
    print(f"\nSaved tiling info to {info_path}")
    print("\n" + "=" * 60)
    print("TILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile Xenium images and masks")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    run_tiling(args.config)
