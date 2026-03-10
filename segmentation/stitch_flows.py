"""
Flow Stitching - Config-Based
==============================
Stitches Cellpose flow predictions using parameters from config.yaml.
"""

import argparse
import yaml
import os
import json
import pickle
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass

import numpy as np
import torch
from cellpose import dynamics


@dataclass
class TileInfo:
    """Metadata + Cellpose outputs for a single tile."""
    row: int
    col: int
    y_start: int
    x_start: int
    y_end: int
    x_end: int
    pad_bottom: int
    pad_right: int
    flows: np.ndarray        # (2, H, W) float32 – dY, dX
    cellprob: np.ndarray     # (H, W) float32
    mask: np.ndarray


class FlowStitcher:
    """Stitches Cellpose flow fields by using inner (non-overlap) regions."""

    def __init__(
        self,
        image_shape: Tuple[int, int],
        tile_size: int,
        overlap: int,
    ):
        self.H, self.W = image_shape
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - 2 * overlap
        
    def stitch_flows_inner_only(self, tiles):
        """Use only flows from inner (non-overlap) regions of each tile."""
        print(f"Stitching {len(tiles)} Cellpose tiles...")
        flows_stitched = np.zeros((2, self.H, self.W), dtype=np.float32)
        cellprob_stitched = np.zeros((self.H, self.W), dtype=np.float32)
        
        for tile in tiles:
            # Calculate inner region (exclude overlap margins)
            inner_y_start = self.overlap if tile.y_start > 0 else 0
            inner_x_start = self.overlap if tile.x_start > 0 else 0
            inner_y_end = self.tile_size - self.overlap if tile.y_end < self.H else self.tile_size - tile.pad_bottom
            inner_x_end = self.tile_size - self.overlap if tile.x_end < self.W else self.tile_size - tile.pad_right
            
            # Extract inner flows
            inner_flows = tile.flows[:, inner_y_start:inner_y_end, inner_x_start:inner_x_end]
            inner_probs = tile.cellprob[inner_y_start:inner_y_end, inner_x_start:inner_x_end]
            
            # Place in stitched array
            dst_y = tile.y_start + inner_y_start
            dst_x = tile.x_start + inner_x_start
            dst_y_end = dst_y + inner_flows.shape[1]
            dst_x_end = dst_x + inner_flows.shape[2]
            
            flows_stitched[:, dst_y:dst_y_end, dst_x:dst_x_end] = inner_flows
            cellprob_stitched[dst_y:dst_y_end, dst_x:dst_x_end] = inner_probs
        
        self.flows = flows_stitched
        self.cellprob = cellprob_stitched
        
        print(f"  Stitched flows: {self.flows.shape}")
        print(f"  Stitched cellprob: {self.cellprob.shape}")
        print(f"  Flow range: [{self.flows.min():.3f}, {self.flows.max():.3f}]")
        print(f"  Cellprob range: [{self.cellprob.min():.3f}, {self.cellprob.max():.3f}]")

    def flows_to_masks_dynamics(
        self, 
        device = None,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        min_size: int = 15,
    ) -> np.ndarray:
        """Convert stitched flows to instance masks using Cellpose dynamics."""
        print("Running Cellpose dynamics on stitched flows...")
        
        masks = dynamics.compute_masks(
            dP=self.flows,
            cellprob=self.cellprob,
            p=None,
            niter=200,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            device=device,
            min_size=min_size,
        )
        
        n_cells = len(np.unique(masks)) - 1
        print(f"  Found {n_cells} cells")
        
        return masks.astype(np.int32)


def run_stitching(config_path: str):
    """Run flow stitching from config file."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    output_cfg = config['output']
    stitching_cfg = config['stitching']
    resources_cfg = config['resources']['stitching']
    
    # Load tiling info
    metadata_dir = os.path.join(output_cfg['base_dir'], "tiles_metadata")
    with open(os.path.join(metadata_dir, 'tiling_info.json'), 'r') as f:
        tiling_info = json.load(f)
    
    cellpose_predictions_dir = os.path.join(output_cfg['base_dir'], "cellpose_predictions")
    stitched_mask_dir = os.path.join(output_cfg['base_dir'], "cell_mask")
    os.makedirs(stitched_mask_dir, exist_ok=True)
    
    print("=" * 60)
    print("FLOW STITCHING")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Method: {stitching_cfg['method']}")
    print(f"Image shape: {tiling_info['image_shape']}")
    print(f"Tile size: {tiling_info['tile_size']}")
    print(f"Overlap: {tiling_info['overlap']}")
    print("=" * 60)
    
    # Load tiles
    print("\nLoading Cellpose predictions...")
    metadata_files = sorted(Path(metadata_dir).glob("tile_*.json"))
    
    tiles = []
    for metadata_file in metadata_files:
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        tile_name = metadata_file.stem
        
        # Load flows
        flow_path = os.path.join(cellpose_predictions_dir, f"{tile_name}_flows.pkl")
        with open(flow_path, "rb") as file:
            flows_data = pickle.load(file)
        
        tiles.append(TileInfo(
            row=metadata["row"],
            col=metadata["col"],
            y_start=metadata["y_start"],
            x_start=metadata["x_start"],
            y_end=metadata["y_end"],
            x_end=metadata["x_end"],
            pad_bottom=metadata["pad_bottom"],
            pad_right=metadata["pad_right"],
            flows=flows_data[1],      # Gradient field (2, H, W)
            cellprob=flows_data[2],   # Cell probability (H, W)
            mask=None
        ))
    
    print(f"Loaded {len(tiles)} tiles")
    
    # Stitch flows
    stitcher = FlowStitcher(
        image_shape=tuple(tiling_info['image_shape']),
        tile_size=tiling_info['tile_size'],
        overlap=tiling_info['overlap'],
    )
    
    stitcher.stitch_flows_inner_only(tiles=tiles)
    
    # Reconstruct masks
    if stitching_cfg['use_cellpose_dynamics']:
        device = torch.device("cuda" if resources_cfg['gpu'] else "cpu")
        
        mask = stitcher.flows_to_masks_dynamics(
            cellprob_threshold=stitching_cfg['flow']['cellprob_threshold'],
            flow_threshold=stitching_cfg['flow']['flow_threshold'],
            min_size=stitching_cfg['flow']['min_size'],
            device=device
        )
    else:
        raise NotImplementedError("Only cellpose_dynamics method supported")
    
    # Save stitched mask
    output_path = os.path.join(stitched_mask_dir, "stitched_mask.npy")
    np.save(output_path, mask)
    
    print(f"\nSaved stitched mask to: {output_path}")
    print(f"Final mask shape: {mask.shape}")
    print(f"Total cells: {len(np.unique(mask)) - 1}")
    
    print("\n" + "=" * 60)
    print("STITCHING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch Cellpose flow predictions")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    run_stitching(args.config)
