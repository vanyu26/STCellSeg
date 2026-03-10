"""
Tile I/O Helpers - Modified for Pipeline
==========================================
Saves tiles and metadata separately to avoid reloading images in downstream steps.

Key changes:
- save_tiles_for_pipeline(): Saves images and metadata to separate directories
- Metadata saved as JSON for easy loading without image data
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


def save_tiles_for_pipeline(
    tiles: List,  # List of TileRecord
    tiles_output_dir: Union[str, Path],
    metadata_output_dir: Union[str, Path],
    prefix: str = "tile",
) -> None:
    """
    Save tiles for pipeline processing.
    
    Saves:
        - Image tiles: {tiles_output_dir}/{prefix}_{row:04d}_{col:04d}.npy
        - Metadata: {metadata_output_dir}/{prefix}_{row:04d}_{col:04d}.json
    
    This allows downstream steps to load only metadata without loading images.
    
    Args:
        tiles: List of TileRecord from tiling pipeline
        tiles_output_dir: Directory for image tiles (.npy)
        metadata_output_dir: Directory for metadata (.json)
        prefix: Filename prefix
    """
    tiles_output_dir = Path(tiles_output_dir)
    metadata_output_dir = Path(metadata_output_dir)
    
    tiles_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_output_dir.mkdir(parents=True, exist_ok=True)
    
    for tile in tiles:
        # Generate filename
        filename_base = f"{prefix}_{tile.row:04d}_{tile.col:04d}"
        
        # Save image tile
        tile_filepath = tiles_output_dir / f"{filename_base}.npy"
        np.save(tile_filepath, tile.image_tile)
        
        # Save metadata separately
        metadata = {
            "row": tile.row,
            "col": tile.col,
            "y_start": tile.y_start,
            "x_start": tile.x_start,
            "y_end": tile.y_end,
            "x_end": tile.x_end,
            "pad_bottom": tile.pad_bottom,
            "pad_right": tile.pad_right,
        }
        
        metadata_filepath = metadata_output_dir / f"{filename_base}.json"
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(tiles)} tiles to {tiles_output_dir}/")
    print(f"Saved {len(tiles)} metadata files to {metadata_output_dir}/")


def load_tile_metadata(metadata_dir: Union[str, Path], pattern: str = "tile_*.json") -> List[dict]:
    """
    Load tile metadata without loading images.
    
    Args:
        metadata_dir: Directory containing metadata JSON files
        pattern: Glob pattern for metadata files
    
    Returns:
        List of metadata dicts
    """
    metadata_dir = Path(metadata_dir)
    files = sorted(metadata_dir.glob(pattern))
    
    metadata_list = []
    for f in files:
        with open(f, 'r') as fp:
            metadata = json.load(fp)
            metadata['filename'] = f.stem  # Add filename for reference
            metadata_list.append(metadata)
    
    return metadata_list


def load_tiles_with_metadata(
    tiles_dir: Union[str, Path],
    metadata_dir: Union[str, Path],
    pattern: str = "tile_*.npy"
) -> List[dict]:
    """
    Load tiles and their metadata.
    
    Args:
        tiles_dir: Directory containing tile .npy files
        metadata_dir: Directory containing metadata .json files
        pattern: Glob pattern for tile files
    
    Returns:
        List of dicts with 'image' and 'metadata' keys
    """
    tiles_dir = Path(tiles_dir)
    metadata_dir = Path(metadata_dir)
    
    tile_files = sorted(tiles_dir.glob(pattern))
    
    result = []
    for tile_file in tile_files:
        # Load image
        image = np.load(tile_file)
        
        # Load corresponding metadata
        metadata_file = metadata_dir / (tile_file.stem + '.json')
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        result.append({
            'image': image,
            'metadata': metadata,
            'filepath': tile_file,
        })
    
    return result


# Backward compatibility - keep original NPZ functions for reference
def save_tiles_npz(
    tiles: List,
    output_dir: Union[str, Path],
    prefix: str = "tile",
) -> None:
    """Original NPZ format - kept for compatibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for tile in tiles:
        filename = f"{prefix}_{tile.row:04d}_{tile.col:04d}.npz"
        filepath = output_dir / filename
        
        metadata = {
            "row": tile.row,
            "col": tile.col,
            "y_start": tile.y_start,
            "x_start": tile.x_start,
            "y_end": tile.y_end,
            "x_end": tile.x_end,
            "pad_bottom": tile.pad_bottom,
            "pad_right": tile.pad_right,
        }
        
        np.savez_compressed(
            filepath,
            image=tile.image_tile,
            mask=tile.mask_tile if tile.mask_tile is not None else np.array([]),
            metadata=json.dumps(metadata),
        )
    
    print(f"Saved {len(tiles)} tiles to {output_dir}/ as .npz")


def load_tiles_npz(tile_dir: Union[str, Path], pattern: str = "tile_*.npz") -> List[dict]:
    """Original NPZ loader - kept for compatibility."""
    tile_dir = Path(tile_dir)
    files = sorted(tile_dir.glob(pattern))
    
    tiles = []
    for f in files:
        data = np.load(f)
        metadata = json.loads(str(data['metadata']))
        tiles.append({
            'image': data['image'],
            'mask': data['mask'] if data['mask'].size > 0 else None,
            'metadata': metadata,
            'filepath': f,
        })
    
    return tiles


if __name__ == "__main__":
    print("Tile I/O helpers loaded successfully")
    print("Main functions:")
    print("  - save_tiles_for_pipeline(): Save tiles + metadata separately")
    print("  - load_tile_metadata(): Load only metadata (fast)")
    print("  - load_tiles_with_metadata(): Load tiles + metadata")
