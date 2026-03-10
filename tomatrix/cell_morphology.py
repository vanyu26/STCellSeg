"""
Cell Morphology Analysis - Config-Based
========================================
Computes morphology metrics from segmentation mask using config.yaml.
"""

import argparse
import yaml
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import regionprops, perimeter_crofton  # need to use scikit >= 0.25.0
from skimage.morphology import convex_hull_image 
from multiprocessing import Pool, cpu_count, current_process


class CellMorphology2D:
    def __init__(self, region): 
        self.region = region
        
    def get_morphology_stats(self):
        area = self.region.area
        perimeter = perimeter_crofton(self.region.image)

        y_centroid, x_centroid = self.region.centroid

        major = self.region.major_axis_length
        minor = self.region.minor_axis_length

        elongation = major / minor if minor > 0 else np.nan

        circularity = (
            4 * np.pi * area / (perimeter ** 2)
            if perimeter > 0 else np.nan
        )

        convex_img = convex_hull_image(self.region.image)
        convex_perim = perimeter_crofton(convex_img)

        convexity = (
            convex_perim / perimeter
            if perimeter > 0 else np.nan
        )

        return {
            "label": self.region.label,
            "area_px": area,
            "perimeter_px": perimeter,
            "x_centroid_px": x_centroid,
            "y_centroid_px": y_centroid,
            "major_axis_length_px": major,
            "minor_axis_length_px": minor,
            "elongation": elongation,
            "eccentricity": self.region.eccentricity,
            "solidity": self.region.solidity,
            "convexity": convexity,
            "circularity": circularity,
        }


def _process_batch(regions):
    """Worker function for parallel processing."""

    proc = current_process()
    print(f"[Process {proc.name}] Processing {len(regions)} cells...")
    
    results = []
    for region in regions:
        cell = CellMorphology2D(region)
        results.append(cell.get_morphology_stats())
    
    print(f"[Process {proc.name}] Done.")
    return results


def _chunk_list(lst, n_chunks):
    """Split list into n_chunks."""
    chunk_size = int(np.ceil(len(lst) / n_chunks))
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def run_morphology(config_path: str):
    """Run morphology analysis from config file."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    output_cfg = config['output']
    image_cfg = config['image']
    resources_cfg = config['resources']['morphology']
    
    # Load tiling info to get mpp
    
    mpp = image_cfg['mpp']
    
    mask_path = os.path.join(output_cfg['base_dir'], "cell_mask", "stitched_mask.npy")
    morphology_dir = os.path.join(output_cfg['base_dir'], "cell_matrix")
    os.makedirs(morphology_dir, exist_ok=True)

    n_workers = resources_cfg.get('threads', cpu_count())
    
    print("=" * 60)
    print("CELL MORPHOLOGY ANALYSIS")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Mask: {mask_path}")
    print(f"MPP: {mpp} µm/px")
    print(f"Workers: {n_workers}")
    print("=" * 60)
    
    # Load mask
    print("\nLoading mask...")
    label_mask = np.load(mask_path)
    print(f"Mask shape: {label_mask.shape}")
    print(f"Number of cells: {len(np.unique(label_mask)) - 1}")
    
    # Extract regions
    print("\nExtracting cell regions...")
    regions = list(regionprops(label_mask))
    print(f"Regions extracted: {len(regions)}")
    
    # Process in parallel
    print(f"\nProcessing with {n_workers} workers...")
    batches = list(_chunk_list(regions, n_workers))
    
    with Pool(n_workers) as pool:
        results = pool.map(_process_batch, batches)
    
    # Flatten results
    results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(results)
    
    # Convert to microns
    print("\nConverting to microns...")
    df["area_um2"] = df["area_px"] * (mpp ** 2)
    df["perimeter_um"] = df["perimeter_px"] * mpp
    df["major_axis_length_um"] = df["major_axis_length_px"] * mpp
    df["minor_axis_length_um"] = df["minor_axis_length_px"] * mpp
    df["x_centroid_um"] = df["x_centroid_px"] * mpp
    df["y_centroid_um"] = df["y_centroid_px"] * mpp
    
    # Save results
    output_path = os.path.join(morphology_dir, "cell_metadata.csv.gz")
    df.to_csv(output_path, compression='gzip', index=False)
    
    print(f"\nSaved morphology to: {output_path}")
    print(f"Total cells: {len(df)}")
    
    # Summary statistics
    #print("\nSummary statistics:")
    #print(df[['area_um2', 'perimeter_um', 'elongation', 'circularity', 'solidity']].describe())
    
    print("\n" + "=" * 60)
    print("MORPHOLOGY ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cell morphology statistics")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    run_morphology(args.config)
