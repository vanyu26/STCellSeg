"""
Transcript Assignment - Config-Based
=====================================
Assigns transcripts to cells using parameters from config.yaml.
"""

import argparse
import yaml
import os
import json
from pathlib import Path
from typing import Tuple
from functools import partial
import gzip

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import coo_matrix, csr_matrix
from scipy.io import mmwrite
from multiprocessing import Pool, cpu_count


class CoordinateTransformer:
    """Transforms transcript coordinates from microns to pixels."""

    def __init__(self, mpp: float, origin: Tuple[float, float] = (0.0, 0.0)):
        self.mpp = mpp
        self.origin = origin

    def to_pixels(self, x_microns: np.ndarray, y_microns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert (x, y) from microns to pixel coordinates."""
        x_origin, y_origin = self.origin
        pixel_x = ((x_microns - x_origin) / self.mpp).round().astype(int)
        pixel_y = ((y_microns - y_origin) / self.mpp).round().astype(int)
        return pixel_x, pixel_y


class ParallelTranscriptLoader:
    """Loads transcript data from Xenium transcripts.zarr."""

    def __init__(self, zarr_path: str):
        self.zarr_path = zarr_path
        self._root = zarr.open(zarr_path, mode="r")
        self.gene_names = self._root.attrs.get("gene_names", [])
    
    def _process_fov_worker(
        self,
        fov_key: str,
        mask: np.ndarray,
        coord_transformer: CoordinateTransformer,
        quality_threshold: float,
        use_valid: bool = True,
        to_df: bool = False
    ):
        """Worker function: process one FOV tile."""
        root = zarr.open(self.zarr_path, mode="r")
        tile = root["grids"]["0"][fov_key]
        
        # Load transcript data
        uid = tile["id"][:, 0]
        location = tile["location"][:]
        gene_id = tile["gene_identity"][:, 0]
        qv = tile["quality_score"][:, 0]
        valid = tile["valid"][:, 0]
        
        if use_valid:
            valid_mask = valid == 1
            if not valid_mask.any():
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32), None
            uid = uid[valid_mask]
            location = location[valid_mask]
            gene_id = gene_id[valid_mask]
        else:
            qv_mask = qv >= quality_threshold
            if not qv_mask.any():
                return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32), None
            uid = uid[qv_mask]
            location = location[qv_mask]
            gene_id = gene_id[qv_mask]
        
        # Convert coordinates
        pixel_x, pixel_y = coord_transformer.to_pixels(
            x_microns=location[:, 0],
            y_microns=location[:, 1]
        )
        
        # Clip to mask bounds
        H, W = mask.shape
        pixel_x = np.clip(pixel_x, 0, W - 1)
        pixel_y = np.clip(pixel_y, 0, H - 1)
        
        # Lookup cell IDs
        cell_ids_assigned = mask[pixel_y, pixel_x]
        
        if to_df:
            df = pd.DataFrame({
                "transcript_id": uid,
                "x_location": location[:, 0],
                "y_location": location[:, 1],
                "z_location": location[:, 2],
                "gene_id": gene_id,
                "cell_id": cell_ids_assigned
            })
        else:
            df = None
        
        # Filter out background
        assigned_mask = cell_ids_assigned > 0
        if not assigned_mask.any():
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.int32), df
        
        cell_ids_final = cell_ids_assigned[assigned_mask]
        gene_ids_final = gene_id[assigned_mask]
        counts = np.ones(len(cell_ids_final), dtype=np.int32)
        
        print(f"  Finished FOV: {fov_key}")
        
        return cell_ids_final, gene_ids_final, counts, df


def aggregate_counts(results, gene_names, to_df):
    """Aggregate results from all FOVs."""
    print("  Aggregating results...")
    
    all_dfs = []
    all_cell_ids = []
    all_gene_ids = []
    all_counts = []
    
    for cell_ids, gene_ids, counts, df in results:
        if to_df and df is not None:
            all_dfs.append(df)
        if len(cell_ids) > 0:
            all_cell_ids.append(cell_ids)
            all_gene_ids.append(gene_ids)
            all_counts.append(counts)
    
    if not all_cell_ids:
        print("  WARNING: No transcripts assigned to any cells")
        return csr_matrix((0, 0)), [], [], None
    
    # Concatenate
    if to_df and all_dfs:
        dfs_concat = pd.concat(all_dfs)
    else:
        dfs_concat = None
    
    cell_ids_concat = np.concatenate(all_cell_ids)
    gene_ids_concat = np.concatenate(all_gene_ids)
    counts_concat = np.concatenate(all_counts)
    
    n_assigned = len(cell_ids_concat)
    print(f"  Total transcripts assigned: {n_assigned}")
    
    # Build count matrix
    print("  Building count matrix...")
    unique_cells = np.sort(np.unique(cell_ids_concat))
    cell_to_col = {c: i for i, c in enumerate(unique_cells)}
    
    col_indices = np.array([cell_to_col[c] for c in cell_ids_concat])
    row_indices = gene_ids_concat
    
    # Coo matrix automatically sum duplicate indices
    count_matrix = coo_matrix(
        (counts_concat, (row_indices, col_indices)),
        shape=(len(gene_names), len(unique_cells)),
    ).tocsr()
    
    print(f"  Matrix shape: {count_matrix.shape[0]} genes × {count_matrix.shape[1]} cells")
    print(f"  Total counts: {count_matrix.sum()}")
    
    return count_matrix, unique_cells, gene_names, dfs_concat


def run_transcript_assignment(config_path: str):
    """Run transcript assignment from config file."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract parameters
    input_cfg = config['input']
    output_cfg = config['output']
    transcript_cfg = config['transcript_assignment']
    resources_cfg = config['resources']['transcript_assignment']
    
    # Load tiling info to get mpp
    with open(os.path.join(output_cfg['base_dir'], 'tiles_metadata', 'tiling_info.json'), 'r') as f:
        tiling_info = json.load(f)
    
    transript_zarr = os.path.join(input_cfg['data_dir'], "transcripts.zarr.zip")
    mpp = tiling_info['mpp']
    
    mask_path = os.path.join(output_cfg['base_dir'], "cell_mask", "stitched_mask.npy")
    transcript_dir = os.path.join(output_cfg['base_dir'], "cell_matrix")
    os.makedirs(transcript_dir, exist_ok=True)
    
    n_workers = resources_cfg.get('threads', cpu_count())
    
    print("=" * 60)
    print("TRANSCRIPT ASSIGNMENT")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Transcripts: {transript_zarr}")
    print(f"Mask: {mask_path}")
    print(f"MPP: {mpp} µm/px")
    print(f"Workers: {n_workers}")
    print(f"Use valid flag: {transcript_cfg['use_valid_flag']}")
    print("=" * 60)
    
    # Load mask
    print("\nLoading mask...")
    mask = np.load(mask_path)
    print(f"Mask shape: {mask.shape}")
    print(f"Number of cells: {len(np.unique(mask)) - 1}")
    
    # Initialize loader
    print("\nInitializing transcript loader...")
    loader = ParallelTranscriptLoader(zarr_path=transript_zarr)
    
    # Get FOV keys
    root = zarr.open(transript_zarr, mode="r")
    level_0 = root["grids"]["0"]
    fov_keys = sorted(level_0.keys())
    print(f"Found {len(fov_keys)} FOV tiles")
    
    # Prepare transformer
    coord_transformer = CoordinateTransformer(
        mpp=mpp,
        origin=tuple(transcript_cfg['origin'])
    )
    
    # Create worker function
    worker_func = partial(
        loader._process_fov_worker,
        mask=mask,
        coord_transformer=coord_transformer,
        quality_threshold=transcript_cfg['quality_threshold'],
        use_valid=transcript_cfg['use_valid_flag'],
        to_df=transcript_cfg['save_transcript_df']
    )
    
    # Process FOVs in parallel
    print(f"\nProcessing FOVs with {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        results = pool.map(worker_func, fov_keys)
    
    # Aggregate results
    counts, cell_ids, gene_names, assigned_df = aggregate_counts(
        results,
        loader.gene_names,
        transcript_cfg['save_transcript_df']
    )
    
    # Save count matrix
    print("\nSaving count matrix...")
    cell_ids_path = os.path.join(transcript_dir, "barcodes.tsv.gz")
    gene_names_path = os.path.join(transcript_dir, "features.tsv.gz")
    matrix_path = os.path.join(transcript_dir, "matrix.mtx.gz")
    
    if transcript_cfg['save_npz']:
        counts_path = os.path.join(transcript_dir, "counts.npz")
        np.savez_compressed(
            counts_path,
            data=counts.data,
            indices=counts.indices,
            indptr=counts.indptr,
            shape=counts.shape,
            cell_ids=cell_ids,
            gene_names=gene_names
        )
    
    with gzip.open(matrix_path, "wb") as f:
        mmwrite(f, counts)
        
    cell_ids_df = pd.DataFrame(cell_ids)
    cell_ids_df.to_csv(cell_ids_path, index = False, sep = "\t", compression = "gzip")
    gene_names_df = pd.DataFrame(gene_names)
    gene_names_df.to_csv(gene_names_path, index = False, sep = "\t", compression = "gzip")
    
    print(f"Matrix saved to: {matrix_path}")
    print(f"Cell ids saved to: {cell_ids_path}")
    print(f"Gene names saved to: {gene_names_path}")
    
    # Save transcript assignments if requested
    if transcript_cfg['save_transcript_df'] and assigned_df is not None:
        print("\nSaving transcript assignments...")
        transcript_path = os.path.join(transcript_dir, "assigned_transcripts.csv.gz")
        assigned_df.to_csv(transcript_path, index=False, compression="gzip")
        print(f"Saved to: {transcript_path}")
    
    print("\n" + "=" * 60)
    print("TRANSCRIPT ASSIGNMENT COMPLETE")
    print("=" * 60)
    print(f"Total cells with transcripts: {len(cell_ids)}")
    print(f"Total genes: {len(gene_names)}")
    print(f"Total UMIs: {counts.sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign transcripts to cells")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    run_transcript_assignment(args.config)
