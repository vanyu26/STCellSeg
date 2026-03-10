# Xenium Segmentation Pipeline

Complete Snakemake pipeline for Xenium spatial transcriptomics data processing, including segmentation, morphology analysis, and transcript assignment.

## Directory Structure

```
pipeline/
├── Snakefile                    # Main workflow definition
├── config/
│   └── config.yaml              # Central configuration file
├── envs/
│   ├── preprocessing.yaml       # Conda env for tiling
│   ├── cellpose.yaml            # Conda env for segmentation
│   └── analysis.yaml            # Conda env for analysis
├── preprocessing/
│   ├── multichannel_downsampling.py
│   ├── tiles_io_helper.py
│   └── run_tiling.py
├── segmentation/
│   ├── run_inference.py
│   └── stitch_flows.py
└── tomatrix/
    ├── cell_morphology.py
    └── transcript_assignment.py
```

## Installation

### 1. Install Snakemake

```bash
conda create -n snakemake -c conda-forge -c bioconda snakemake
conda activate snakemake
```

### 2. Create Conda Environments

The pipeline will automatically create the required environments, but you can pre-build them:

```bash
cd pipeline

# Preprocessing environment
conda env create -f envs/zarr.yaml

# Cellpose environment (with GPU support)
conda env create -f envs/cellpose.yaml

# Analysis environment
conda env create -f envs/scikitimage.yaml
```

## Configuration

Edit `config/config.yaml` to specify:

### Input Data

```yaml
input:
  morphology_images:
    - "path/to/morphology_focus_0000.ome.tif"  # DAPI channel
    - "path/to/morphology_focus_0001.ome.tif"  # PolyT channel
  channel_names:
    - "DAPI"
    - "PolyT"
  mask_path: "path/to/cells.zarr.zip"  # Optional initial mask
  transcripts_zarr: "path/to/transcripts.zarr.zip"
```

### Output Directories

```yaml
output:
  base_dir: "output"
  tiles_dir: "output/tiles"
  # ... (other directories created automatically)
```

### Processing Parameters

```yaml
image:
  pyramid_level: 1  # 0=native, 1=2x down, 2=4x down

tiling:
  tile_size: 1024
  overlap: 64

cellpose:
  model_type: "cyto3"
  diameter: 30
  use_gpu: true
```

## Usage

### Run Full Pipeline

```bash
cd pipeline
snakemake --use-conda --configfile config/config.yaml --cores 32
```

## Pipeline Stages

### 1. Tiling (Preprocessing)

**Environment:** `processing`  
**Script:** `preprocessing/run_tiling.py`

- Loads multi-channel OME-TIFF at specified pyramid level
- Downsamples mask to match
- Tiles into overlapping patches
- Saves tiles and metadata separately

**Outputs:**
- `output/tiles/tile_RRRR_CCCC.npy` — Image tiles
- `output/tiles_metadata/tile_RRRR_CCCC.json` — Metadata
- `output/tiles_metadata/tiling_info.json` — Global tiling parameters

### 2. Cellpose Inference (Segmentation)

**Environment:** `cellpose` (GPU-enabled)  
**Script:** `segmentation/run_inference.py`

- Runs Cellpose on each tile independently
- Saves masks and flows for stitching

**Outputs:**
- `output/cellpose_predictions/tile_RRRR_CCCC_mask.npy`
- `output/cellpose_predictions/tile_RRRR_CCCC_flows.pkl`

### 3. Flow Stitching (Segmentation)

**Environment:** `cellpose` (GPU-enabled)  
**Script:** `segmentation/stitch_flows.py`

- Loads flow predictions from all tiles
- Stitches using inner regions (avoids overlap artifacts)
- Reconstructs final mask using Cellpose dynamics

**Outputs:**
- `output/stitched_masks/stitched_mask.npy`

### 4. Cell Morphology (Analysis)

**Environment:** `scikitimage`  
**Script:** `tomatrix/cell_morphology.py`

- Computes morphology metrics for each cell
- Converts pixel measurements to microns
- Parallel processing across CPU cores

**Metrics:**
- Area, perimeter, centroid (μm, μm²)
- Elongation, eccentricity, circularity
- Solidity, convexity

**Outputs:**
- `output/morphology/cell_morphology.csv.gz`

### 5. Transcript Assignment (Analysis)

**Environment:** `processing`  
**Script:** `tomatrix/transcript_assignment.py`

- Assigns transcripts to cells based on spatial coordinates
- Builds sparse cell × gene count matrix
- Parallel processing by FOV tile

**Outputs:**
- `output/transcript_assignment/matrix.mtx.gz` — Sparse count matrix
- `output/transcript_assignment/barcodes.tsv.gz` — Cell ids
- `output/transcript_assignment/features.tsv.gz` — Gene ids
- `output/transcript_assignment/assigned_transcripts.csv.gz` — Full assignments (optional)

## Resource Configuration

Adjust in `config.yaml`:

```yaml
resources:
  preprocessing:
    threads: 4
  
  cellpose_inference:
    threads: 1
    gpu: true
    gpu_id: 0
  
  stitching:
    threads: 1
    gpu: true
  
  morphology:
    threads: 32  # Use all cores for parallel processing
  
  transcript_assignment:
    threads: 32
```

## GPU Configuration

For Cellpose GPU support:

1. Ensure CUDA is installed
2. Update `envs/cellpose.yaml` with your CUDA version:
   ```yaml
   - pytorch::pytorch-cuda=11.8  # Change to match your CUDA
   ```

## Troubleshooting

### Out of Memory (GPU)

Reduce tile size in config:
```yaml
tiling:
  tile_size: 512  # Instead of 1024
```

### Out of Memory (CPU)

Reduce worker count:
```yaml
resources:
  morphology:
    threads: 16  # Instead of 32
```

## Outputs Summary

After successful run:

```
output/
├── tiles/                     # Image tiles (.npy)
├── tiles_metadata/            # Tile metadata (.json)
├── cellpose_predictions/      # Per-tile masks and flows
├── stitched_masks/            
│   └── stitched_mask.npy      # Final segmentation
├── morphology/
│   └── cell_morphology.csv.gz  # Morphology statistics
└── cell_matrix/
    ├── matrix.mtx.gz             #  Gene x cell matrix
    ├── barcodes.tsv.gz             #  Cell ids
    ├── features.tsv.gz             #  Gene ids
    └── assigned_transcripts.csv.gz  # Full assignments (if enabled)
```

