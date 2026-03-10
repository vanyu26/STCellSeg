# Xenium Pipeline - Visual Overview

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA                               │
├─────────────────────────────────────────────────────────────┤
│ • morphology_focus_0000.ome.tif (DAPI)                      │
│ • morphology_focus_0001.ome.tif (PolyT)                     │
│ • cells.zarr.zip (optional initial mask)                    │
│ • transcripts.zarr.zip                                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: TILING (preprocessing env)                         │
├─────────────────────────────────────────────────────────────┤
│ Script: preprocessing/run_tiling.py                         │
│ Time:   5-10 min                                            │
│ CPU:    4 threads                                           │
│ Memory: 16 GB                                               │
├─────────────────────────────────────────────────────────────┤
│ • Loads OME-TIFF at pyramid level 1 (0.85 µm/px)            │
│ • Tiles into 1024×1024 patches with 64px overlap            │
│ • Saves tiles + metadata separately                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: ~100-200 tiles                                      │
│ • output/tiles/tile_RRRR_CCCC.npy        (images)           │
│ • output/tiles_metadata/tile_RRRR_CCCC.json (metadata)      │
│ • output/tiles_metadata/tiling_info.json (global params)   │
└──────────────────────────────────────────────────────────── ─┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: CELLPOSE INFERENCE (cellpose env)                 │
├─────────────────────────────────────────────────────────────┤
│ Script: segmentation/run_inference.py                       │
│ Time:   5-10 min                                            │
│ GPU:    1 × NVIDIA (12 GB VRAM)                             │
│ Memory: 12 GB                                               │
├─────────────────────────────────────────────────────────────┤
│ • Runs Cellpose4 SAM model on each tile                     │
│ • Saves masks + flows per tile                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: Cellpose predictions per tile                       │
│ • output/cellpose_predictions/tile_RRRR_CCCC_mask.npy       │
│ • output/cellpose_predictions/tile_RRRR_CCCC_flows.pkl      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: FLOW STITCHING (cellpose env)                      │
├─────────────────────────────────────────────────────────────┤
│ Script: segmentation/stitch_flows.py                        │
│ Time:   ~10 min (GPU) ~ 20-30 min (CPU)                     │
│ GPU:    1 × NVIDIA (12 GB VRAM)                             │
│ Memory: 12 GB                                               │
├─────────────────────────────────────────────────────────────┤
│ • Loads metadata from tiles_metadata/                       │
│ • Loads flows from cellpose_predictions/                    │
│ • Stitches using inner regions (no overlap)                 │
│ • Runs Cellpose dynamics on stitched flows                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: Final segmentation                                  │
│ • output/stitched_masks/stitched_mask.npy                   │
│   Shape: (Height, Width) int32                              │
│   Cells: ~50,000-100,000 unique IDs                         │
└─────────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│ STEP 4A: MORPHOLOGY  │   │ STEP 4B: TRANSCRIPT ASSIGN   │
│ (scikitimage env)    │   │ (processing env)             │
├──────────────────────┤   ├──────────────────────────────┤
│ cell_morphology.py   │   │ transcript_assignment.py     │
│ Time: 5-10 min       │   │ Time: 10-20 min              │
│ CPU: 32 threads      │   │ CPU: 32 threads              │
│ Memory: 32 GB        │   │ Memory: 64 GB                │
├──────────────────────┤   ├──────────────────────────────┤
│ • Loads mask         │   │ • Loads mask                 │
│ • regionprops()      │   │ • Loads transcripts.zarr     │
│ • Parallel by cell   │   │ • Parallel by FOV            │
│ • Converts to µm     │   │ • Builds sparse matrix       │
└──────────────────────┘   └──────────────────────────────┘
            │                           │
            ▼                           ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│ OUTPUT: Morphology   │   │ OUTPUT: Gene Expression      │
│ cell_morphology.     │   │ counts                       │
│   csv.gz             │   │ • sparse matrix (CSR)        │
│ • area_um2           │   │ • cell_ids                   │
│ • perimeter_um       │   │ • gene_names                 │
│ • centroid_x/y_um    │   │                              │
│ • elongation         │   │ assigned_transcripts.csv.gz  │
│ • circularity        │   │ (optional, large)            │
│ • solidity           │   │                              │
└──────────────────────┘   └──────────────────────────────┘
```

## Conda Environments

```
┌───────────────────────┬───────────────────────────────────┐
│ Environment           │ Key Dependencies                  │
├───────────────────────┼───────────────────────────────────┤
│ processing            │ • tifffile                        │
│                       │ • zarr                            │
│                       │ • scikit-image                    │
│                       │ • numpy, scipy                    │
│                       │ • pyyaml                          │
├───────────────────────┼───────────────────────────────────┤
│ cellpose              │ • cellpose ≥3.0                   │
│ (GPU-enabled)         │ • pytorch + pytorch-cuda          │
│                       │ • numpy, scipy                    │
│                       │ • pyyaml                          │
├───────────────────────┼───────────────────────────────────┤
│ scikit-image          │ • pandas                          │
│                       │ • scipy (sparse matrices)         │
│                       │ • scikit-image (regionprops)      │
│                       │ • zarr (transcript loading)       │
│                       │ • numpy, pyyaml                   │
└───────────────────────┴───────────────────────────────────┘
```

## File Organization

```
xenium_pipeline/
│
├── 📋 Workflow Definition
│   └── Snakefile                       # Orchestrates entire pipeline
│
├── ⚙️ Configuration
│   └── config/
│       └── config.yaml                 # ** EDIT THIS FILE **
│
├── 🐍 Conda Environments
│   └── envs/
│       ├── preprocessing.yaml
│       ├── cellpose.yaml
│       └── analysis.yaml
│
├── 📁 Module 1: Preprocessing
│   └── preprocessing/
│       ├── multichannel_downsampling.py   # Core tiling logic
│       ├── tiles_io_helper.py             # Modified: separate metadata
│       └── run_tiling.py                  # New: config wrapper
│
├── 📁 Module 2: Segmentation
│   └── segmentation/
│       ├── run_inference.py               # Modified: config-based
│       └── stitch_flows.py                # Modified: config-based
│
├── 📁 Module 3: tomatrix
│   └── tomatirx/
│       ├── cell_morphology.py             # Modified: config-based
│       └── transcript_assignment.py       # Modified: config-based
│
└── 📖 Documentation
    ├── README.md                       # Full documentation
    ├── QUICKSTART.md                   # Quick start guide
    └── SUMMARY.md                      # This summary
```

## Execution Flow

```
Terminal Command:
$ snakemake --use-conda --configfile config/config.yaml --cores 32

    │
    ├─→ Checks config/config.yaml
    │
    ├─→ Creates conda environments (first time only)
    │   ├── preprocessing
    │   ├── cellpose
    │   └── analysis
    │
    ├─→ Determines what needs to run
    │   └── Checks input files vs output files
    │
    ├─→ Runs rules in order:
    │   │
    │   ├── 1. tiling
    │   │   └── conda activate preprocessing
    │   │   └── python preprocessing/run_tiling.py --config config.yaml
    │   │   └── Saves to output/tiles/ + output/tiles_metadata/
    │   │
    │   ├── 2. cellpose_inference
    │   │   └── conda activate cellpose
    │   │   └── python segmentation/run_inference.py --config config.yaml
    │   │   └── Saves to output/cellpose_predictions/
    │   │
    │   ├── 3. stitch_flows
    │   │   └── conda activate cellpose
    │   │   └── python segmentation/stitch_flows.py --config config.yaml
    │   │   └── Saves to output/stitched_masks/
    │   │
    │   ├── 4a. cell_morphology 
    │   │   └── conda activate analysis      
    │   │   └── python analysis/cell_morphology.py 
    │   │   └── Saves to output/morphology/         
    │   │                                           
    │   └── 4b. transcript_assignment 
    │       └── conda activate analysis
    │       └── python analysis/transcript_assignment.py
    │       └── Saves to output/transcript_assignment/
    │
    └─→ Logs everything to logs/
        ├── tiling.log
        ├── cellpose_inference.log
        ├── stitch_flows.log
        ├── cell_morphology.log
        └── transcript_assignment.log
```


