from pathlib import Path
import os

configfile: "config/config.yaml"

OUTDIR = Path(config['output']['base_dir'])
PROCESSING_ENV = "A:/conda/env/processing" # change to envs/zarr.yaml
CELLPOSE_ENV = "A:/conda/env/cellpose4" # change to envs/cellpose.yaml
SCIKIT_ENV = "A:/conda/env/scikitimage" # change to envs/scikitimage.yaml

LOG_DIR = OUTDIR / Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ----------------------------
# Rule all
# ----------------------------
rule all:
    input:
        tiles = OUTDIR / "tiles",
        tiles_metadata = OUTDIR / "tiles_metadata",
        cellpose_predictions = OUTDIR / "cellpose_predictions",
        cell_mask = OUTDIR / "cell_mask",
        morphology = OUTDIR / "cell_matrix" / "cell_metadata.csv.gz",
        cell_matrix = OUTDIR / "cell_matrix" / "matrix.mtx.gz",
        cell_id = OUTDIR / "cell_matrix" / "barcodes.tsv.gz",
        gene_id = OUTDIR / "cell_matrix" / "features.tsv.gz",

# ----------------------------
# Tiling
# ----------------------------
rule tiling:
    output: 
        tiles = directory(OUTDIR / "tiles"),
        tiles_metadata = directory(OUTDIR / "tiles_metadata")
    params:
        config_file = "config/config.yaml"
    conda:
        PROCESSING_ENV
    threads:
        config["resources"]["preprocessing"]["threads"]
    log:
        LOG_DIR / "tiling.log"
    shell:
        """
        python preprocessing/run_tiling.py --config {params.config_file} > {log} 2>&1
        """

# ----------------------------
# Cellpose inference
# ----------------------------
rule cellpose_inference:
    input: 
        tiles = rules.tiling.output.tiles,
        tiles_metadata = rules.tiling.output.tiles_metadata
    output:
        cellpose_predictions = directory(OUTDIR / "cellpose_predictions")
    params:
        config_file = "config/config.yaml"
    conda:
        CELLPOSE_ENV
    threads:
        config["resources"]["cellpose_inference"]["threads"]
    resources:
        gpu = 1 if config["resources"]["cellpose_inference"]["gpu"] else 0
    log:
        LOG_DIR / "cellpose_inference.log"
    shell:
        """
        python segmentation/run_inference.py --config {params.config_file} > {log} 2>&1
        """

# ----------------------------
# Stitch flows
# ----------------------------
rule stitch_flows:
    input:
        cellpose_predictions = rules.cellpose_inference.output.cellpose_predictions
    output:
        cell_mask = directory(OUTDIR / "cell_mask")
    params:
        config_file = "config/config.yaml"
    conda:
        CELLPOSE_ENV
    threads:
        config["resources"]["stitching"]["threads"]
    resources:
        gpu = 1 if config["resources"]["stitching"]["gpu"] else 0
    log:
        LOG_DIR / "stitch_flows.log"
    shell:
        """
        python segmentation/stitch_flows.py --config {params.config_file} > {log} 2>&1
        """

# ----------------------------
# Cell morphology
# ----------------------------
rule cell_morphology:
    input:
        cell_mask = rules.stitch_flows.output.cell_mask
    output:
        OUTDIR / "cell_matrix" / "cell_metadata.csv.gz"
    params:
        config_file = "config/config.yaml"
    conda:
        SCIKIT_ENV
    threads:
        config["resources"]["morphology"]["threads"]
    log:
        LOG_DIR / "cell_morphology.log"
    shell:
        """
        python tomatrix/cell_morphology.py --config {params.config_file} > {log} 2>&1
        """

# ----------------------------
# Transcript assignment
# ----------------------------
rule transcript_assignment:
    input:
        cell_mask = rules.stitch_flows.output.cell_mask
    output:
        OUTDIR / "cell_matrix" / "matrix.mtx.gz",
        OUTDIR / "cell_matrix" / "barcodes.tsv.gz",
        OUTDIR / "cell_matrix" / "features.tsv.gz",
    params:
        config_file = "config/config.yaml"
    conda:
        PROCESSING_ENV
    threads:
        config["resources"]["transcript_assignment"]["threads"]
    log:
        LOG_DIR / "transcript_assignment.log"
    shell:
        """
        python tomatrix/transcript_assignment.py --config {params.config_file} > {log} 2>&1
        """