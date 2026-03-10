"""
Downsampling & Tiling
=====================
Loads separate OME-TIFF files (one per channel) at a specified pyramid level,
downsamples the native-resolution mask to match, then tiles both into
fixed-size patches.

Config keys
-----------
channel_paths   : list of paths to OME-TIFF files, one per channel
channel_names   : optional list of human-readable names (e.g., ["DAPI", "PolyT"])
level           : which pyramid level to load (0 = native, 1 = 2x down, etc.)
mask_path       : path to the Xenium cells.zarr.zip store
mask_type       : "nucleus" (masks/0) or "cell" (masks/1)
tile_size       : spatial size of each tile (pixels)
overlap         : overlap on each side in pixels (0 = no overlap)
pad_mode        : "reflect" | "constant" for edge tiles
pad_value       : fill value when pad_mode="constant"

NOTE: directory loading
    Set any channel_path to a directory to scan for the first OME-TIFF found
    inside it. Useful when Xenium output folders are passed directly.
"""

from __future__ import annotations
import os 
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import tiles_io_helper
import numpy as np
import tifffile
import zarr
from skimage.transform import resize
import argparse


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TilingConfig:
    channel_paths:  List[str] = field(default_factory=list)  # One OME-TIFF per channel
    channel_names:  Optional[List[str]] = None               # e.g., ["DAPI", "PolyT", "CD45"]
    mpp:            float = 1.0          # default to original resolution 
    mask_path:      str   = ""
    mask_type:      str   = "nucleus"    # "nucleus" -> masks/0  |  "cell" -> masks/1
    tile_size:      int   = 512
    overlap:        int   = 0
    pad_mode:       str   = "reflect"    # "reflect" | "constant"
    pad_value:      float = 0.0


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class TileRecord:
    row:        int
    col:        int
    y_start:    int            # top-left in the downsampled image (before padding)
    x_start:    int
    y_end:      int            # bottom-right (exclusive)
    x_end:      int
    pad_bottom: int
    pad_right:  int
    image_tile: np.ndarray     # (C, tile_size, tile_size) float32
    mask_tile:  Optional[np.ndarray]  # (tile_size, tile_size) int32, or None


# ---------------------------------------------------------------------------
# 1.  OME-TIFF reader  (multiscale pyramid)
# ---------------------------------------------------------------------------

class MultiChannelReader:
    """
    Loads a specified pyramid level from separate OME-TIFF files (one per channel)
    and stacks them into a single multi-channel array.
    
    Args
    ----
    channel_paths : List[str]
        Paths to OME-TIFF files, one per channel, in the desired channel order.
        Each file must have the same pyramid structure.
    level : int
        Which pyramid level to load (0 = finest/native, 1 = 2x downsampled, etc.)
    channel_names : Optional[List[str]]
        Human-readable names for each channel (e.g., ["DAPI", "PolyT", "CD45"]).
        If None, defaults to ["ch0", "ch1", ...].

    """

    def __init__(
        self,
        channel_paths: List[str],
        mpp: float,
        channel_names: Optional[List[str]] = None,
    ):
        self.channel_paths = channel_paths
        self.mpp = mpp
        self.channel_names = channel_names or [f"ch{i}" for i in range(len(channel_paths))]
        
        if len(self.channel_names) != len(self.channel_paths):
            raise ValueError(
                f"channel_names length ({len(self.channel_names)}) does not match "
                f"channel_paths length ({len(self.channel_paths)})"
            )
    def _convert_resolution(self):
        mpp = str(self.mpp)
        res = {"1": 0, "0.425": 1, "0.85": 2}
        try: 
            return res[mpp]
        except KeyError:
            print("No pyramid level at selected resolution... run customized downsampling on image level 0")
            # TODO: implement customized downsampling 
            
    def read_ome_level(self):
        # TODO: could add more sanity check and verbose
        level = self._convert_resolution()
        channels = []
        for path in self.channel_paths:
            data = tifffile.imread(path, is_ome = False, level = level)
            channels.append(data.astype(np.float32))
        
        stacked = np.stack(channels, axis=0)  # (C, H, W)
        self.image_shape = channels[0].shape
        return stacked


# ---------------------------------------------------------------------------
# 2.  Instance mask loader + downsampler
# ---------------------------------------------------------------------------

class MaskLoader:
    """
    Loads a native-resolution instance mask from a Xenium cells.zarr.zip store
    and downsamples it via nearest-neighbour to a target spatial shape.

    Xenium cells.zarr.zip layout (from 10x Genomics docs)
    ------------------------------------------------------
    masks/0  — nucleus segmentation mask  (uint32, 2D, shape = morphology image)
    masks/1  — cell segmentation mask     (uint32, 2D, shape = morphology image)

    Non-zero pixel values are instance label IDs; 0 is background.
    For cell masks, label ID = cell index + 1.

    Nearest-neighbour (order=0) is mandatory for instance masks:
    bilinear/bicubic interpolation creates fractional values at cell
    boundaries, producing spurious IDs not present in the original mask.

    Args
    ----
    mask_path   : path to cells.zarr.zip (or unzipped cells.zarr directory)
    mask_type   : "nucleus" -> masks/0  |  "cell" -> masks/1
    """

    MASK_INDEX = {"nucleus": "0", "cell": "1"} # could modify if there are other subcellular level segmentation 

    def __init__(self, mask_path: str, mask_type: str = "nucleus"):
        if mask_type not in self.MASK_INDEX:
            raise ValueError(f"mask_type must be 'nucleus' or 'cell', got '{mask_type}'")
        self.mask_path = mask_path
        self.mask_type = mask_type
        self._raw      = self._load(mask_path, self.MASK_INDEX[mask_type])

    @property
    def native_shape(self) -> Tuple[int, int]:
        return self._raw.shape  # (H, W)

    def get_native(self) -> np.ndarray:
        return self._raw

    def resize_to(self, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Return the mask downsampled to target_shape (H, W) as int32.
        Returns original array unchanged if shapes already match.
        """
        if tuple(self._raw.shape) == tuple(target_shape):
            return self._raw.astype(np.int32)

        if target_shape[0] > self._raw.shape[0] or target_shape[1] > self._raw.shape[1]:
            warnings.warn(
                f"target_shape {target_shape} is larger than native mask shape "
                f"{self._raw.shape} -- this would upsample the mask."
            )

        resized = resize(
            self._raw.astype(np.float32),
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
        return resized.astype(np.int32)

    @staticmethod
    def _load(mask_path: str, mask_index: str) -> np.ndarray:
        """
        Open cells.zarr.zip (or .zarr directory) and read masks/<mask_index>.
        """
        store = zarr.storage.ZipStore(mask_path)
        data = zarr.open(store, mode="r")

        arr = np.squeeze(data['masks'][mask_index])
        if arr.ndim != 2:
            raise ValueError(
                f"Instance mask must be 2-D after squeezing; got shape {arr.shape}"
            )
        return arr.astype(np.int32)



# ---------------------------------------------------------------------------
# 3.  Tiler
# ---------------------------------------------------------------------------

class Tiler:
    """
    Splits a (C, H, W) image and optional (H, W) mask into fixed-size tiles.

    Overlap
    -------
    stride = tile_size - 2 * overlap
    A final anchor tile at (length - tile_size) ensures full spatial coverage
    even when image dimensions are not divisible by stride.

    Edge padding
    ------------
    Image : reflect padding by default (avoids dark borders that confuse models)
    Mask  : always zero-padded (0 = background) — reflecting cell IDs into
            the padded border would introduce false instances at tile edges
    """

    def __init__(self, tile_size, overlap):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - 2 * overlap
        if self.stride <= 0:
            raise ValueError(
                f"overlap ({overlap}) must be < tile_size / 2 "
                f"({tile_size // 2})"
            )

    def tile(
        self,
        image: np.ndarray,
        mask:  Optional[np.ndarray] = None,
    ) -> List[TileRecord]:
        """
        Parameters
        ----------
        image : (C, H, W) float32
        mask  : (H, W) int32, or None

        Returns
        -------
        List of TileRecord — one per grid position.
        """
        C, H, W = image.shape
        T       = self.tile_size

        if mask is not None and mask.shape != (H, W):
            raise ValueError(
                f"Mask shape {mask.shape} does not match image spatial dims ({H}, {W}). "
                "Call MaskLoader.resize_to(image_shape) first."
            )

        records = []
        for row, y in enumerate(self._positions(H, T)):
            for col, x in enumerate(self._positions(W, T)):

                y_end = min(y + T, H)
                x_end = min(x + T, W)

                img_patch  = image[:, y:y_end, x:x_end]
                mask_patch = mask[y:y_end, x:x_end] if mask is not None else None

                pad_b = T - img_patch.shape[1]
                pad_r = T - img_patch.shape[2]

                if pad_b > 0 or pad_r > 0:
                    img_patch = self._pad_image(img_patch, pad_b, pad_r)
                    if mask_patch is not None:
                        mask_patch = self._pad_mask(mask_patch, pad_b, pad_r)

                records.append(TileRecord(
                    row=row, col=col,
                    y_start=y,   x_start=x,
                    y_end=y_end, x_end=x_end,
                    pad_bottom=pad_b, pad_right=pad_r,
                    image_tile=img_patch.astype(np.float32),
                    mask_tile=(mask_patch.astype(np.int32) if mask_patch is not None else None),
                ))

        return records

    def grid_shape(self, H: int, W: int) -> Tuple[int, int]:
        """(n_rows, n_cols) for a given image size."""
        return (len(self._positions(H, self.tile_size)),
                len(self._positions(W, self.tile_size)))


    def _positions(self, length: int, tile_size: int) -> List[int]:
        positions = list(range(0, length - tile_size + 1, self.stride))
        last      = length - tile_size
        if last > 0 and (not positions or positions[-1] < last):
            positions.append(last)
        if not positions:
            positions = [0]
        return positions

    def _pad_image(self, patch: np.ndarray, pad_b: int, pad_r: int, pad_mode="constant", pad_value= 0.0) -> np.ndarray:
        # TODO: only use constant padding, not sure if reflect padding will affect the segmentation result
        if pad_mode == "reflect":
            h, w = patch.shape[1], patch.shape[2]
            mode = "reflect" if (h > 1 and w > 1) else "constant"
        else:
            mode = "constant"
        kwargs = {} if mode == "reflect" else {"constant_values": pad_value}
        return np.pad(patch, ((0, 0), (0, pad_b), (0, pad_r)), mode=mode, **kwargs)

    @staticmethod
    def _pad_mask(patch: np.ndarray, pad_b: int, pad_r: int) -> np.ndarray:
        return np.pad(patch, ((0, pad_b), (0, pad_r)),
                      mode="constant", constant_values=0)


# ---------------------------------------------------------------------------
# 4.  Tiling from reader
# ---------------------------------------------------------------------------

def load_and_tile(channel_paths, channel_names, mpp,  
                  tile_size, overlap, mask_path = None, mask_type = None) -> Tuple[List[TileRecord], MultiChannelReader]:
    """
    Full pipeline: load image → select pyramid level → resize mask → tile.

    Parameters
    ----------
    config : TilingConfig

    Returns
    -------
    records : List[TileRecord]
    """
    reader = MultiChannelReader(
        channel_paths=channel_paths,
        mpp = mpp,
        channel_names=channel_names,
    )

    image = reader.read_ome_level() # (C, H, W)
    img_H, img_W = image.shape[1], image.shape[2] 

    if mask_path is not None:
        mask_loader = MaskLoader(mask_path, mask_type=mask_type)
        mask        = mask_loader.resize_to((img_H, img_W))
        print(
            f"Mask: native {mask_loader.native_shape} → resized {mask.shape}  "
            f"({np.unique(mask).size - 1} cells + background)"
        )
    else:
        mask = None

    tiler   = Tiler(tile_size=tile_size, overlap=overlap)
    records = tiler.tile(image, mask)
    n_rows, n_cols = tiler.grid_shape(img_H, img_W)
    print(
        f"Tiled: {img_H}×{img_W} {len(records)} tiles "
        f"({n_rows} rows × {n_cols} cols)  "
        f"tile_size={tile_size}  overlap={overlap}"
    )

    return records, reader

