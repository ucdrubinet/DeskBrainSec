#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_wsi_directory> <output_base_directory>"
    exit 1
fi

# Input and output directories
WSI_DIR="$1"
BASE_OUTPUT_DIR="$2"

# Create necessary subdirectories
THUMBNAIL_DIR="${BASE_OUTPUT_DIR}/thumbnails"
MASK_DIR="gauss_otsu"  # Changed to gauss_otsu
TILES_DIR="tiles"
TISSUE_TILES_DIR="${BASE_OUTPUT_DIR}/tissue_tiles"

# Create all required directories
mkdir -p "$THUMBNAIL_DIR" "$MASK_DIR" "$TILES_DIR" "$TISSUE_TILES_DIR"

# Step 1: Generate thumbnails and background masks
echo "Generating thumbnails and background masks..."
python3 bg_seg_script.py "$WSI_DIR" "$THUMBNAIL_DIR" 64 "gauss_otsu" "Background Segmentation"

# Step 2: Run the C tiling program to generate tiles
echo "Generating tiles..."
./tiling_program "$MASK_DIR" "$TILES_DIR"

# Step 3: Use the generated masks to filter tiles
echo "Filtering tiles based on background masks..."
python3 tile_filter.py "$TILES_DIR" "$MASK_DIR" "$TISSUE_TILES_DIR"

echo "Processing complete!"
echo "Tissue tiles have been saved to: $TISSUE_TILES_DIR"
