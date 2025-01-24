#!/bin/bash

# Check if correct number of arguments provided
if [[ $# < 2 || $# > 3 ]]; then
    echo "Usage: $0 <input_wsi_directory> <output_base_directory> [--debug]"
    exit 1
fi

# handle optional --debug argument when testing inference
if [[ $# == 3 ]]; then
    case "$3" in
        --debug) DEBUG=true;;
        *) echo "Usage: $0 <arg1> <arg2> [--debug]"
        exit 2;;
    esac
fi

# Input and output directories
WSI_DIR="$1"
BASE_OUTPUT_DIR="$2"

# Create necessary subdirectories
THUMBNAIL_DIR="${BASE_OUTPUT_DIR}/thumbnails"
MASK_DIR="gauss_otsu"  # Changed to gauss_otsu
TILES_DIR="tiles2"
TISSUE_TILES_DIR="${BASE_OUTPUT_DIR}/tissue_tiles"
INFERENCE_OUTPUT_DIR="${BASE_OUTPUT_DIR}/inference_output"
POSTPROCESS_DIR="${BASE_OUTPOUT_DIR}/postprocess_output"
MODEL_WEIGHTS="checkpoints/PlaqueTissueClassifier_Epoch_0.pth"

# Create all required directories
mkdir -p "$THUMBNAIL_DIR" "$MASK_DIR" "$TILES_DIR" "$TISSUE_TILES_DIR"

# Step 1: Generate thumbnails and background masks
echo "Generating thumbnails and background masks..."
python bg_seg_script.py "$WSI_DIR" "$THUMBNAIL_DIR" 64 "gauss_otsu" "Background Segmentation"

# Step 2: Run the C tiling program to generate tiles
echo "Generating tiles..."
make tiling
./tiling_program "$WSI_DIR" "$TILES_DIR"

# Step 3: Use the generated masks to filter tiles
echo "Filtering tiles based on background masks..."
python tile_filter.py "$TILES_DIR" "$MASK_DIR"

echo "Processing complete!"
echo "Tissue tiles have been saved to: $TISSUE_TILES_DIR"

# Step 4: Perform sliding window inference on tiles to generate plaque and tissue predictions
if $DEBUG; then
    echo "Inference in debug mode..."
    python inference_script.py "$TILES_DIR" "$INFERENCE_OUTPUT_DIR" 128 "$MODEL_WEIGHTS" --debug
else
    echo "Performing inference..."
    python inference_script.py "$TILES_DIR" "$INFERENCE_OUTPUT_DIR" 128 "$MODEL_WEIGHTS"
fi

# Step 5: Apply post processing to tissue mask and count plaques
echo "Applying post processing..."
python post_process_script.py "$POSTPROCESS_DIR" "$INFERENCE_OUTPUT_DIR"

echo "Done!"