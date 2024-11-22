import os
import numpy as np
from PIL import Image
import shutil
from wsi import util

def process_slide_tiles(slide_id, tiles_base_dir, mask_dir, output_base_dir):
    """
    Process tiles for a single slide using its mask
    
    Args:
        slide_id: ID of the slide to process
        tiles_base_dir: Base directory containing tiles folders
        mask_dir: Directory containing the masks
        output_base_dir: Base directory for output tissue tiles
    """
    # Construct paths
    slide_tiles_dir = os.path.join(tiles_base_dir, slide_id, '0')
    mask_path = os.path.join(mask_dir, f"{slide_id}_filter.png")
    print(mask_path)
    
    if not os.path.exists(mask_path):
        print(f"Warning: No mask found for slide {slide_id}")
        return
        
    # Load mask
    mask = util.pil_to_np_rgb(Image.open(mask_path), display_info=False)
    print(mask.shape)
    print(mask)
    currow = mask[80].tolist()
    print(currow)
    
    # Get dimensions of the mask
    mask_height, mask_width = mask.shape[:2]
    tiles_height, tiles_width = len(os.listdir(slide_tiles_dir)), len(os.listdir(os.path.join(slide_tiles_dir, os.listdir(slide_tiles_dir)[0])))
    print(tiles_height, tiles_width)
    
    # Process each y coordinate directory
    for y_dir in os.listdir(slide_tiles_dir):
        y_path = os.path.join(slide_tiles_dir, y_dir)
        if not os.path.isdir(y_path):
            continue
            
        y = int(y_dir)
        
        # Process each x coordinate tile
        for x_file in os.listdir(y_path):
            if not x_file.endswith('.jpg'):
                continue
                
            x = int(os.path.splitext(x_file)[0])
            
            # Get tile path
            tile_path = os.path.join(y_path, x_file)
            
            # Calculate corresponding position in mask
            # Scale x and y coordinates to mask size
            mask_x = int(x * mask_width / tiles_width)
            mask_y = int(y * mask_height / tiles_height)
            if mask[mask_y, mask_x] == 255:
                # Create output directory structure
                output_x_dir = os.path.join(output_base_dir, slide_id, '0', y_dir)
                os.makedirs(output_x_dir, exist_ok=True)
                
                # Copy tile to output directory
                shutil.copy2(
                    tile_path,
                    os.path.join(output_x_dir, x_file)
                )

def filter_tiles_using_masks(tiles_dir, mask_dir, tissue_tiles_dir):
    """
    Filter tiles based on binary masks
    
    Args:
        tiles_dir: Directory containing original tiles
        mask_dir: Directory containing the binary masks
        tissue_tiles_dir: Directory to save tiles containing tissue
    """
    # Process each slide directory
    for slide_id in os.listdir(tiles_dir):
        print("A SLIDE ID IS", slide_id)
        if not os.path.isdir(os.path.join(tiles_dir, slide_id)):
            continue
        
        try:
            shutil.rmtree(tissue_tiles_dir)
        except FileNotFoundError: pass
        print(f"Processing slide: {slide_id}")
        process_slide_tiles(slide_id, tiles_dir, mask_dir, tissue_tiles_dir)
        print(f"Completed processing slide: {slide_id}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Filter WSI tiles based on background masks')
    parser.add_argument('tiles_dir', help='Directory containing tiles')
    parser.add_argument('mask_dir', help='Directory containing background masks')
    parser.add_argument('tissue_tiles_dir', help='Directory to save tissue tiles')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.tissue_tiles_dir, exist_ok=True)
    print(args)
    
    # Process the tiles
    filter_tiles_using_masks(args.tiles_dir, args.mask_dir, args.tissue_tiles_dir)

if __name__ == '__main__':
    main()
