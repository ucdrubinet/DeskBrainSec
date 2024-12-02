import os
from glob import glob
from PIL import Image
from csv import writer
from wsi import util

def sort_file_numbers(directory):
    contents = glob(os.path.join(directory, "*"))
    contents = [i.split('/')[-1] for i in contents]
    contents.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return contents

def process_slide_tiles(slide_id, tiles_base_dir, mask_dir):
    """
    Process tiles for a single slide using its mask
    
    Args:
        slide_id: ID of the slide to process
        tiles_base_dir: Base directory containing tiles folders
        mask_dir: Directory containing the masks
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
    print("Binary mask shape:", mask.shape)
    
    # Get dimensions of the mask
    mask_height, mask_width = mask.shape[:2]

    # Get the height of the WSI in tiles from num of directories in slide directory
    # Each directory represents a row of tiles
    slide_tiles_rows = sort_file_numbers(slide_tiles_dir)
    tiles_height = len(slide_tiles_rows)
    # Get the width of the WSI in tiles from num of imgs in first dir of the slide directory
    # Each img represents a column index in a row of tiles
    tiles_width = len(sort_file_numbers(os.path.join(slide_tiles_dir, slide_tiles_rows[0])))
    print("WSI height & width in tiles", tiles_height, tiles_width)

    # Create amn array to track if a tile has tissue (1) or is all background (0)
    tiles_is_tissue_present = []

    # Process each y coordinate directory (i.e. every row of tiles)
    for y_dir in slide_tiles_rows:
        y_path = os.path.join(slide_tiles_dir, y_dir)
        if not os.path.isdir(y_path):
            continue
        # convert directory name to y index
        y = int(y_dir)
        # init row in array
        tiles_is_tissue_present.append([])

        # Process each x coordinate tile
        for x_file in sort_file_numbers(y_path):
            if not x_file.endswith('.jpg'):
                continue
                
            x = int(os.path.splitext(x_file)[0])
            
            # Calculate corresponding position in mask
            # Scale x and y coordinates to mask size
            mask_x = int(x * mask_width / tiles_width)
            mask_y = int(y * mask_height / tiles_height)
            
            # append tissue status to array
            if mask[mask_y, mask_x] == 255:
                tiles_is_tissue_present[-1].append(1)
            else:
                tiles_is_tissue_present[-1].append(0)
    print(len(tiles_is_tissue_present))
    # write array with tissue presence status to CSV to the slide directory
    with open(os.path.join(tiles_base_dir, slide_id, 'tissue_presence.csv'), 'w') as f:
        writer(f).writerows(tiles_is_tissue_present)


def filter_tiles_using_masks(tiles_dir, mask_dir):
    """
    Filter tiles based on binary masks
    
    Args:
        tiles_dir: Directory containing original tiles
        mask_dir: Directory containing the binary masks
    """
    # Process each slide directory
    for slide_id in sorted(os.listdir(tiles_dir)):
        print(f"Processing slide: {slide_id}")
        process_slide_tiles(slide_id, tiles_dir, mask_dir)
        print(f"Completed processing slide: {slide_id}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Filter WSI tiles based on background masks')
    parser.add_argument('tiles_dir', help='Directory containing tiles')
    parser.add_argument('mask_dir', help='Directory containing background masks')
    
    args = parser.parse_args()
    
    print(args)
    
    # Process the tiles
    filter_tiles_using_masks(args.tiles_dir, args.mask_dir)

if __name__ == '__main__':
    main()
