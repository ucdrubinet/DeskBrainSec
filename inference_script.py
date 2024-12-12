import time, os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from training.model import *
from training.datasets import *
from inference.heatmap import *

PLAQUE_RESULTS_DIR = "plaque"
TISSUE_RESULTS_DIR = "tissue"
INFERENCE_NUMPY_DIR = "numpy"
INFERENCE_IMG_DIR = "images"
TISSUE_PRESENCE_FILE = "tissue_presence.csv"

def saveBrainSegImage(nums, save_dir) :
    """
    Converts 2D array with {0,1,2} into RGB
     to determine different segmentation areas
     and saves image at given directory
    
    Input:
       nums: 2D-NumPy Array containing classification
       save_dir: string indicating save location
    """ 
    
    nums = np.repeat(nums[:,:, np.newaxis], 3, axis=2)
    
    # nums[:,:,0] = RED, nums[:,:,1] = Green, nums[:,:,2] = Blue
    idx_1 = np.where(nums[:,:,0] == 1)  # Index of label 1 (WM)
    idx_2 = np.where(nums[:,:,0] == 2)  # Index of label 2 (GM)

    # For label 0, leave as black color
    # For label 1, set to yellow color: R255G255B0 (WM)
    nums[:,:,0].flat[np.ravel_multi_index(idx_1, nums[:,:,0].shape)] = 255
    nums[:,:,1].flat[np.ravel_multi_index(idx_1, nums[:,:,1].shape)] = 255
    nums[:,:,2].flat[np.ravel_multi_index(idx_1, nums[:,:,2].shape)] = 0
    # For label 2, set to cyan color: R0G255B255 (GM)
    nums[:,:,0].flat[np.ravel_multi_index(idx_2, nums[:,:,0].shape)] = 0
    nums[:,:,1].flat[np.ravel_multi_index(idx_2, nums[:,:,1].shape)] = 255
    nums[:,:,2].flat[np.ravel_multi_index(idx_2, nums[:,:,2].shape)] = 255

    nums = nums.astype(np.uint8) # PIL save only accepts uint8 {0,..,255}
    save_img = Image.fromarray(nums, 'RGB')
    save_img.save(save_dir)
    print("Saved at: " + save_dir)

def plot_heatmap(final_output, save_path) :
    """
    Plots Confidence Heatmap of Plaques = [0,1]
    
    Inputs:
        final_output (NumPy array of 
        3*img_height*height_width) :
            Contains Plaque Confidence with each axis
            representing different types of plaque
            
    Outputs:
        Subplots containing Plaque Confidences
    """
    # TODO: modify to save instead of showing figure
    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(221)
    ax.set_title('cored')

    im = ax.imshow(final_output[0], cmap=plt.cm.get_cmap('viridis', 20), vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])

    ax = fig.add_subplot(222)
    ax.set_title('diffuse')

    im = ax.imshow(final_output[1], cmap=plt.cm.get_cmap('viridis', 20), vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])

    ax = fig.add_subplot(223)
    ax.set_title('CAA')
    im = ax.imshow(final_output[2], cmap=plt.cm.get_cmap('viridis', 20), vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])

def inference_loop(filenames, model, tiles_dir, output_dir, stride, **kwargs):
    for slidename in filenames:
        print("Now processing: ", slidename)
        # Check if directory contains tiles
        if not os.path.exists(os.path.join(tiles_dir, slidename, "0", "0", "0.jpg")):
            print("Tile directory empty. Skipping slide.")
            continue
        # Check if tiles are square and get the dimension if so
        sample_tile = Image.open(os.path.join(tiles_dir, slidename, "0", "0", "0.jpg"))
        width, height = sample_tile.size
        if width != height:
            print("Tile is not a square. Skipping slide.")
            continue
        tile_size = width

        # Check if tile folder contains a CSV with tissue / background masks
        if not os.path.exists(os.path.join(tiles_dir, slidename, TISSUE_PRESENCE_FILE)):
            print("No tissue mask present. Skipping slide.")
            continue
    
        # Read tissue presence map into list
        with open(os.path.join(tiles_dir, slidename, TISSUE_PRESENCE_FILE), newline='') as f:
            tissue_is_present = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))

        row_nums = (len(tissue_is_present))
        col_nums = len(tissue_is_present[0])   
        
        # Initialize outputs accordingly:
        heatmap_res = tile_size // stride
        # TODO: remove batch_size variable. For now, set it to heatmap res, so every batch processes a row of tiles
        batch_size = heatmap_res
        plaque_output = np.zeros((3, heatmap_res*row_nums, heatmap_res*col_nums))
        seg_output = np.zeros((heatmap_res*row_nums, heatmap_res*col_nums), dtype=np.uint8)

        # set model to evaluation mode
        model.eval()
        start_time = time.perf_counter() # To evaluate Time taken per inference

        # TODO: remove debug condition
        if 'debug' in kwargs and kwargs['debug']:
            row_range = [(row_nums // 2) - 1, row_nums // 2]
            col_range = [(col_nums // 2) - 1, col_nums // 2]
        else:
            row_range = range(row_nums)
            col_range = range(col_nums)

        for row in row_range:
            for col in col_range:
                print(f"Tile ({row},{col})")
                # if tile has no tissue, skip
                if not tissue_is_present[row][col]:
                    print("No tissue. Skipping.")
                    continue
                print("Contains tissue.")
                # Load tile with padding into dataset
                image_datasets = HeatmapDataset(os.path.join(tiles_dir, slidename, '0'), row, col, stride=stride)
                dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=heatmap_res,
                                                    shuffle=False, num_workers=2)
                
                running_plaques = torch.Tensor(0)
                # create numpy array of heatmap
                output_class = np.zeros((heatmap_res, heatmap_res), dtype=np.uint8)
                
                for idx, data in enumerate(dataloader):
                    # get plaque and tissue outputs
                    p_out, t_out = model(data)
                    # binarize output of plaque prediction and save
                    p_preds = (torch.nn.functional.sigmoid(p_out.data) > 0.5)
                    running_plaques = torch.cat([running_plaques, p_preds])
                    _, t_preds = torch.max(t_out.data, 1)
                    i = (idx // (heatmap_res//batch_size))
                    j = (idx % (heatmap_res//batch_size))
                    output_class[i,j*batch_size:(j+1)*batch_size] = t_preds.data
            
                # Final Outputs of Brain Segmentation
                seg_output[row*heatmap_res:(row+1)*heatmap_res, col*heatmap_res:(col+1)*heatmap_res] = output_class
                
                # Final Outputs of Plaque Detection:
                cored = np.asarray(running_plaques[:,0]).reshape(heatmap_res, heatmap_res)
                diffuse = np.asarray(running_plaques[:,1]).reshape(heatmap_res, heatmap_res)
                caa = np.asarray(running_plaques[:,2]).reshape(heatmap_res,heatmap_res)
                
                plaque_output[0, row*heatmap_res:(row+1)*heatmap_res, col*heatmap_res:(col+1)*heatmap_res] = cored
                plaque_output[1, row*heatmap_res:(row+1)*heatmap_res, col*heatmap_res:(col+1)*heatmap_res] = diffuse
                plaque_output[2, row*heatmap_res:(row+1)*heatmap_res, col*heatmap_res:(col+1)*heatmap_res] = caa

                seg_output[row*heatmap_res:(row+1)*heatmap_res, col*heatmap_res:(col+1)*heatmap_res] = output_class

        # Saving Confidence=[0,1] for Plaque Detection
        if not os.path.exists(os.path.join(output_dir, PLAQUE_RESULTS_DIR, INFERENCE_NUMPY_DIR)):
            os.makedirs(os.path.join(output_dir, PLAQUE_RESULTS_DIR, INFERENCE_NUMPY_DIR))
        np.save(os.path.join(output_dir, PLAQUE_RESULTS_DIR, INFERENCE_NUMPY_DIR, slidename), plaque_output)
        
        # Saving BrainSeg Classification={0,1,2}
        if not os.path.exists(os.path.join(output_dir, TISSUE_RESULTS_DIR, INFERENCE_NUMPY_DIR)):
            os.makedirs(os.path.join(output_dir, TISSUE_RESULTS_DIR, INFERENCE_NUMPY_DIR))
        np.save(os.path.join(output_dir, TISSUE_RESULTS_DIR, INFERENCE_NUMPY_DIR, slidename), seg_output)
        if not os.path.exists(os.path.join(output_dir, TISSUE_RESULTS_DIR, INFERENCE_IMG_DIR)):
            os.makedirs(os.path.join(output_dir, TISSUE_RESULTS_DIR, INFERENCE_IMG_DIR))
        saveBrainSegImage(seg_output, os.path.join(output_dir, TISSUE_RESULTS_DIR, INFERENCE_IMG_DIR, slidename + '.png'))
        
        # Time Statistics for Inference
        end_time = time.perf_counter()
        print("Time to process " + slidename + ": ", end_time-start_time, "sec")

# CLI Args
# SAVE_PLAQ_DIR (for numpy of plaque confidence)
# SAVE_IMG_DIR (for segmentation pngs)
# SAVE_NP_DIR (for np version of segmentation)
# IMG_SIZE (size of each tile image of the WSI) --> rename to TILE_SIZE
# STRIDE (how many pixels to jump right or down to center each 256 pixel window)
# Don't set this to more than 128, because then you're not checking some pixels
# TILE DIR (directory of tile images) --> tile_dir
# TILE_PRESENCE_FILE (path to csv defining tissue / background mask) --> check for "tissue_presence.csv" in file structure

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tiles_dir", help="path to directory with PNG tiles of the WSI", type=str)
    parser.add_argument("output_dir", help="path to directory to put inference outputs", type=str)
    parser.add_argument("stride", help="number of pixels to jump when doing sliding window inference (int between 0 and 128)", type=int)
    parser.add_argument("model_weights", help="path to model weights .pth file", type=str)

    args = parser.parse_args()

    filenames = os.listdir(args.tiles_dir)
    print(filenames)
    model = PlaqueTissueClassifier().to('cpu')
    checkpoint = torch.load(args.model_weights, map_location=torch.device('cpu'))

    new_state_dict = dict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '') # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    inference_loop(filenames, model, args.tiles_dir, args.output_dir, args.stride)

if __name__ == "__main__":
    print("Hi")
    main()