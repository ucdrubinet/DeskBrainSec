import os
from PIL import Image
import numpy as np
import pandas as pd
from skimage import morphology, measure
from tqdm import tqdm
import cv2
import argparse
from inference_script import saveBrainSegImage, PLAQUE_RESULTS_DIR, TISSUE_RESULTS_DIR, IMG_DIR, NUMPY_DIR

PLAQUE_COUNTS_CSV = "WSI_CERAD_AREA.csv"

# Post-Processing BrainSeg - Jeff, Kolin, Wenda
def method_6(mask_img, down_factor=4):
    """Downsample => Area_opening (Remove local maxima) =>
    Swap index of GM and WM => Area_opening => Swap index back =>
    Area_closing => Morphological opening => Upsample"""
    def swap_GM_WM(arr):
        """Swap GM and WM in arr (swaps index 1 and index 2)"""
        arr_1 = (arr == 1)
        arr[arr == 2] = 1
        arr[arr_1] = 2
        del arr_1
        return arr

    mask_img = Image.fromarray(mask_img)
    width, height = mask_img.width, mask_img.height
    area_threshold_prop = 0.05
    area_threshold = int(area_threshold_prop * width * height // down_factor**2)

    # Downsample the image
    mask_arr = np.array(
        mask_img.resize((width // down_factor, height // down_factor), Image.NEAREST))
    del mask_img
    print('Finish downsampling')

    # Apply area_opening to remove local maxima with area < 20000 px
    mask_arr = morphology.area_opening(mask_arr, area_threshold=3200 // down_factor**2)
    print('Finish area_opening #1')

    # Swap index of GM and WM
    mask_arr = swap_GM_WM(mask_arr)
    print('Finish swapping index')

    # Apply area_opening to remove local maxima with area < 20000 px
    mask_arr = morphology.area_opening(mask_arr, area_threshold=3200 // down_factor**2)
    print('Finish area_opening #2')

    # Swap index back
    mask_arr = swap_GM_WM(mask_arr)
    print('Finish swapping index back')

    # Apply area_closing to remove local minima with area < 12500 px
    mask_arr = morphology.area_closing(mask_arr, area_threshold=2000 // down_factor**2)
    print('Finish area_closing')

    # Apply remove_small_objects to remove tissue residue with area < 0.05 * width * height
    tissue_arr = morphology.remove_small_objects(mask_arr > 0, min_size=area_threshold,
                                                 connectivity=2)
    mask_arr[np.invert(tissue_arr)] = 0
    del tissue_arr
    print('Finish remove_small_objects')

    # Apply opening with disk-shaped kernel (r=8) to smooth boundary
    mask_arr = morphology.opening(mask_arr, footprint=morphology.disk(radius=32 // down_factor))
    print('Finish morphological opening')

    # Upsample the output
    mask_arr = np.array(Image.fromarray(mask_arr).resize((width, height), Image.NEAREST))
    print('Finish upsampling')

    return mask_arr

def postprocess_tissue(postprocess_dir, inference_dir):
    tissue_numpy_pre_dir = os.path.join(inference_dir, TISSUE_RESULTS_DIR, NUMPY_DIR)
    tissue_numpy_post_dir = os.path.join(postprocess_dir, TISSUE_RESULTS_DIR, NUMPY_DIR)
    if not os.path.exists(tissue_numpy_post_dir):
        os.makedirs(tissue_numpy_post_dir)
    tissue_img_post_dir = os.path.join(postprocess_dir, TISSUE_RESULTS_DIR, IMG_DIR)
    if not os.path.exists(tissue_img_post_dir):
        os.makedirs(tissue_img_post_dir)

    filenames = sorted(os.listdir(tissue_numpy_pre_dir))
    filenames = [os.path.splitext(file)[0] for file in filenames]

    for filename in tqdm(filenames) :
        fileLoc = os.path.join(tissue_numpy_pre_dir, filename + ".npy")
        print("Loading: " + fileLoc)
        seg_pic = np.load(fileLoc)
        processed = method_6(seg_pic)
        np.save(os.path.join(tissue_numpy_post_dir, filename + ".npy"), processed)
        saveBrainSegImage(processed, os.path.join(tissue_img_post_dir, filename + '.png'))
        
# Post-Processing to count Plaques
def count_blobs(mask, threshold=1500):
    labels = measure.label(mask, connectivity=2, background=0)
    img_mask = np.zeros(mask.shape, dtype='uint8')
    labeled_mask = np.zeros(mask.shape, dtype='uint16')
    sizes = []
    
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(mask.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = np.count_nonzero(labelMask)
        
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > threshold:
            sizes.append(numPixels)
            img_mask = cv2.add(img_mask, labelMask)
            
            # Save confirmed unique location of plaque
            labeled_mask[labels==label] = label

    return sizes, img_mask, labeled_mask

# from PIL import Image
def saveMask(mask_array, save_dir):
    
    mask_array = np.repeat(mask_array[:,:, np.newaxis], 3, axis=2)
    
    # mask_array[:,:,0] = RED, mask_array[:,:,1] = Green, mask_array[:,:,2] = Blue
    idx = np.where(mask_array[:,:,0] == 255)  # Index of label 1 (WM)

    # For label 0, leave as black color
    # For label 1, set to cyan color: R0G255B255
    mask_array[:,:,0].flat[np.ravel_multi_index(idx, mask_array[:,:,0].shape)] = 0
    mask_array[:,:,1].flat[np.ravel_multi_index(idx, mask_array[:,:,1].shape)] = 255
    mask_array[:,:,2].flat[np.ravel_multi_index(idx, mask_array[:,:,2].shape)] = 255

    mask_array = mask_array.astype(np.uint8) # PIL save only accepts uint8 {0,..,255}
    save_img = Image.fromarray(mask_array, 'RGB')
    save_img.save(save_dir)
    print("Saved at: " + save_dir)

from skimage.color import hsv2rgb
from PIL import Image

def saveUniqueMaskImage(maskArray, save_dir):
    '''
    Plots post-processed detected Plaques
    with the diversity of Colour distingushing
    the density of Plaques
    
    ie. More Diversity of Colour
    == More Plaque Count for that certain Plaque type
    
    Inputs:
        maskArray = Numpy Array containing Unique plaque
        save_dir  = String for Save Directory
    '''
    
    max_val = np.amax(np.unique(maskArray))
    maskArray = np.asarray(maskArray, dtype=np.float64)
    maskArray = np.repeat(maskArray[:,:, np.newaxis], 3, axis=2)

    for label in np.unique(maskArray) :

        # For label 0, leave as black color (BG)
        if label == 0:
            continue

        idx = np.where(maskArray[:,:,0] == label) 

        # For label, create HSV space based on unique labels
        maskArray[:,:,0].flat[np.ravel_multi_index(idx, maskArray[:,:,0].shape)] = label / max_val
        maskArray[:,:,1].flat[np.ravel_multi_index(idx, maskArray[:,:,1].shape)] = label % max_val
        maskArray[:,:,2].flat[np.ravel_multi_index(idx, maskArray[:,:,2].shape)] = 1

    rgb_maskArray = hsv2rgb(maskArray)
    rgb_maskArray = rgb_maskArray * 255
    rgb_maskArray = rgb_maskArray.astype(np.uint8) # PIL save only accepts uint8 {0,..,255}
    
    save_img = Image.fromarray(rgb_maskArray, 'RGB')
    save_img.save(save_dir)
    print("Saved at: " + save_dir)

def classify_blobs(labeled_mask, seg_area):
    """
    Classifies each certain plaques according to each
    Segmentation Area and gives each count
    
    Input:
        labeled_mask (NumPy Array): 
            contains plaque information 
            Note: See count_blobs()'s 
            labeled_mask output for more info
        
        seg_area (NumPy Array):
            contains segmentation information
            based on BrainSeg's classification
            
    Output:
        count_dict (Dictionary):
            contains number of plaques at each
            segmentaion area
            
        Other Variables:
            - Background Count
            - WM Count
            - GM Count
            - Unclassified Count
    """
    
    # 0: Background, 1: WM, 2: GM
    count_dict = {0: 0, 1: 0, 2: 0, "uncounted": 0}
    # Loop over unique components
    for label in np.unique(labeled_mask) :
        if label == 0:
            continue
            
        plaque_loc = np.where(labeled_mask == label)
        plaque_area = seg_area[plaque_loc]
        indexes, counts = np.unique(plaque_area, return_counts=True)
        class_idx = indexes[np.where(counts == np.amax(counts))]
        
        try:
            class_idx = class_idx.item()
            count_dict[class_idx] += 1
                
        except:
            count_dict["uncounted"] += 1
            
    return count_dict, count_dict[0], count_dict[1], count_dict[2], count_dict["uncounted"]

def postprocess_plaque(postprocess_dir, inference_dir, confidence_thresholds, pixel_thresholds):
    plaque_numpy_pre_dir = os.path.join(inference_dir, PLAQUE_RESULTS_DIR, NUMPY_DIR)
    plaque_numpy_post_dir = os.path.join(postprocess_dir, PLAQUE_RESULTS_DIR, NUMPY_DIR)
    if not os.path.exists(plaque_numpy_post_dir):
        os.makedirs(plaque_numpy_post_dir)
    plaque_img_post_dir = os.path.join(postprocess_dir, PLAQUE_RESULTS_DIR, IMG_DIR)
    if not os.path.exists(plaque_img_post_dir):
        os.makedirs(plaque_img_post_dir)
    tissue_numpy_post_dir = os.path.join(postprocess_dir, TISSUE_RESULTS_DIR, NUMPY_DIR)
    plaque_csv_file_path = os.path.join(postprocess_dir, PLAQUE_RESULTS_DIR, PLAQUE_COUNTS_CSV)
    
    print("Creating new CSV for plaque counts")
    filenames = sorted(os.listdir(plaque_numpy_pre_dir))
    filenames = [os.path.splitext(file)[0] for file in filenames]
    plaque_count_df = pd.DataFrame({"WSI_ID": filenames})
    plaque_class = ['cored', 'diffuse', 'caa']

    # Post-process Plaque Confidence
    # and Count Plaques at each region

    # For each plaque class
    for index in range(len(plaque_class)):
        preds = np.zeros(len(plaque_count_df))
        confidence_threshold = confidence_thresholds[index]
        pixel_threshold = pixel_thresholds[index]
        
        bg = np.zeros(len(plaque_count_df))
        wm = np.zeros(len(plaque_count_df))
        gm = np.zeros(len(plaque_count_df))
        unknowns = np.zeros(len(plaque_count_df))

        for i, WSIname in enumerate(tqdm(filenames)):
            plaque_numpy_pre_path = os.path.join(plaque_numpy_pre_dir, WSIname + '.npy')
            plaque_numpy = np.load(plaque_numpy_pre_path)
            tissue_numpy_post_path = os.path.join(tissue_numpy_post_dir, WSIname + '.npy')
            tissue_numpy = np.load(tissue_numpy_post_path)

            mask = plaque_numpy[index] > confidence_threshold
            mask = mask.astype(np.float32)

            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

            # Apply morphological closing, then opening operations 
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            labels, img_mask, labeled_mask = count_blobs(closing, threshold=pixel_threshold)
            counts, bg[i], wm[i], gm[i], unknowns[i] = classify_blobs(labeled_mask, tissue_numpy)
        
            save_img = os.path.join(plaque_img_post_dir , f"{WSIname}_{plaque_class[index]}.png")
            save_np = os.path.join(plaque_numpy_post_dir, f"{WSIname}_{plaque_class[index]}.npy")
            np.save(save_np, labeled_mask)
            saveUniqueMaskImage(labeled_mask, save_img) # To show Colored Result
            #saveMask(img_mask, save_img)  # To show Classification Result
            
            preds[i] = len(labels)
            

        print(confidence_threshold, pixel_threshold)

        plaque_count_df['CNN_{}_count'.format(plaque_class[index])] = preds
        plaque_count_df['BG_{}_count'.format(plaque_class[index])] = bg
        plaque_count_df['GM_{}_count'.format(plaque_class[index])] = gm
        plaque_count_df['WM_{}_count'.format(plaque_class[index])] = wm
        plaque_count_df['{}_no-count'.format(plaque_class[index])] = unknowns
        

    plaque_count_df.to_csv(plaque_csv_file_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("postprocess_dir", help="path to directory to put post-processing outputs", type=str)
    parser.add_argument("inference_dir", help="path to directory with inference outputs", type=str)
    args = parser.parse_args()

    # two hyperparameters (For Plaque-Counting)
    confidence_thresholds = [0.1, 0.95, 0.9]
    pixel_thresholds = [100, 1, 200]

    if not os.path.exists(args.postprocess_dir):
        os.makedirs(args.postprocess_dir)

    postprocess_tissue(args.postprocess_dir, args.inference_dir)
    postprocess_plaque(args.postprocess_dir, args.inference_dir, confidence_thresholds, pixel_thresholds)

if __name__ == "__main__":
    print("hi")
    main()