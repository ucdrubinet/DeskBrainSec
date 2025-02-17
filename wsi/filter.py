# Adapted from https://github.com/deroneriksson/python-wsi-preprocessing

import os
import numpy as np
import scipy.ndimage as scipy_img
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation
from functools import reduce
from PIL import Image

import wsi.util as util
from wsi.util import Time


def filter_rgb_to_grayscale(np_img, output_type="uint8", display_info=True):
  """
  Convert an RGB NumPy array to a grayscale NumPy array.

  Shape (h, w, c) to (h, w).

  Args:
    np_img: RGB Image as a NumPy array.
    output_type: Type of array to return (float or uint8)
    display_info: Boolean flag to print NumPy array info

  Returns:
    Grayscale image as NumPy array with shape (h, w).
  """
  if display_info: 
    t = Time()
  rgb_ratio = [0.299, 0.587, 0.114]
  #rgb_ratio = [0.2125, 0.7154, 0.0721]
  grayscale = np.dot(np_img[..., :3], rgb_ratio)
  if output_type != "float":
    grayscale = grayscale.astype("uint8")
  if display_info: 
    util.np_info(grayscale, "Gray", t.elapsed())
  return grayscale


def filter_complement(np_img, output_type="uint8", display_info=True):
  """
  Obtain the complement of an image as a NumPy array.

  Args:
    np_img: Image as a NumPy array.
    type: Type of array to return (float or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    Complement image as Numpy array.
  """
  if display_info:
    t = Time()
  if output_type == "float":
    complement = 1.0 - np_img
  else:
    complement = 255 - np_img
  if display_info:
    util.np_info(complement, "Complement", t.elapsed())
  return complement


def filter_hysteresis_threshold(np_img, low=50, high=100, output_type="uint8", display_info=True):
  """
  Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.

  Args:
    np_img: Image as a NumPy array.
    low: Low threshold.
    high: High threshold.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
  """
  if display_info:
    t = Time()
  hyst = sk_filters.apply_hysteresis_threshold(np_img, low, high)
  if output_type == "bool":
    pass
  elif output_type == "float":
    hyst = hyst.astype(float)
  else:
    hyst = (255 * hyst).astype("uint8")
  if display_info:
    util.np_info(hyst, "Hysteresis Threshold", t.elapsed())
  return hyst


def filter_otsu_threshold(np_img, output_type="uint8", display_info=True):
  """
  Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.

  Args:
    np_img: Image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
  """
  if display_info:
    t = Time()
  otsu_thresh_value = sk_filters.threshold_otsu(np_img)
  otsu = (np_img > otsu_thresh_value)
  if output_type == "bool":
    pass
  elif output_type == "float":
    otsu = otsu.astype(float)
  else:
    otsu = otsu.astype("uint8") * 255
  if display_info:
    util.np_info(otsu, "Otsu Threshold", t.elapsed())
  return otsu


def filter_local_otsu_threshold(np_img, disk_size=3, output_type="uint8", display_info=True):
  """
  Compute local Otsu threshold for each pixel and return binary image based on pixels being less than the
  local Otsu threshold.

  Args:
    np_img: Image as a NumPy array.
    disk_size: Radius of the disk structuring element used to compute the Otsu threshold for each pixel.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) where local Otsu threshold values have been applied to original image.
  """
  if display_info:
    t = Time()
  local_otsu = sk_filters.rank.otsu(np_img, sk_morphology.disk(disk_size))
  if output_type == "bool":
    pass
  elif output_type == "float":
    local_otsu = local_otsu.astype(float)
  else:
    local_otsu = local_otsu.astype("uint8") * 255
  if display_info:
    util.np_info(local_otsu, "Otsu Local Threshold", t.elapsed())
  return local_otsu


def filter_entropy(np_img, neighborhood=9, threshold=5, output_type="uint8", display_info=True):
  """
  Filter image based on entropy (complexity).

  Args:
    np_img: Image as a NumPy array.
    neighborhood: Neighborhood size (defines height and width of 2D array of 1's).
    threshold: Threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
  """
  if display_info:
    t = Time()
  entr = sk_filters.rank.entropy(np_img, np.ones((neighborhood, neighborhood))) > threshold
  if output_type == "bool":
    pass
  elif output_type == "float":
    entr = entr.astype(float)
  else:
    entr = entr.astype("uint8") * 255
  if display_info:
    util.np_info(entr, "Entropy", t.elapsed())
  return entr

def filter_entropy_scale(np_img, neighborhood_scale=3, threshold=5, output_type="uint8", display_info=True):
  """
  Wrapper of filter_entropy() with neighborhood size proportional to np_img shape.

  Args:
    np_img: Image as a NumPy array.
    neighborhood_scale: Pixel width of neighborhood per 100 pixels.
    threshold: Threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
  """
  avg_width = (np_img.shape[0] + np_img.shape[1]) / 2
  neighborhood = int(avg_width / 100) * neighborhood_scale
  return filter_entropy(np_img, neighborhood, threshold, output_type, display_info)

def filter_canny(np_img, sigma=1, low_threshold=0, high_threshold=25, output_type="uint8", display_info=True):
  """
  Filter image based on Canny algorithm edges.

  Args:
    np_img: Image as a NumPy array.
    sigma: Width (std dev) of Gaussian.
    low_threshold: Low hysteresis threshold value.
    high_threshold: High hysteresis threshold value.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) representing Canny edge map (binary image).
  """
  if display_info:
    t = Time()
  can = sk_feature.canny(np_img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    can = can.astype(float)
  else:
    can = can.astype("uint8") * 255
  if display_info:
    util.np_info(can, "Canny Edges", t.elapsed())
  return can


def mask_percent(np_img):
  """
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is masked.
  """
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage


def tissue_percent(np_img):
  """
  Determine the percentage of a NumPy array that is tissue (not masked).

  Args:
    np_img: Image as a NumPy array.

  Returns:
    The percentage of the NumPy array that is tissue.
  """
  return 100 - mask_percent(np_img)


def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, 
                                output_type="uint8", display_info=True):
  """
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Minimum size of small object to remove.
    avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8).
  """
  if display_info:
    t = Time()
  rem_sm = np_img.astype(bool)  # make sure mask is boolean
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  mask_percentage = mask_percent(rem_sm)
  if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
    new_min_size = min_size / 2
    print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
      mask_percentage, overmask_thresh, min_size, new_min_size))
    rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
  np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  if display_info:
    util.np_info(np_img, "Remove Small Objs", t.elapsed())
  return np_img


def filter_remove_small_holes(np_img, min_size=3000, output_type="uint8", display_info=True):
  """
  Filter image to remove small holes less than a particular size.

  Args:
    np_img: Image as a NumPy array of type bool.
    min_size: Remove small holes below this size.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8).
  """
  if display_info:
    t = Time()
  rem_sm = sk_morphology.remove_small_holes(np_img, area_threshold=min_size)

  if output_type == "bool":
    pass
  elif output_type == "float":
    rem_sm = rem_sm.astype(float)
  else:
    rem_sm = rem_sm.astype("uint8") * 255

  if display_info:
    util.np_info(rem_sm, "Remove Small Holes", t.elapsed())
  return rem_sm


def filter_contrast_stretch(np_img, low=40, high=60, display_info=True):
  """
  Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in
  a specified range.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    low: Range low value (0 to 255).
    high: Range high value (0 to 255).
    display_info: Boolean flag to print NumPy array info

  Returns:
    Image as NumPy array with contrast enhanced.
  """
  if display_info:
    t = Time()
  low_p, high_p = np.percentile(np_img, (low * 100 / 255, high * 100 / 255))
  contrast_stretch = sk_exposure.rescale_intensity(np_img, in_range=(low_p, high_p))
  if display_info:
    util.np_info(contrast_stretch, "Contrast Stretch", t.elapsed())
  return contrast_stretch


def filter_histogram_equalization(np_img, nbins=256, output_type="uint8", display_info=True):
  """
  Filter image (gray or RGB) using histogram equalization to increase contrast in image.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    nbins: Number of histogram bins.
    output_type: Type of array to return (float or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
     NumPy array (float or uint8) with contrast enhanced by histogram equalization.
  """
  if display_info:
    t = Time()
  # if uint8 type and nbins is specified, convert to float so that nbins can be a value besides 256
  if np_img.dtype == "uint8" and nbins != 256:
    np_img = np_img / 255
  hist_equ = sk_exposure.equalize_hist(np_img, nbins=nbins)
  if output_type == "float":
    pass
  else:
    hist_equ = (hist_equ * 255).astype("uint8")
  if display_info:
    util.np_info(hist_equ, "Hist Equalization", t.elapsed())
  return hist_equ


def filter_adaptive_equalization(np_img, nbins=256, clip_limit=0.01, output_type="uint8", display_info=True):
  """
  Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions
  is enhanced.

  Args:
    np_img: Image as a NumPy array (gray or RGB).
    nbins: Number of histogram bins.
    clip_limit: Clipping limit where higher value increases contrast.
    output_type: Type of array to return (float or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
     NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
  """
  if display_info:
    t = Time()
  adapt_equ = sk_exposure.equalize_adapthist(np_img, nbins=nbins, clip_limit=clip_limit)
  if output_type == "float":
    pass
  else:
    adapt_equ = (adapt_equ * 255).astype("uint8")
  if display_info:
    util.np_info(adapt_equ, "Adapt Equalization", t.elapsed())
  return adapt_equ


def filter_local_equalization(np_img, disk_size=50, display_info=True):
  """
  Filter image (gray) using local equalization, which uses local histograms based on the disk structuring element.

  Args:
    np_img: Image as a NumPy array.
    disk_size: Radius of the disk structuring element used for the local histograms
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array with contrast enhanced using local equalization.
  """
  if display_info:
    t = Time()
  local_equ = sk_filters.rank.equalize(np_img, selem=sk_morphology.disk(disk_size))
  if display_info:
    util.np_info(local_equ, "Local Equalization", t.elapsed())
  return local_equ


def filter_rgb_to_hed(np_img, output_type="uint8", display_info=True):
  """
  Filter RGB channels to HED (Hematoxylin - Eosin - Diaminobenzidine) channels.

  Args:
    np_img: RGB image as a NumPy array.
    output_type: Type of array to return (float or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (float or uint8) with HED channels.
  """
  if display_info:
    t = Time()
  hed = sk_color.rgb2hed(np_img)
  if output_type == "float":
    hed = sk_exposure.rescale_intensity(hed, out_range=(0.0, 1.0))
  else:
    hed = (sk_exposure.rescale_intensity(hed, out_range=(0, 255))).astype("uint8")

  if display_info:
    util.np_info(hed, "RGB to HED", t.elapsed())
  return hed


def filter_rgb_to_hsv(np_img, display_info=True):
  """
  Filter RGB channels to HSV (Hue, Saturation, Value).

  Args:
    np_img: RGB image as a NumPy array.
    display_info: Boolean flag to print NumPy array info

  Returns:
    Image as NumPy array in HSV representation.
  """

  if display_info:
    t = Time()
  hsv = sk_color.rgb2hsv(np_img)
  if display_info:
    util.np_info(hsv, "RGB to HSV", t.elapsed())
  return hsv


def filter_hsv_to_h(hsv, output_type="int", display_info=True):
  """
  Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
  values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
  https://en.wikipedia.org/wiki/HSL_and_HSV

  Args:
    hsv: HSV image as a NumPy array.
    output_type: Type of array to return (float or int).
    display_info: If True, display NumPy array info and filter time.

  Returns:
    Hue values (float or int) as a 1-dimensional NumPy array.
  """
  if display_info:
    t = Time()
  h = hsv[:, :, 0]
  h = h.flatten()
  if output_type == "int":
    h *= 360
    h = h.astype("int")
  if display_info:
    util.np_info(hsv, "HSV to H", t.elapsed())
  return h


def filter_hsv_to_s(hsv):
  """
  Experimental HSV to S (saturation).

  Args:
    hsv:  HSV image as a NumPy array.

  Returns:
    Saturation values as a 1-dimensional NumPy array.
  """
  s = hsv[:, :, 1]
  s = s.flatten()
  return s


def filter_hsv_to_v(hsv):
  """
  Experimental HSV to V (value).

  Args:
    hsv:  HSV image as a NumPy array.

  Returns:
    Value values as a 1-dimensional NumPy array.
  """
  v = hsv[:, :, 2]
  v = v.flatten()
  return v


def filter_hed_to_hematoxylin(np_img, output_type="uint8", display_info=True):
  """
  Obtain Hematoxylin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.

  Args:
    np_img: HED image as a NumPy array.
    output_type: Type of array to return (float or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array for Hematoxylin channel.
  """
  if display_info:
    t = Time()
  hema = np_img[:, :, 0]
  if output_type == "float":
    hema = sk_exposure.rescale_intensity(hema, out_range=(0.0, 1.0))
  else:
    hema = (sk_exposure.rescale_intensity(hema, out_range=(0, 255))).astype("uint8")
  if display_info:
    util.np_info(hema, "HED to Hematoxylin", t.elapsed())
  return hema


def filter_hed_to_eosin(np_img, output_type="uint8", display_info=True):
  """
  Obtain Eosin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.

  Args:
    np_img: HED image as a NumPy array.
    output_type: Type of array to return (float or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array for Eosin channel.
  """
  if display_info:
    t = Time()
  eosin = np_img[:, :, 1]
  if output_type == "float":
    eosin = sk_exposure.rescale_intensity(eosin, out_range=(0.0, 1.0))
  else:
    eosin = (sk_exposure.rescale_intensity(eosin, out_range=(0, 255))).astype("uint8")
  if display_info:
    util.np_info(eosin, "HED to Eosin", t.elapsed())
  return eosin

def filter_hed_to_dab(np_img, output_type="uint8", display_info=True):
  """
  Obtain diaminobenzoate (DAB) channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.

  Args:
    np_img: HED image as a NumPy array.
    output_type: Type of array to return (float or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array for Diaminobenzoate channel.
  """
  if display_info:
    t = Time()
  dab = np_img[:, :, 2]
  if output_type == "float":
    dab = sk_exposure.rescale_intensity(dab, out_range=(0.0, 1.0))
  else:
    dab = (sk_exposure.rescale_intensity(dab, out_range=(0, 255))).astype("uint8")
  if display_info:
    util.np_info(dab, "HED to DAB", t.elapsed())
  return dab

def filter_binary_fill_holes(np_img, output_type="bool", display_info=True):
  """
  Fill holes in a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) where holes have been filled.
  """
  if display_info:
    t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = scipy_img.binary_fill_holes(np_img)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_info:
    util.np_info(result, "Binary Fill Holes", t.elapsed())
  return result


def filter_binary_erosion(np_img, disk_size=5, iterations=1, output_type="uint8", display_info=True):
  """
  Erode a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for erosion.
    iterations: How many times to repeat the erosion.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) where edges have been eroded.
  """
  if display_info:
    t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = scipy_img.binary_erosion(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_info:
    util.np_info(result, "Binary Erosion", t.elapsed())
  return result


def filter_binary_dilation(np_img, disk_size=5, iterations=1, output_type="uint8", display_info=True):
  """
  Dilate a binary object (bool, float, or uint8).

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for dilation.
    iterations: How many times to repeat the dilation.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) where edges have been dilated.
  """
  if display_info:
    t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = scipy_img.binary_dilation(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_info:
    util.np_info(result, "Binary Dilation", t.elapsed())
  return result

def filter_binary_dilation_scale(np_img, disk_size_scale=1, iterations=1, output_type="uint8", display_info=True):
  """
  Wrapper for filter_binary_dilation() where disk_size is proportional to image width.

  Args:
    np_img: Binary image as a NumPy array.
    disk_size_scale: Int pixel radius of disk per 100 pixels in image width.
    iterations: How many times to repeat the dilation.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) where edges have been dilated.
  """
  avg_width = (np_img.shape[0] + np_img.shape[1]) / 2
  disk_size = int(avg_width / 100) * disk_size_scale
  return filter_binary_dilation(np_img, disk_size, iterations, output_type, display_info)

def filter_binary_opening(np_img, disk_size=3, iterations=1, output_type="uint8", display_info=True):
  """
  Open a binary object (bool, float, or uint8). Opening is an erosion followed by a dilation.
  Opening can be used to remove small objects.

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for opening.
    iterations: How many times to repeat.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) following binary opening.
  """
  if display_info:
    t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = scipy_img.binary_opening(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_info:
    util.np_info(result, "Binary Opening", t.elapsed())
  return result


def filter_binary_closing(np_img, disk_size=3, iterations=1, output_type="uint8", display_info=True):
  """
  Close a binary object (bool, float, or uint8). Closing is a dilation followed by an erosion.
  Closing can be used to remove small holes.

  Args:
    np_img: Binary image as a NumPy array.
    disk_size: Radius of the disk structuring element used for closing.
    iterations: How many times to repeat.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (bool, float, or uint8) following binary closing.
  """
  if display_info:
    t = Time()
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = scipy_img.binary_closing(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_info:
    util.np_info(result, "Binary Closing", t.elapsed())
  return result


def filter_kmeans_segmentation(np_img, compactness=10, n_segments=800, display_info=True):
  """
  Use K-means segmentation (color/space proximity) to segment RGB image where each segment is
  colored based on the average color for that segment.

  Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment.
  """
  if display_info:
    t = Time()
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  result = sk_color.label2rgb(labels, np_img, kind='avg')
  if display_info:
    util.np_info(result, "K-Means Segmentation", t.elapsed())
  return result


def filter_rag_threshold(np_img, compactness=10, n_segments=800, threshold=9, display_info=True):
  """
  Use K-means segmentation to segment RGB image, build region adjacency graph based on the segments, combine
  similar regions based on threshold value, and then output these resulting region segments.

  Args:
    np_img: Binary image as a NumPy array.
    compactness: Color proximity versus space proximity factor.
    n_segments: The number of segments.
    threshold: Threshold value for combining regions.
    display_info: Boolean flag to print NumPy array info

  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment (and similar segments have been combined).
  """
  if display_info:
    t = Time()
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  g = sk_future.graph.rag_mean_color(np_img, labels)
  labels2 = sk_future.graph.cut_threshold(labels, g, threshold)
  result = sk_color.label2rgb(labels2, np_img, kind='avg')
  if display_info:
    util.np_info(result, "RAG Threshold", t.elapsed())
  return result


def filter_threshold(np_img, threshold, output_type="bool", display_info=True):
  """
  Return mask where a pixel has a value if it exceeds the threshold value.

  Args:
    np_img: Binary image as a NumPy array.
    threshold: The threshold value to exceed.
    output_type: Type of array to return (bool, float, or uint8).
    display_info: Boolean flag to print NumPy array info
    
  Returns:
    NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
    pixel exceeds the threshold value.
  """
  if display_info:
    t = Time()
  result = (np_img > threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  if display_info:
    util.np_info(result, "Threshold", t.elapsed())
  return result

def filter_gaussian(np_img, sigma=1, output_type="uint8", display_info=True):
  """
  Return Numpy image with Gaussian blur.

  Args:
    np_img: Numpy uint8 or float64 array of a grayscale or color image.
    sigma: Float value to determine width of Gaussian. Higher values result in more blur.
    output_type: Type of array to return (uint8 or float64).
    display_info: Boolean flag to print NumPy array info

  Returns:
    Numpy array representing the Gaussian blurred image.
  """
  if display_info:
    t = Time()
  result = sk_filters.gaussian(np_img, sigma, preserve_range=True, channel_axis=None)
  
  if output_type == "uint8":
    result = np.astype(result, "uint8")

  if display_info:
    util.np_info(result, "Gaussian", t.elapsed())
  return result

def filter_gaussian_scale(np_img, sigma_scale=0.25, output_type="uint8", display_info=True):
  """
  Wrapper of filter_gaussian() where filter sigma is proportional to image shape. 

  Args:
    np_img: Numpy uint8 or float64 array of a grayscale or color image.
    sigma: Float sigma value per 100 pixels of image width.
    output_type: Type of array to return (uint8 or float64).
    display_info: Boolean flag to print NumPy array info

  Returns:
    Numpy array representing the Gaussian blurred image.
  """
  avg_width = (np_img.shape[0] + np_img.shape[1]) / 2
  sigma = int(avg_width / 100) * sigma_scale
  return filter_gaussian(np_img, sigma, output_type, display_info)

def filter_median(np_img, size=5, display_info=True):
  """
  Return Numpy image with median blur applied.

  Args:
    np_img: Numpy uint8 or float64 array of a grayscale or color image.
    size: Int width of the median filter.
    display_info: Boolean flag to print NumPy array info 
  Returns:
    Numpy array representing the median blurred image.
  """
  if display_info:
    t = Time()

  if np_img.ndim == 2:
    result = scipy_img.median_filter(np_img, size=size)
  else:
    result = np.zeros_like(np_img)
    for i in range(3):
      result[:, :, i] = scipy_img.median_filter(np_img[:, :, i], size=size)
  
  if display_info:
    util.np_info(result, "Median", t.elapsed())
  return result

def filter_median_scale(np_img, size_scale=3, display_info=True):
  """
  Wrapper for filter_median() where filter size is proportional to image shape.

  Args:
    np_img: Numpy uint8 or float64 array of a grayscale or color image.
    size_scale: Int pixel width of median filter per 100 pixels of image width.
    display_info: Boolean flag to print NumPy array info 
  Returns:
    Numpy array representing the median blurred image.
  """
  avg_width = (np_img.shape[0] + np_img.shape[1]) / 2
  size = int(avg_width / 100) * size_scale
  return filter_median(np_img, size, display_info)

def filter_pipeline(np_img, filter_pipeline, name=None, display_info=True):
  """
  Apply pipeline of filtering operations on Numpy image

  Args:
    np_img: Numpy uint8 or float64 array to filter
    filter_pipeline: Sequential list of filtering functions to apply to np_img
    name (optional): Pipeline name to print
    display_info: Boolean flag to print NumPy array info

  Returns:
    Numpy array with filters applied
  """
  if display_info:
    t = Time()
  result = reduce(lambda v, f: f(v), filter_pipeline, np_img)
  if display_info:
    util.np_info(result, name, t.elapsed())
  return result

def filter_img_dir(img_dir, filter_dir, pipeline, pipeline_name=None):
  """
  Apply pipeline of filtering operations on a directory of image files.

  Args:
    img_dir: Path to image directory.
    filter_dir: Path to directory to save filtered images.
    pipeline: Sequential list of filtering functions to apply.
    pipeline_name (optional): Pipeline name to print.
  """
  # Numpy array with structure [size (pixels), runtime (microsecs)]
  size_runtime = np.zeros((len(os.listdir(img_dir)), 2))
  for i, file in enumerate(os.listdir(img_dir)):
    t = Time()
    print(file)
    img = util.pil_to_np_rgb(Image.open(os.path.join(img_dir, file)), display_info=False)
    filtered = util.np_to_pil(filter_pipeline(img, pipeline, pipeline_name))
    filename = os.path.splitext(file)
    print(os.path.join(filter_dir, filename[0] + "_filter" + filename[1]))
    util.save_pil(filtered, os.path.join(filter_dir, filename[0] + "_filter" + filename[1]))
    size_runtime[i] = np.array([t.elapsed().microseconds, img.shape[0] * img.shape[1]])
    print("-" * 70)
  print("Average runtime: %d microsec"% np.mean(size_runtime[:, 0]))
  print("Average microsec per pixel: %.4f\n"% np.mean(size_runtime[:, 0] / size_runtime[:, 1]))

def mask_img_dir(rgb_dir, filter_dir):
  """
  Apply binary masks from filter directory to RGB image files in directory.

  Args:
  rgb_dir: Path to RGB image directory.
  filter_dir: Path to directory with binary masks.
  """
  rgb_files = os.listdir(rgb_dir)
  filter_files = os.listdir(filter_dir)
  for rgb in rgb_files:
    filename = os.path.splitext(rgb)
    if filename[0] + "_filter" + filename[1] in filter_files:
      rgb_np = util.pil_to_np_rgb(
        Image.open(
          os.path.join(
            rgb_dir, 
            rgb
            )),
        display_info=False)
      filter_np = util.pil_to_np_rgb(
        Image.open(
          os.path.join(
            filter_dir, 
            filename[0] + "_filter" + filename[1]
            )),
        display_info=False)
      masked = util.np_to_pil(util.mask_rgb(rgb_np, filter_np, display_info=False))
      util.save_pil(masked, os.path.join(filter_dir, filename[0] + "_mask" + filename[1]))
  print("Applied all masks.")