# Adapted from https://github.com/deroneriksson/python-wsi-preprocessing

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = False

def save_pil(pil_img, savepath):
  """
  Save a PIL image to path.

  Args:
    pil_img: The PIL image to save.
    savepath: The path to the image file.
  """
  dir = os.path.dirname(savepath)
  if dir != '' and not os.path.exists(dir):
    os.makedirs(dir)
  pil_img.save(savepath)

def pil_to_np_rgb(pil_img, display_info=True):
  """
  Convert a PIL Image to a NumPy array.

  Note that RGB PIL (w, h) -> NumPy (h, w, 3).

  Args:
    pil_img: The PIL Image.

  Returns:
    The PIL image converted to a NumPy array.
  """
  if display_info:
    t = Time()
  rgb = np.asarray(pil_img)
  if display_info:
    np_info(rgb, "RGB", t.elapsed())
  return rgb


def np_to_pil(np_img):
  """
  Convert a NumPy array to a PIL Image.

  Args:
    np_img: The image represented as a NumPy array.

  Returns:
     The NumPy array converted to a PIL Image.
  """
  if np_img.dtype == "bool":
    np_img = np_img.astype("uint8") * 255
  elif np_img.dtype == "float64":
    np_img = (np_img * 255).astype("uint8")
  return Image.fromarray(np_img)


def np_info(np_arr, name=None, elapsed=None):
  """
  Display information (shape, type, max, min, etc) about a NumPy array.

  Args:
    np_arr: The NumPy array.
    name: The (optional) name of the array.
    elapsed: The (optional) time elapsed to perform a filtering operation.
  """

  if name is None:
    name = "NumPy Array"
  if elapsed is None:
    elapsed = "---"

  if ADDITIONAL_NP_STATS is False:
    print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
  else:
    # np_arr = np.asarray(np_arr)
    max = np_arr.max()
    min = np_arr.min()
    mean = np_arr.mean()
    is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
    print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
      name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


def display_img(np_img, text=None, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False):
  """
  Convert a NumPy array to a PIL image, add text to the image, and display the image.

  Args:
    np_img: Image as a NumPy array.
    text: The text to add to the image.
    font_path: The path to the font to use.
    size: The font size
    color: The font color
    background: The background color
    border: The border color
    bg: If True, add rectangle background behind text
  """
  result = np_to_pil(np_img)
  # if gray, convert to RGB for display
  if result.mode == 'L':
    result = result.convert('RGB')
  draw = ImageDraw.Draw(result)
  if text is not None:
    font = ImageFont.truetype(font_path, size)
    if bg:
      (x, y) = draw.textsize(text, font)
      draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
    draw.text((2, 0), text, color, font=font)
  plt.imshow(result)


def mask_rgb(rgb, mask, display_info=True):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  if display_info:
    t = Time()
  if mask.dtype == "uint8":
    result = rgb & np.dstack([mask, mask, mask])
  else:
    result = rgb * np.dstack([mask, mask, mask])
  if display_info:
    np_info(result, "Mask RGB", t.elapsed())
  return result

def show_histogram(np_img, title, grayscale=False, channel_idx=0):
  """
  Show a histogram distribution of pixel values for a grayscale or color image.

  Args:
    np_img: Grayscale or color Numpy image in [height, width, channel] or [height, width] order.
    title: Title for historgram.
    grayscale: Flags if image is grayscale or color. False by default.
    channel_idx: Index of color channel. 0 by default.
  """
  if grayscale is True:
    channel = np_img
  else:
    if channel_idx >= np_img.shape[2] or channel_idx < 0:
      print("Channel index %d out of range. Setting to channel 0." % channel_idx)
      channel_idx = 0
    channel = np_img[:, :, channel_idx]

  if np_img.dtype == "uint8":
    rng = (0, 255)
  else:
    rng = (0, 1)

  histogram, bin_edges = np.histogram(channel, bins=256, range=rng)
  plt.plot(bin_edges[0:-1], histogram)
  plt.title(title)
  plt.show()

def img_dir_stats(img_dir):
  files = os.listdir(img_dir)
  dims = np.zeros((len(files), 2))
  size = np.zeros(len(files))

  for i, file in enumerate(files):
    img = Image.open(os.path.join(img_dir, file))
    dims[i] = np.array(img.size)
    size[i] = os.path.getsize(os.path.join(img_dir, file))

  print(img_dir)
  print("Avg dimensions: %d x %d pixels" % (np.mean(dims, 0)[0], np.mean(dims, 0)[1]))
  print("Avg size: %.2f MB" % (np.mean(size) / 1000000))
  
class Time:
  """
  Class for displaying elapsed time.
  """

  def __init__(self):
    self.start = datetime.datetime.now()

  def elapsed_display(self):
    time_elapsed = self.elapsed()
    print("Time elapsed: " + str(time_elapsed))

  def elapsed(self):
    self.end = datetime.datetime.now()
    time_elapsed = self.end - self.start
    return time_elapsed
