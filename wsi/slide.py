# Adapted from https://github.com/deroneriksson/python-wsi-preprocessing

import openslide
from openslide import OpenSlideError
import math
import os

import wsi.util as util
from wsi.util import Time

def open_slide(filepath):
  """
  Open a whole-slide image (*.svs, etc).

  Args:
    filepath: Path to the slide file.

  Returns:
    An OpenSlide object representing a whole-slide image.
  """
  try:
    slide = openslide.open_slide(filepath)
  except OpenSlideError:
    slide = None
  except FileNotFoundError:
    slide = None
  return slide

def slide_to_scaled_pil_image(filepath, scale):
  """
  Convert a WSI training slide to a scaled-down PIL image.

  Args:
    filepath: Path to the slide file.
    scale: Integer factor to scale down slide.

  Returns:
    Scaled-down PIL image.
  """
  svs = open_slide(filepath)
  w, h = svs.dimensions
  w = math.floor(w / scale)
  h = math.floor(h / scale)
  return svs.get_thumbnail((w,h))

def slide_to_scaled_np_image(filepath, scale):
  """
  Convert a WSI training slide to a scaled-down NumPy image.

  Args:
    filepath: Path to the slide file.
    scale: Integer factor to scale down slide.

  Returns:
    Scaled-down NumPy image.
  """
  pil_img = slide_to_scaled_pil_image(filepath, scale)
  np_img = util.pil_to_np_rgb(pil_img)
  return np_img


def show_slide(filepath):
  """
  Display a WSI slide on the screen, where the slide has been scaled down and converted to a PIL image.

  Args:
    filepath: Path to the slide file.
  """
  pil_img = slide_to_scaled_pil_image(filepath)
  pil_img.show()

def save_pil(pil_img, savepath):
  """
  Save a thumbnail of a PIL image.

  Args:
    pil_img: The PIL image to save as a thumbnail.
    savepath: The path to the thumbnail.
  """
  dir = os.path.dirname(savepath)
  if dir != '' and not os.path.exists(dir):
    os.makedirs(dir)
  pil_img.save(savepath)

def save_slide_thumbnail(filepath, savepath, scale):
  """
  Save a thumbnail of a WSI file downsized by scale factor.

  Args:
    filepath: Path to WSI file.
    savepath: Path to thumbnail image file.
    scale: Factor to downscale WSI.
  """
  pil = slide_to_scaled_pil_image(filepath, scale)
  save_pil(pil, savepath)

def save_slide_dir_thumbnails(slide_dir, thumbnail_dir, scale=64, format="png"):
  """
  Save thumbnails of all WSI files in a directory to another specified directory.

  Args:
    slide_dir: Name of directory with WSI files.
    thumbnail_dir: Name of directory to save thumbnails.
    format: File format of thumbnails.
  """
  if not os.path.exists(slide_dir):
    return FileNotFoundError
  if not os.path.exists(thumbnail_dir):
    os.makedirs(thumbnail_dir)
  for slide_path in os.listdir(slide_dir):
    thumbnail_path = os.path.join(thumbnail_dir, os.path.splitext(slide_path)[0] + "." + format)
    save_slide_thumbnail(os.path.join(slide_dir, slide_path), thumbnail_path, scale)

def slide_info(filepath, display_all_properties=False):
  """
  Display information (such as properties) about training images.

  Args:
    filepath: Path to the slide file.
    display_all_properties: If True, display all available slide properties.
  """
  t = Time()

  print("\nOpening Slide: %s" % filepath)
  slide = open_slide(filepath)
  print("Level count: %d" % slide.level_count)
  print("Level dimensions: " + str(slide.level_dimensions))
  print("Level downsamples: " + str(slide.level_downsamples))
  print("Dimensions: " + str(slide.dimensions))
  objective_power = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
  print("Objective power: " + str(objective_power))
  print("Associated images:")
  for ai_key in slide.associated_images.keys():
    print("  " + str(ai_key) + ": " + str(slide.associated_images.get(ai_key)))
  print("Format: " + str(slide.detect_format(filepath)))
  if display_all_properties:
    print("Properties:")
    for prop_key in slide.properties.keys():
      print("  Property: " + str(prop_key) + ", value: " + str(slide.properties.get(prop_key)))

  t.elapsed_display()