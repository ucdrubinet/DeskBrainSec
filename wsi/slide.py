# Adapted from https://github.com/deroneriksson/python-wsi-preprocessing

import openslide
from openslide import OpenSlideError
from PIL import Image
import math

import util
from util import Time

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
  level = svs.get_best_level_for_downsample(scale)
  wsi = svs.read_region((0,0), level, svs.level_dimensions[level])
  return wsi.convert("RGB").resize((w,h), Image.BILINEAR)

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