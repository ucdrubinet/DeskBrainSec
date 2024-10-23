#!/usr/bin/env python3.10
import argparse
from wsi import filter, slide
from functools import partial
import sys

OTSU = "otsu"
GAUSSIAN_OTSU = "gauss_ostu"
MEDIAN_SATURATION_OTSU_DILATION = "med_sat_ot_di"
ENTROPY = "entropy"
ENTROPY_FILL = "entr_fill"

otsu_pipeline = [
  partial(filter.filter_rgb_to_grayscale, display_info=False),
  partial(filter.filter_complement, display_info=False),
  partial(filter.filter_otsu_threshold, display_info=False)
]

gauss_otsu_pipeline = [
  partial(filter.filter_rgb_to_grayscale, display_info=False),
  partial(filter.filter_complement, display_info=False),
  partial(filter.filter_gaussian_scale, sigma_scale=1, display_info=False),
  partial(filter.filter_otsu_threshold, display_info=False)
]

msod_pipeline = [
    partial(filter.filter_median_scale, size_scale=3, display_info=False),
    partial(filter.filter_rgb_to_hsv, display_info=False),
    (lambda hsv_img: hsv_img[:, : , 1]),
    partial(filter.filter_otsu_threshold, display_info=False),
    partial(filter.filter_binary_dilation_scale, disk_size_scale=1, display_info=False),
]

entr_pipeline = [
  partial(filter.filter_rgb_to_grayscale, display_info=False),
  partial(filter.filter_entropy_scale, neighborhood_scale=3, threshold=3, display_info=False)
]

entr_fill_pipeline = [
  partial(filter.filter_rgb_to_grayscale, display_info=False),
  partial(filter.filter_entropy_scale, neighborhood_scale=3, threshold=3, display_info=False),
  partial(filter.filter_binary_fill_holes, display_info=False)
]

filter_pipelines = {
  OTSU : otsu_pipeline,
  GAUSSIAN_OTSU : gauss_otsu_pipeline,
  MEDIAN_SATURATION_OTSU_DILATION : msod_pipeline,
  ENTROPY : entr_pipeline,
  ENTROPY_FILL : entr_fill_pipeline
}

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("wsi_dir", help="name of directory with SVS files", type=str)
  parser.add_argument("thumbnail_dir", help="name of directory to save WSI thumbnails", type=str)
  parser.add_argument("scale", help="Factor to downsize WSI to thumbnail", type=int)
  parser.add_argument("filter_pipeline", help="Name of filter pipeline")
  parser.add_argument("pipeline_name", help="Print-friendly pipeline name", type=str)

  args = parser.parse_args()
  if args.filter_pipeline not in filter_pipelines.keys():
    print(f"{args.filter_pipeline} must be one of {list(filter_pipelines)}")
    return(1)
  
  slide.save_slide_dir_thumbnails(
    args.wsi_dir, 
    args.thumbnail_dir, 
    args.scale
  )
  filter.filter_img_dir(
    args.thumbnail_dir,
    args.filter_pipeline,
    filter_pipelines[args.filter_pipeline],
    args.pipeline_name
  )
  filter.mask_img_dir(
    args.thumbnail_dir,
    args.filter_pipeline
  )
  return 0

if __name__ == "__main__":
  sys.exit(main())
  
