#!/bin/bash

wsi_dir="data"
thumbnail_dir="thumbnails"
scale=128
filter_pipeline="gauss_otsu"
pipeline_name="Gaussian Otsu pipeline"

# segment background of WSI thumbnails
python bg_seg_script.py $wsi_dir $thumbnail_dir $scale $filter_pipeline "$pipeline_name"