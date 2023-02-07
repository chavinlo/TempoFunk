#!/bin/bash

# Create the frames directory if it doesn't exist
mkdir -p frames

# Loop through all the webm files in the current directory
for webm_file in /workspace/data/raw/webm/*.webm; do
  # Get the base name of the file without the extension
  base_name=$(basename "$webm_file" .webm)
  # Use FFmpeg to extract the frames from the webm file
  ffmpeg -i "$webm_file" "/workspace/data/frames/$base_name"_%06d.png
done
