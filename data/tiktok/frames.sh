#!/bin/bash

# loop through all raw video files
for video in /workspace/TempoFunk/data/tiktok/raw/*/video.webm; do
  # extract the video ID from the file path
  video_id=$(echo $video | awk -F '/' '{print $7}')
  echo $video_id
  # create a directory for the frames
  mkdir -p /workspace/TempoFunk/data/tiktok/frames/$video_id/
  # extract the frames from the video
  ffmpeg -i $video /workspace/TempoFunk/data/tiktok/frames/$video_id/%06d.png
done
