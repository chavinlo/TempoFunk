#!/bin/bash

# Make a folder for the raw videos
mkdir raw

# Read each line of the list.txt file
while read line; do
  # Get the video ID by splitting the URL on '/' and taking the last element
  video_id=$(echo $line | awk -F/ '{print $NF}')
  echo "Downloading video with ID $video_id"

  # Download the video to the raw folder
  yt-dlp $line -o raw/mp4/$video_id.mp4

  # Extract the metadata and save it to a JSON file in the raw folder
  yt-dlp --dump-json $line > raw/mp4/$video_id.json
done < list.txt
