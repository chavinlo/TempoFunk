for video in /workspace/TempoFunk/data/tiktok/raw/*.webm; do
  # extract the video ID from the file path
  video_id=$(basename "$video" .webm)
  echo $video_id
  # create a directory for the frames
  mkdir -p /workspace/TempoFunk/data/tiktok/frames/$video_id/
  # extract the frames from the video
  ffmpeg -i $video /workspace/TempoFunk/data/tiktok/frames/$video_id/%06d.png
done
