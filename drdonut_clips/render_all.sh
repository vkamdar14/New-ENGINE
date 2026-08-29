#!/usr/bin/env bash
# Renders every mined DrDonut clip. Needs ffmpeg and the source VODs.
# Download each source to <VIDEO_ID>.mp4 in this directory first.
set -euo pipefail

# clip01_oafS9P4zFZo_265  @4:25  10 people, 26.0x  [BACK TO EDIT]
ffmpeg -y -ss 265.00 -i oafS9P4zFZo.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip01_oafS9P4zFZo_265.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip01_oafS9P4zFZo_265.mp4

# clip02_oafS9P4zFZo_0  @0:00  8 people, 16.2x  [BACK TO EDIT]
ffmpeg -y -ss 0.00 -i oafS9P4zFZo.mp4 -t 8.00 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip02_oafS9P4zFZo_0.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip02_oafS9P4zFZo_0.mp4

# clip03_oafS9P4zFZo_622  @10:22  7 people, 17.8x  [BACK TO EDIT]
ffmpeg -y -ss 622.00 -i oafS9P4zFZo.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip03_oafS9P4zFZo_622.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip03_oafS9P4zFZo_622.mp4

# clip04_oafS9P4zFZo_657  @10:57  5 people, 18.5x  [BACK TO EDIT]
ffmpeg -y -ss 657.00 -i oafS9P4zFZo.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip04_oafS9P4zFZo_657.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip04_oafS9P4zFZo_657.mp4

# clip05_oafS9P4zFZo_1076  @17:56  4 people, 4.3x  [BACK TO EDIT]
ffmpeg -y -ss 1076.00 -i oafS9P4zFZo.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip05_oafS9P4zFZo_1076.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip05_oafS9P4zFZo_1076.mp4

# clip06_9HM5HJRqqtQ_9345  @155:45  10 people, 78.9x  [BACK TO EDIT]
ffmpeg -y -ss 9345.00 -i 9HM5HJRqqtQ.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip06_9HM5HJRqqtQ_9345.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip06_9HM5HJRqqtQ_9345.mp4

# clip07_9HM5HJRqqtQ_10128  @168:48  9 people, 68.6x  [BACK TO EDIT]
ffmpeg -y -ss 10128.00 -i 9HM5HJRqqtQ.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip07_9HM5HJRqqtQ_10128.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip07_9HM5HJRqqtQ_10128.mp4

# clip08_9HM5HJRqqtQ_6620  @110:20  4 people, 46.3x  [BACK TO EDIT]
ffmpeg -y -ss 6620.00 -i 9HM5HJRqqtQ.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip08_9HM5HJRqqtQ_6620.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip08_9HM5HJRqqtQ_6620.mp4

# clip09_9HM5HJRqqtQ_5360  @89:20  4 people, 39.1x  [BACK TO EDIT]
ffmpeg -y -ss 5360.00 -i 9HM5HJRqqtQ.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip09_9HM5HJRqqtQ_5360.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip09_9HM5HJRqqtQ_5360.mp4

# clip10_9HM5HJRqqtQ_7132  @118:52  5 people, 16.5x  [BACK TO EDIT]
ffmpeg -y -ss 7132.00 -i 9HM5HJRqqtQ.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip10_9HM5HJRqqtQ_7132.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip10_9HM5HJRqqtQ_7132.mp4

# clip11_9HM5HJRqqtQ_11012  @183:32  4 people, 34.0x  [BACK TO EDIT]
ffmpeg -y -ss 11012.00 -i 9HM5HJRqqtQ.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip11_9HM5HJRqqtQ_11012.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip11_9HM5HJRqqtQ_11012.mp4

# clip12_9HM5HJRqqtQ_8725  @145:25  4 people, 33.0x  [BACK TO EDIT]
ffmpeg -y -ss 8725.00 -i 9HM5HJRqqtQ.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip12_9HM5HJRqqtQ_8725.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip12_9HM5HJRqqtQ_8725.mp4

# clip13_9HM5HJRqqtQ_6098  @101:38  4 people, 26.9x  [BACK TO EDIT]
ffmpeg -y -ss 6098.00 -i 9HM5HJRqqtQ.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip13_9HM5HJRqqtQ_6098.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip13_9HM5HJRqqtQ_6098.mp4

# clip14_bULfieZael0_14872  @247:52  9 people, 218.2x  [BACK TO EDIT]
ffmpeg -y -ss 14872.00 -i bULfieZael0.mp4 -t 16.30 -vf 'scale=1080:-2:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-out_w)/2:(in_h-out_h)/2,subtitles='"'"'clip14_bULfieZael0_14872.ass'"'"'' -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart clip14_bULfieZael0_14872.mp4
