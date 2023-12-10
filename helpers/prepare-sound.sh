#!/usr/bin/env bash

# ffmpeg command to convert to:
# - single channel audio
# - 44.1kHz sample rate
# - 16-bit sample size

ffmpeg -y -i $1 -ac 1 -ar 44100 -sample_fmt s16 $1.tmp.wav

# - 0.75s duration
# - with 0.25s fade out
# ffmpeg -y -i $1 -ac 1 -ar 44100 -sample_fmt s16 -t 0.75 -af "afade=t=out:st=0.25:d=0.5" $1.tmp.wav

rm $1
mv $1.tmp.wav $1