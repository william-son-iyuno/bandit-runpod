#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 INPUT_VIDEO WORK_NAME OUTPUT_VIDEO" >&2
  exit 2
fi

input_video=$1
work_name=$2
output_video=$3
experiment_root=/Users/william.son/Desktop/iyuno/test_data/experiments/bandit_v2_speech_remux
chunk_seconds=300
work_dir="$experiment_root/$work_name"
input_chunks="$work_dir/input_chunks"
speech_chunks="$work_dir/speech_chunks"
api_results="$work_dir/api_results"
mkdir -p "$input_chunks" "$speech_chunks" "$api_results"

first_chunk_duration=""
if [[ -s "$input_chunks/chunk_000.flac" ]]; then
  first_chunk_duration=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "$input_chunks/chunk_000.flac" || true)
fi

if [[ -z "$first_chunk_duration" || "$first_chunk_duration" == "N/A" ]] \
  || awk -v duration="$first_chunk_duration" -v limit="$chunk_seconds" \
    'BEGIN { exit !(duration > limit + 1) }'; then
  total_duration=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$input_video")
  chunk_count=$(awk -v duration="$total_duration" \
    -v chunk="$chunk_seconds" \
    'BEGIN { print int((duration + chunk - 0.001) / chunk) }')
  for ((index = 0; index < chunk_count; index++)); do
    start=$((index * chunk_seconds))
    printf -v chunk_path '%s/chunk_%03d.flac' "$input_chunks" "$index"
    ffmpeg -hide_banner -y -i "$input_video" -ss "$start" -t "$chunk_seconds" \
      -map 0:a:0 -vn -sample_fmt s16 -c:a flac "$chunk_path"
  done
fi

ssh_info=$(runpodctl ssh info 2kxlgz9yd1ms6c)
ssh_ip=$(jq -r .ip <<<"$ssh_info")
ssh_port=$(jq -r .port <<<"$ssh_info")
ssh_key=$HOME/.runpod/ssh/RunPod-Key-Go
tunnel_socket="/tmp/bandit-api-${work_name}.sock"
local_port=18000

ssh-keygen -R "[$ssh_ip]:$ssh_port" >/dev/null 2>&1 || true
ssh -o StrictHostKeyChecking=accept-new \
  -i "$ssh_key" -p "$ssh_port" \
  -M -S "$tunnel_socket" -fN \
  -L "$local_port":127.0.0.1:8000 root@"$ssh_ip"

cleanup_tunnel() {
  ssh -S "$tunnel_socket" -O exit root@"$ssh_ip" >/dev/null 2>&1 || true
}
trap cleanup_tunnel EXIT

api_token=$(ssh -i "$ssh_key" -p "$ssh_port" \
  root@"$ssh_ip" 'cat /workspace/.bandit-api-key')

chunk_count=$(find "$input_chunks" -type f -name 'chunk_*.flac' | wc -l | tr -d ' ')
chunk_index=0
for chunk in "$input_chunks"/chunk_*.flac; do
  chunk_index=$((chunk_index + 1))
  chunk_name=$(basename "$chunk" .flac)
  result_zip="$api_results/$chunk_name.zip"
  speech_file="$speech_chunks/${chunk_name}_speech.flac"
  if [[ -s "$speech_file" ]]; then
    echo "[$chunk_index/$chunk_count] already complete: $chunk_name"
    continue
  fi

  echo "[$chunk_index/$chunk_count] separating: $chunk_name"
  curl --fail --show-error --progress-bar --max-time 1800 \
    -X POST "http://127.0.0.1:$local_port/separate" \
    -H "X-API-Key: $api_token" \
    -F "file=@$chunk" \
    -o "$result_zip"
  unzip -t "$result_zip"
  unzip -p "$result_zip" speech.flac > "$speech_file"
  unzip -p "$result_zip" metadata.json > "$api_results/${chunk_name}_metadata.json"
done

concat_manifest="$work_dir/speech_concat.txt"
find "$speech_chunks" -type f -name 'chunk_*_speech.flac' -print \
  | sort \
  | awk '{gsub("'\''", "'\''\\'\'''\''"); print "file '\''" $0 "'\''"}' \
  > "$concat_manifest"

combined_speech="$work_dir/combined_speech.flac"
ffmpeg -hide_banner -y -f concat -safe 0 -i "$concat_manifest" \
  -af 'asetpts=N/SR/TB' -sample_fmt s16 -c:a flac "$combined_speech"

ffmpeg -hide_banner -y -i "$input_video" -i "$combined_speech" \
  -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a aac -b:a 192k -ar 48000 \
  -shortest -movflags +faststart "$output_video"

ffprobe -v error \
  -show_entries format=filename,duration,size:stream=index,codec_type,codec_name,channels,sample_rate,width,height \
  -of json "$output_video"
ffmpeg -v error -i "$output_video" -map 0:v:0 -map 0:a:0 -f null -
