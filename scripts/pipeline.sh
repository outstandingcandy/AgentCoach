# Run full pipeline with fine-tuned model
  # --stages field_registration,tracking,event_detection,highlights \
python scripts/run_full_pipeline.py \
  --video data/raw_videos/football_sunday_full.mp4 \
  --output output/full_pipeline/full_v2 \
  --stages tracking,event_detection,highlights \
  --no-timestamp \
  --no-viz \
  --config configs/clip_000_physical.yaml
