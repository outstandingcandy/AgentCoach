# Run full pipeline with fine-tuned model
  # --stages field_registration,tracking \
python scripts/run_full_pipeline.py \
  --video data/raw_videos/football_sunday_full.mp4 \
  --output output/full_pipeline/full \
  --stages event_detection \
  --no-timestamp \
  --config configs/clip_000_physical.yaml
