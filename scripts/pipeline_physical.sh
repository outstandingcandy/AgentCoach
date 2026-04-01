# Run full pipeline with fine-tuned model
python scripts/run_full_pipeline.py \
  --video data/raw_videos/football_sunday_output_009.mp4 \
  --output output/pipeline_009_1 \
  --stages 1,2 \
  --config configs/clip_000_physical.yaml \
  --no-timestamp
