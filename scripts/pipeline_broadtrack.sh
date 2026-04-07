# Run full pipeline with fine-tuned model
python scripts/run_full_pipeline.py \
  --video data/raw_videos/football_sunday_output_000.mp4 \
  --output output/pipeline_broadtrack \
  --stages field_registration \
  --config configs/clip_000_broadtrack.yaml
