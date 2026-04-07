# Run full pipeline with fine-tuned model
python scripts/run_full_pipeline.py \
  --video data/raw_videos/football_sunday_output_000.mp4 \
  --output output/full_pipeline \
  --stages field_registration,tracking \
  --config configs/clip_000_finetuned.yaml
