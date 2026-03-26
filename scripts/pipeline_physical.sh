# Run full pipeline with fine-tuned model
python scripts/run_full_pipeline.py \
  --video data/raw_videos/football_sunday_output_000.mp4 \
  --output output/pipeline_physical \
  --stages 1,2,3 \
  --config configs/clip_000_physical.yaml
