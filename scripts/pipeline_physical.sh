# Run full pipeline with fine-tuned model
python scripts/run_full_pipeline.py \
  --video data/raw_videos/football_sunday_output_007.mp4 \
  --output output/pipeline_physical_stage2_test_3 \
  --stages 2 \
  --config configs/clip_000_physical.yaml \
  --no-timestamp
