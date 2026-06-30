# Run full pipeline with fine-tuned model
python scripts/run_full_pipeline.py \
  --video data/raw_videos/football_sunday_output_000.mp4 \
  --output output/pipeline_homograpy \
  --stages field_registration \
  --config workspace/configs/clip_000_homography.yaml
