#!/usr/bin/env bash
set -e
cd /home/ubuntu/AgentCoach
source .venv/bin/activate
export GOALINSIGHT_VIDEO_S3_BUCKET=goalinsight-videos-683638520402-us-east-1
export GOALINSIGHT_VIDEO_CLOUDFRONT_DOMAIN=d2bjm7xbpppopk.cloudfront.net
export GOALINSIGHT_VIDEO_CF_KEY_PAIR_ID=K2YIV8PFBOU4I2
export GOALINSIGHT_VIDEO_CF_PRIVATE_KEY_PATH=/home/ubuntu/AgentCoach/deploy/cloudfront/cf_private.pem
exec python -m goalinsight.web --workspace ./workspace --host 0.0.0.0 --port 8000
