#!/bin/sh

set -e

python gemma4-sft/train.py

sleep 30

python gemma4-rl/train.py