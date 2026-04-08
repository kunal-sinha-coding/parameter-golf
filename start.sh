#!/usr/bin/env bash

set -e  # exit on error

# Clone repo
git clone https://github.com/kunal-sinha-coding/parameter-golf.git

# Enter directory
cd parameter-golf

# Set global git config
git config --global user.email "kunalsinha@live.com"
git config --global user.name "Kunal Sinha"

# Run the data script
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
