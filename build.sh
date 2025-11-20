#!/usr/bin/env bash
# exit on error
set -o errexit

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Download the NLTK 'punkt' model
python -m nltk.downloader punkt