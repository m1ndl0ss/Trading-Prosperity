#!/bin/bash
# Run the bot hunting analysis for IMC Prosperity Round 1
cd "$(dirname "$0")"
echo "Installing dependencies..."
pip install -r requirements.txt -q
echo "Running bot hunter analysis..."
python analyze_bot.py
