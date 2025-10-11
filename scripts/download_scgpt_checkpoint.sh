#!/bin/bash

# Script to download scGPT checkpoint from Google Drive
# The model is available at: https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y

echo "=== scGPT Checkpoint Download Instructions ==="
echo ""
echo "Please download the scGPT whole-human model manually from:"
echo "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
echo ""
echo "Files to download:"
echo "1. best_model_ckpt.pt (the main checkpoint file)"
echo ""
echo "Save the file to: models/scgpt/best_model_ckpt.pt"
echo ""
echo "Alternatively, you can use gdown if you have it installed:"
echo "pip install gdown"
echo "gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y -O models/scgpt/"
echo ""
echo "Expected file size: ~2-5GB"
echo ""
echo "After downloading, verify with:"
echo "ls -lh models/scgpt/best_model_ckpt.pt"