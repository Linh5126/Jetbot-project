#!/bin/sh
# Build the TensorRT FP16 engines ON THE NANO (an engine only runs on the TensorRT/GPU that built it).
# Inputs come from the PC: scripts/export_onboard_models.py -> models/onboard/*.onnx
set -e
cd "$(dirname "$0")/../models/onboard"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
$TRTEXEC --onnx=yolov8n_pose_288x480.onnx --saveEngine=yolov8n_pose_288x480_fp16.engine --fp16 --workspace=1024
$TRTEXEC --onnx=osnet_x0_5.onnx --saveEngine=osnet_x0_5_fp16.engine --fp16 --workspace=512 \
  --minShapes=input:1x3x256x128 --optShapes=input:4x3x256x128 --maxShapes=input:8x3x256x128
if [ -f collision_resnet18.onnx ]; then  # optional: scripts/train_collision.py
  $TRTEXEC --onnx=collision_resnet18.onnx --saveEngine=collision_resnet18_fp16.engine --fp16 --workspace=512
fi
ls -l *.engine
