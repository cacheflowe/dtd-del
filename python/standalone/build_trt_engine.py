"""
Offline ONNX -> TensorRT engine converter, run OUTSIDE TouchDesigner via
.venv\\Scripts\\python.exe -- mirrors TDDepthAnything's accelerate.bat pattern
(https://github.com/olegchomp/TDDepthAnything): conversion happens once, in a
completely separate process, so the (comparatively heavy, ONNX-parsing,
kernel-autotuning) TensorRT builder never has to run inside TD's own process.

The resulting .engine file is loaded at runtime by python/util/trt_inference_manager.py
using ONLY the raw `tensorrt` Python package (deserialize + execute) -- no onnxruntime
involved at all, which is the whole point: see docs/learnings/onnx-runtime.md and
python/util/onnx_inference_manager.py's providers() docstring for why onnxruntime's own
TensorRT execution provider crashes TD's process outright.

Usage:
    .venv\\Scripts\\python.exe python\\standalone\\build_trt_engine.py --model yolo26s-pose

TensorRT engines are tied to the exact GPU/driver/TensorRT version that built them --
rebuild whenever any of those change, or the model itself changes.
"""

import argparse
import os
import time

import numpy as np
import tensorrt as trt

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# Matches python/standalone/webcam_yolo_pose.py's MODEL_INPUT_SIZE -- yolo26*-pose.onnx's
# own fixed 640x640 input, no dynamic-shape optimization profile needed.
MODEL_INPUT_SIZE = 640
WORKSPACE_BYTES = 1 << 30  # 1GB -- generous default; shrink if VRAM is tight elsewhere


def get_onnx_path(variant):
	return os.path.join(_PROJECT_ROOT, 'data', 'ml', 'yolo26', f'{variant}.onnx')


def get_engine_path(variant, fp16):
	suffix = 'fp16' if fp16 else 'fp32'
	return os.path.join(_PROJECT_ROOT, 'data', 'ml', 'yolo26', f'{variant}.{suffix}.engine')


def build_engine(onnx_path, engine_path, fp16=True):
	builder = trt.Builder(TRT_LOGGER)
	network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
	network = builder.create_network(network_flags)
	parser = trt.OnnxParser(network, TRT_LOGGER)

	with open(onnx_path, 'rb') as f:
		if not parser.parse(f.read()):
			for i in range(parser.num_errors):
				print(f"[TRT] Parser error: {parser.get_error(i)}")
			raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

	# Check for dynamic shapes and add optimization profile if needed
	input_tensor = network.get_input(0)
	input_shape = input_tensor.shape
	print(f"[TRT] Input '{input_tensor.name}' shape: {input_shape}")
	
	has_dynamic = any(dim == -1 for dim in input_shape)
	if has_dynamic:
		print("[TRT] Dynamic batch dimension detected -- adding optimization profile with batch=1")
		profile = builder.create_optimization_profile()
		# Replace -1 (dynamic batch) with 1 for min/opt/max
		min_shape = tuple(1 if dim == -1 else dim for dim in input_shape)
		opt_shape = min_shape
		max_shape = min_shape
		profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
		config = builder.create_builder_config()
		config.add_optimization_profile(profile)
	else:
		config = builder.create_builder_config()

	config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
	if fp16:
		if not builder.platform_has_fast_fp16:
			print("[TRT] WARNING: platform reports no fast-fp16 support -- building fp16 anyway, may be slow.")
		config.set_flag(trt.BuilderFlag.FP16)

	print(f"[TRT] Building engine from {onnx_path} (fp16={fp16}) -- this can take a "
		"few minutes, same one-time cost as the ORT-cached engine builds we saw earlier.")
	t0 = time.perf_counter()
	serialized_engine = builder.build_serialized_network(network, config)
	if serialized_engine is None:
		raise RuntimeError("Engine build failed (builder returned None) -- check parser/logger output above.")
	print(f"[TRT] Build finished in {time.perf_counter() - t0:.1f}s")

	os.makedirs(os.path.dirname(engine_path), exist_ok=True)
	with open(engine_path, 'wb') as f:
		f.write(serialized_engine)
	print(f"[TRT] Saved engine: {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--model', default='yolo26s-pose', 
		choices=['yolo26n-pose', 'yolo26s-pose', 'yolo26s-seg'])
	parser.add_argument('--fp32', action='store_true', help='Build fp32 instead of the default fp16.')
	args = parser.parse_args()

	onnx_path = get_onnx_path(args.model)
	if not os.path.isfile(onnx_path):
		print(f"[TRT] ONNX model not found: {onnx_path}")
		raise SystemExit(1)

	engine_path = get_engine_path(args.model, fp16=not args.fp32)
	build_engine(onnx_path, engine_path, fp16=not args.fp32)


if __name__ == '__main__':
	main()
