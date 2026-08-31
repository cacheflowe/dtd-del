"""
Minimal, SYNCHRONOUS proof-of-concept test: does loading and running a raw TensorRT
engine inside TD's own process crash it?

Deliberately does NOT use TRTInferenceManager's threaded loading path yet -- everything
here runs on whichever thread calls run_poc() (the main/cook thread if called from a
Script DAT, or td_http_api's /run thread if called that way). No background thread, no
Par/OP access from a thread, no tracking/postprocess logic -- the goal is only to answer
one question in isolation before building anything real on top of it.

Usage from TD (Textport or a DAT's onStart, or via td_http_api's /run):
    import trt_poc_test
    trt_poc_test.run_poc()

Expected on success: engine loads, one dummy inference runs, output shape (1, 300, 57)
prints (matches yolo26s-pose.onnx's end2end export -- see onnx_yolo26_pose.py's module
docstring), no crash.
"""

import sys
import os
import time

# Add .venv site-packages so TD can find tensorrt, cuda-python, pywin32
_PROJECT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_VENV_SITE = os.path.join(_PROJECT_FOLDER, '.venv', 'Lib', 'site-packages')
_VENV_ROOT = os.path.join(_PROJECT_FOLDER, '.venv')
if _VENV_SITE not in sys.path:
	sys.path.insert(0, _VENV_SITE)
	print(f"[TRT-POC-INIT] Added {_VENV_SITE} to sys.path")
else:
	print(f"[TRT-POC-INIT] {_VENV_SITE} already in sys.path")
# Add win32 subdirectories (pywin32.pth does this for venv python, we do it manually for TD)
for subdir in ['win32', 'win32/lib', 'pythonwin']:
	p = os.path.join(_VENV_SITE, subdir)
	if p not in sys.path:
		sys.path.insert(0, p)
print(f"[TRT-POC-INIT] Added win32 subdirectories to sys.path")
# Add .venv root to DLL search path for pywin32 DLLs
os.add_dll_directory(_VENV_ROOT)
print(f"[TRT-POC-INIT] Added {_VENV_ROOT} to DLL search path")
# pywin32.pth runs this import to handle environments where post_install wasn't run
import pywin32_bootstrap  # noqa: E402
print(f"[TRT-POC-INIT] pywin32_bootstrap imported")

import numpy as np


def _cuda_check(err):
	"""cuda-python's cudart functions return (cudaError_t, ...) tuples -- raise on
	anything but cudaSuccess instead of silently continuing with a bad buffer/stream."""
	from cuda import cudart
	if err != cudart.cudaError_t.cudaSuccess:
		raise RuntimeError(f"CUDA error: {err} ({cudart.cudaGetErrorString(err)[1]})")


MODEL_INPUT_SIZE = 640


def run_poc(engine_path=None):
	if engine_path is None:
		project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
		engine_path = os.path.join(project_folder, 'data', 'ml', 'yolo26', 'yolo26s-pose.fp16.engine')

	print(f"[TRT-POC] engine path: {engine_path}")
	if not os.path.isfile(engine_path):
		print(f"[TRT-POC] FAILED: engine file not found. Build it first with "
			f"python/standalone/build_trt_engine.py")
		return False

	print("[TRT-POC] Step 1: importing tensorrt + cuda-python...")
	import tensorrt as trt
	from cuda import cudart
	print(f"[TRT-POC]   tensorrt version: {trt.__version__}")

	print("[TRT-POC] Step 2: deserializing engine...")
	logger = trt.Logger(trt.Logger.WARNING)
	runtime = trt.Runtime(logger)
	with open(engine_path, 'rb') as f:
		engine = runtime.deserialize_cuda_engine(f.read())
	if engine is None:
		print("[TRT-POC] FAILED: deserialize_cuda_engine() returned None")
		return False
	context = engine.create_execution_context()
	print("[TRT-POC]   OK -- engine + execution context created")

	input_name = engine.get_tensor_name(0)
	output_name = engine.get_tensor_name(1)
	input_shape = tuple(engine.get_tensor_shape(input_name))
	output_shape = tuple(engine.get_tensor_shape(output_name))
	print(f"[TRT-POC]   input='{input_name}' shape={input_shape}")
	print(f"[TRT-POC]   output='{output_name}' shape={output_shape}")

	print("[TRT-POC] Step 3: allocating device buffers...")
	input_nbytes = int(np.prod(input_shape)) * np.dtype(np.float32).itemsize
	output_nbytes = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize
	err, d_input = cudart.cudaMalloc(input_nbytes)
	_cuda_check(err)
	err, d_output = cudart.cudaMalloc(output_nbytes)
	_cuda_check(err)
	err, stream = cudart.cudaStreamCreate()
	_cuda_check(err)
	context.set_tensor_address(input_name, d_input)
	context.set_tensor_address(output_name, d_output)
	print("[TRT-POC]   OK -- device buffers allocated")

	print("[TRT-POC] Step 4: running one dummy inference (all-zero input)...")
	dummy_input = np.zeros(input_shape, dtype=np.float32)
	h_output = np.empty(output_shape, dtype=np.float32)

	t0 = time.perf_counter()
	err = cudart.cudaMemcpyAsync(d_input, dummy_input.ctypes.data, dummy_input.nbytes,
		cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)[0]
	_cuda_check(err)
	ok = context.execute_async_v3(stream)
	if not ok:
		print("[TRT-POC] FAILED: execute_async_v3() returned False")
		return False
	err = cudart.cudaMemcpyAsync(h_output.ctypes.data, d_output, h_output.nbytes,
		cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)[0]
	_cuda_check(err)
	err = cudart.cudaStreamSynchronize(stream)[0]
	_cuda_check(err)
	elapsed_ms = (time.perf_counter() - t0) * 1000

	print(f"[TRT-POC]   OK -- inference ran in {elapsed_ms:.2f}ms, output shape {h_output.shape}, "
		f"sample values: {h_output.flat[:5]}")
	print("[TRT-POC] SUCCESS -- raw TensorRT loaded and ran inside TD without crashing.")

	cudart.cudaFree(d_input)
	cudart.cudaFree(d_output)
	cudart.cudaStreamDestroy(stream)
	return True
