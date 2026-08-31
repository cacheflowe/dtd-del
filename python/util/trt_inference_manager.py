"""
TRTInferenceManager - PROOF OF CONCEPT base class for TouchDesigner model inference
using raw TensorRT, NOT onnxruntime.

Parallel to python/util/onnx_inference_manager.py's ONNXInferenceManager, same
onCook/threading shape (only the GPU-bound call runs on the background worker thread,
pre/postprocess stay on the main thread -- see td-threaded-inference-optimization.md),
but every onnxruntime concept is replaced with its TensorRT equivalent:
  InferenceSession        -> deserialized ICudaEngine + IExecutionContext
  providers()/EP selection -> N/A, TensorRT IS the only backend here
  session.run()            -> explicit H2D copy -> execute -> D2H copy (TensorRT has no
                              onnxruntime-IOBinding-style implicit copy helper)

WHY THIS CLASS EXISTS AT ALL: onnxruntime's own TensorRT execution provider crashes TD's
process outright -- confirmed live, the crash happens inside `ort.get_available_providers()`
itself, before any session is even created (see docs/learnings/onnx-runtime.md and
onnx_inference_manager.py's providers() docstring for the full investigation). Working
theory: TD bundles its own ancient (1.10.0, confirmed via ctypes probe) onnxruntime.dll,
and onnxruntime's provider-bridge DLL loading collides with it at the process level.

Raw TensorRT (this file) never imports onnxruntime at all, so it should never touch that
collision -- inspired directly by github.com/olegchomp/TDDepthAnything, which uses this
exact pattern (offline engine build via a separate Python process -- see
python/standalone/build_trt_engine.py -- then raw TensorRT Python bindings at runtime
inside TD) and documents "In-TouchDesigner inference" with TensorRT working.

STATUS: unverified inside TD as of writing -- this is the first real test of whether
raw TensorRT avoids the crash class the onnxruntime EP hit. Confirm this loads/runs
without crashing TD before building anything real (tracking, TD table output, etc.) on
top of it -- see python/scripts/trt_yolo26_pose.py for the minimal test harness.

Requires `cuda-python` (device memory + stream management -- lighter than pycuda, no
compiler needed) alongside `tensorrt`, both in requirements.txt.
"""

import os
import queue
import threading
import time

import numpy as np
import tensorrt as trt
from cuda import cudart

import td


# Manager registry for shutdown_and_register() pattern (prevents GPU session leaks on reload)
_manager_registry = {}


def shutdown_and_register(comp_path, new_manager):
	"""Call from each trt_*.py script's module-level instance creation, after constructing
	the new manager: shuts down whatever was previously registered for this exact comp_path
	(if anything), then registers the new one. Prevents leaking GPU memory/worker threads."""
	prev = _manager_registry.get(comp_path)
	if prev is not None:
		prev.shutdown()
	_manager_registry[comp_path] = new_manager


def printTRT(*args):
	print("[TRT]", *args)


def _cuda_check(err):
	"""cuda-python's cudart functions return (cudaError_t, ...) tuples -- raise on
	anything but cudaSuccess instead of silently continuing with a bad buffer/stream."""
	if err != cudart.cudaError_t.cudaSuccess:
		raise RuntimeError(f"CUDA error: {err} ({cudart.cudaGetErrorString(err)[1]})")


# ========== Parent COMP performance parameter management ==========
# Constants for custom performance parameters on the parent COMP
PERF_HISTORY_LEN = 30    # number of recent (throttled) samples averaged for effective FPS
PERF_LOG_INTERVAL = 1.0  # seconds between effective FPS samples
PERF_PAR_PAGE = 'Performance'
PERF_PAR_NAME = 'Effectivefps'
LATENCY_PAR_NAME = 'Pipelineframes'
PREPROCESS_MS_PAR_NAME = 'Preprocessms'
INFERENCE_MS_PAR_NAME = 'Inferencems'
POSTPROCESS_MS_PAR_NAME = 'Postprocessms'
SKIPPED_PCT_PAR_NAME = 'Frameskippedpct'
SYNC_UPDATE_INTERVAL = 0.2  # Update interval for rolling averages (seconds)
SYNC_WINDOW_SECONDS = 1.0   # Rolling window size for averaging


def _append_readonly_float(base_comp, name, label):
	"""Self-install a single read-only float custom par on PERF_PAR_PAGE if it doesn't
	already exist. Shared helper for _ensure_perf_par()."""
	if hasattr(base_comp.par, name):
		return
	page = next((pg for pg in base_comp.customPages if pg.name == PERF_PAR_PAGE), None)
	if page is None:
		page = base_comp.appendCustomPage(PERF_PAR_PAGE)
	p = page.appendFloat(name, label=label, size=1)
	p[0].default = 0.0
	p[0].readOnly = True


def _ensure_perf_par(base_comp):
	"""Self-install read-only performance parameter custom pars on the parent COMP if they
	don't already exist. Matches ONNXInferenceManager's pattern so TRT and ONNX comps have
	identical performance readouts."""
	if base_comp is None:
		return
	_append_readonly_float(base_comp, PERF_PAR_NAME, 'Effective FPS (Inference)')
	_append_readonly_float(base_comp, LATENCY_PAR_NAME, 'Pipeline Latency (Frames)')
	_append_readonly_float(base_comp, PREPROCESS_MS_PAR_NAME, 'Preprocess (ms)')
	_append_readonly_float(base_comp, INFERENCE_MS_PAR_NAME, 'Inference (ms)')
	_append_readonly_float(base_comp, POSTPROCESS_MS_PAR_NAME, 'Postprocess (ms)')
	_append_readonly_float(base_comp, SKIPPED_PCT_PAR_NAME, 'Frames Skipped (%)')


class TRTInferenceManager:
	"""Base class for raw-TensorRT model loading + threaded inference in TouchDesigner.

	Usage (mirrors ONNXInferenceManager):
	    Subclass and implement:
	    - get_engine_path(): return path to the .engine file (see build_trt_engine.py)
	    - get_onnx_path(): return path to the source .onnx file (for auto-build)
	    - get_build_model_name(): return --model argument for build_trt_engine.py
	    - preprocess(nA): TD numpyArray() -> model input tensor (float32, matching the
	      engine's expected input shape exactly -- no dynamic-shape profile support here)
	    - postprocess(outputs): List of model output ndarrays -> whatever the script needs
	      (exactly matching ONNX's session.run() return format: outputs[0], outputs[1], etc.)

	Supports models with ONE input and ANY NUMBER of outputs. Most models have 1 output
	(pose detection), some have 2 (instance segmentation: detections + mask prototypes).
	Multi-input models are not yet supported (raise NotImplementedError on load).
	
	Auto-build: If .engine file is missing, will automatically build it via subprocess
	calling python/standalone/build_trt_engine.py. First load takes 6+ minutes.
	"""

	def __init__(self):
		self.scriptOp = None

		self.engine = None
		self.context = None
		self.stream = None
		self.load_error = None
		self.is_loading = False
		self.loading_thread = None

		# Input/output tensor metadata and buffers (supports multiple outputs)
		self._input_name = None
		self._output_names = []  # List of output tensor names
		self._input_shape = None
		self._output_shapes = []  # List of output shapes
		self._d_input = None
		self._d_outputs = []  # List of device buffers for each output
		self._h_outputs = []  # List of host buffers for D2H copy results

		# Same "one persistent worker thread, maxsize=1 queue" pattern as
		# ONNXInferenceManager -- see that class's __init__ comment for the rationale.
		self._worker_thread = None
		self._work_queue = queue.Queue(maxsize=1)
		self.is_inferencing = False
		self.pending_result = None
		self.input_tensor_cache = None  # Pre-processed input for thread
		self.last_preprocess_ms = 0.0
		self.last_inference_ms = 0.0
		self.last_postprocess_ms = 0.0
		
		# Frame skipping tracking (matches ONNXInferenceManager)
		self.frames_skipped = 0  # Track how many frames we've skipped
		self.frames_skipped_final = 0  # Final count of skipped frames to report
		self._capture_abs_frame = None  # absTime.frame at the moment input was captured
		self.last_pipeline_frames = 0  # measured capture->output latency, in TD frames
		
		# Pipeline latency tracking - rolling MEDIAN of last_pipeline_frames (matches ONNXInferenceManager)
		self._pipeline_frames_samples = []
		self._smoothed_pipeline_frames = 0.0
		self._last_sync_update_time = 0.0
		
		# Rolling average samples for parent COMP performance stats (matches ONNXInferenceManager)
		self._preprocess_ms_samples = []
		self._inference_ms_samples = []
		self._postprocess_ms_samples = []
		self._skipped_pct_samples = []
		self._last_perf_metrics_update_time = 0.0
		
		# Effective FPS tracking (matches ONNXInferenceManager)
		self._perf_history = []  # recent effective-fps samples
		self._last_perf_log_time = 0.0
		
		# Utility references (same as ONNXInferenceManager)
		try:
			import numpy_util as npu
			self.npu = npu
		except ImportError:
			self.npu = None

	# ========== Subclass hooks ==========

	def get_engine_path(self):
		raise NotImplementedError("Subclass must implement get_engine_path()")

	def get_onnx_path(self):
		"""Return path to the source .onnx file (for auto-build). Override in subclass."""
		raise NotImplementedError("Subclass must implement get_onnx_path()")

	def get_build_model_name(self):
		"""Return --model argument for build_trt_engine.py. Override in subclass."""
		raise NotImplementedError("Subclass must implement get_build_model_name()")

	def preprocess(self, nA):
		raise NotImplementedError("Subclass must implement preprocess()")

	def postprocess(self, outputs):
		"""Transform model outputs to final result.
		
		Args:
			outputs: List of output arrays from run_inference() - outputs[0], outputs[1], etc.
			         Matches ONNX's session.run() return format exactly.
		
		Returns:
			Whatever the script needs (usually a numpy array for Script TOP output)
		"""
		raise NotImplementedError("Subclass must implement postprocess()")

	def on_engine_loaded(self, engine, context):
		"""Optional hook, called once after the engine, context, and device buffers are
		all ready. Runs on the loading thread -- do NOT touch TD Par/OP objects here,
		same rule as ONNXInferenceManager.on_model_loaded()."""
		pass

	def _warmup_engine(self, input_shape, has_fp16=False):
		"""Run several dummy inference passes to let TensorRT optimize CUDA kernels.
		TensorRT engines need warmup to reach optimal performance - first few runs are
		slower as it selects kernels and allocates memory patterns. Running 10-20 warmup
		passes after loading can give 2-3x speedup on actual inference.
		
		Runs on the loading thread, safe to do blocking CUDA work here."""
		try:
			# Create dummy input tensor (zeros are fine for warmup)
			dummy_input = np.zeros(input_shape, dtype=np.float32)
			
			# Run warmup iterations - more iterations for FP16 (needs more kernel tuning)
			warmup_iterations = 20 if has_fp16 else 10
			
			for _ in range(warmup_iterations):
				# Copy input to device
				err = cudart.cudaMemcpyAsync(
					self._d_input, dummy_input.ctypes.data, dummy_input.nbytes,
					cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)[0]
				_cuda_check(err)
				
				# Execute
				ok = self.context.execute_async_v3(self.stream)
				if not ok:
					break
				
				# Copy ALL outputs back (even though we discard results)
				for d_output, h_output in zip(self._d_outputs, self._h_outputs):
					err = cudart.cudaMemcpyAsync(
						h_output.ctypes.data, d_output, h_output.nbytes,
						cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)[0]
					_cuda_check(err)
				
				# Wait for completion
				err = cudart.cudaStreamSynchronize(self.stream)[0]
				_cuda_check(err)
		except:
			pass  # Warmup failure shouldn't block engine usage

	def printTRT(self, *args):
		"""Instance method wrapper for printTRT() function, so subclasses can call self.printTRT()."""
		printTRT(*args)

	# ========== Loading ==========

	def loadTRT(self, scriptOp):
		"""Initiate threaded engine loading."""
		if self.is_loading:
			return
		self.engine = None
		scriptOp.par.Loadstatus = "loading"
		self.loading_thread = threading.Thread(target=self._load_engine_thread)
		self.loading_thread.daemon = True
		self.loading_thread.start()

	def _load_engine_thread(self):
		"""Background thread for deserializing the engine and allocating device buffers.
		Nothing here touches TD Par/OP objects -- see onnx_inference_manager.py's
		THREAD CONFLICT lesson (docs/learnings and memory notes) for why that matters."""
		self.is_loading = True
		self.load_error = None
		try:
			engine_path = self.get_engine_path()
			if not os.path.isfile(engine_path):
				# Auto-build if ONNX exists
				self.printTRT(f"[TRT] Engine not found: {engine_path}")
				self._auto_build_engine()
				# Check again after build
				if not os.path.isfile(engine_path):
					raise FileNotFoundError(
						f"Engine build failed or file still missing: {engine_path}")

			logger = trt.Logger(trt.Logger.WARNING)
			runtime = trt.Runtime(logger)
			with open(engine_path, 'rb') as f:
				engine = runtime.deserialize_cuda_engine(f.read())
			if engine is None:
				raise RuntimeError("deserialize_cuda_engine() returned None -- engine "
					"file may be corrupt or built with an incompatible TensorRT version.")
			context = engine.create_execution_context()

			# Detect all inputs and outputs (most models have 1 input, but may have multiple outputs)
			num_tensors = engine.num_io_tensors
			input_names = []
			output_names = []
			for i in range(num_tensors):
				name = engine.get_tensor_name(i)
				mode = engine.get_tensor_mode(name)
				if mode == trt.TensorIOMode.INPUT:
					input_names.append(name)
				else:
					output_names.append(name)

			# For now, support single input (matches all our current models)
			if len(input_names) != 1:
				raise NotImplementedError(f"Multiple inputs not yet supported (found {len(input_names)})")
			
			input_name = input_names[0]
			input_shape = tuple(engine.get_tensor_shape(input_name))
			input_nbytes = int(np.prod(input_shape)) * np.dtype(np.float32).itemsize

			# Allocate device memory for input
			err, d_input = cudart.cudaMalloc(input_nbytes)
			_cuda_check(err)
			context.set_tensor_address(input_name, d_input)

			# Allocate device/host memory for ALL outputs (list, matches ONNX pattern)
			d_outputs = []
			h_outputs = []
			output_shapes = []
			for output_name in output_names:
				output_shape = tuple(engine.get_tensor_shape(output_name))
				output_nbytes = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize
				
				err, d_output = cudart.cudaMalloc(output_nbytes)
				_cuda_check(err)
				context.set_tensor_address(output_name, d_output)
				
				h_output = np.empty(output_shape, dtype=np.float32)
				d_outputs.append(d_output)
				h_outputs.append(h_output)
				output_shapes.append(output_shape)

			# Create CUDA stream
			err, stream = cudart.cudaStreamCreate()
			_cuda_check(err)

			# Store everything as instance attributes
			self.engine = engine
			self.context = context
			self.stream = stream
			self._input_name = input_name
			self._output_names = output_names
			self._input_shape = input_shape
			self._output_shapes = output_shapes
			self._d_input = d_input
			self._d_outputs = d_outputs
			self._h_outputs = h_outputs

			# Check if engine is using FP16 precision
			has_fp16 = False
			try:
				for i in range(engine.num_layers):
					layer = engine.get_layer(i)
					if layer.precision == trt.DataType.HALF:
						has_fp16 = True
						break
			except:
				# If we can't check layers, assume FP16 if engine name suggests it
				has_fp16 = 'fp16' in engine_path.lower()
			
			self.on_engine_loaded(engine, context)
			
			# Warmup: run several dummy inference passes to let TensorRT optimize
			# (CUDA kernel selection, memory allocation patterns, etc.)
			self._warmup_engine(input_shape, has_fp16)
		except Exception as e:
			self.load_error = str(e)
		finally:
			self.is_loading = False

	def get_loading_status(self):
		if self.engine is not None:
			return "loaded"
		elif self.is_loading:
			return "loading"
		elif self.load_error:
			return f"error: {self.load_error}"
		return "not loaded"

	def needs_prewarm(self):
		"""True while the engine hasn't loaded (or failed) and this scriptOp needs
		force-cooking to progress. Without this, TOPs with no downstream wires never
		cook and the engine never loads. See schedule_prewarm_cook() docstring."""
		return self.engine is None and self.load_error is None

	def schedule_prewarm_cook(self, scriptOp, callbacksDAT, delayFrames=1):
		"""Force scriptOp to cook on a future frame if the engine still needs loading.
		Safe to call from module-level code (scriptOp's own Callbacks DAT import time) -
		this never calls .cook() directly, only td.run(), so it never reenters compile.
		See ONNXInferenceManager.schedule_prewarm_cook() for full rationale."""
		if not self.needs_prewarm():
			return
		td.run(
			f"op({callbacksDAT.path!r}).module.inference_manager._prewarm_tick("
			f"op({scriptOp.path!r}), op({callbacksDAT.path!r}))",
			delayFrames=delayFrames,
		)

	def _prewarm_tick(self, scriptOp, callbacksDAT):
		"""Force-cook, invoked via schedule_prewarm_cook's deferred td.run(). Reschedules
		itself until the engine loads or fails, then the chain stops."""
		if not self.needs_prewarm():
			return
		scriptOp.cook(force=True)
		self.schedule_prewarm_cook(scriptOp, callbacksDAT)

	# ========== Parent COMP Performance Stats ==========

	@staticmethod
	def _rolling_average(samples, new_value, now, window_seconds=SYNC_WINDOW_SECONDS):
		"""Append (now, new_value), prune anything older than window_seconds, and return
		the average of what remains. Matches ONNXInferenceManager's rolling average."""
		samples.append((now, new_value))
		cutoff = now - window_seconds
		while samples and samples[0][0] < cutoff:
			samples.pop(0)
		return sum(v for _, v in samples) / len(samples) if samples else 0.0

	def _record_perf_sample(self):
		"""Throttled (once per PERF_LOG_INTERVAL) in-memory performance sample -- keeps
		the last PERF_HISTORY_LEN readings and pushes the rolling-average effective FPS
		to a read-only 'Effectivefps' custom par on the parent COMP. Matches
		ONNXInferenceManager's pattern."""
		now = time.perf_counter()
		if now - self._last_perf_log_time < PERF_LOG_INTERVAL:
			return
		self._last_perf_log_time = now

		total_ms = self.last_preprocess_ms + self.last_inference_ms + self.last_postprocess_ms
		eff_fps = 1000.0 / total_ms if total_ms > 0 else 0.0
		self._perf_history.append(eff_fps)
		if len(self._perf_history) > PERF_HISTORY_LEN:
			self._perf_history.pop(0)

		avg_fps = sum(self._perf_history) / len(self._perf_history)
		try:
			base = self.scriptOp.parent()
			_ensure_perf_par(base)
			base.par.Effectivefps = round(avg_fps, 1)
		except:
			pass

	def _update_sync_estimate(self):
		"""Maintain a realistic (rolling-MEDIAN) estimate of pipeline latency, recomputed a
		few times a second (SYNC_UPDATE_INTERVAL). Uses the window's MEDIAN, not a rolling
		max or average. Matches ONNXInferenceManager's pattern."""
		now = time.perf_counter()
		if now - self._last_sync_update_time < SYNC_UPDATE_INTERVAL:
			return
		self._last_sync_update_time = now

		self._pipeline_frames_samples.append((now, self.last_pipeline_frames))
		cutoff = now - SYNC_WINDOW_SECONDS
		self._pipeline_frames_samples = [(t, v) for t, v in self._pipeline_frames_samples if t >= cutoff]
		if not self._pipeline_frames_samples:
			return
		values = sorted(v for _, v in self._pipeline_frames_samples)
		mid = len(values) // 2
		self._smoothed_pipeline_frames = (
			values[mid] if len(values) % 2 == 1
			else (values[mid - 1] + values[mid]) / 2.0
		)

		try:
			base = self.scriptOp.parent()
			_ensure_perf_par(base)
			base.par.Pipelineframes = round(self._smoothed_pipeline_frames, 1)
		except:
			pass

	def _update_perf_metrics(self):
		"""Update parent COMP's read-only performance parameters with rolling averages
		of per-stage timing (Preprocessms/Inferencems/Postprocessms) and frame-skip-rate
		(Frameskippedpct). Matches ONNXInferenceManager's pattern so TRT and ONNX comps
		have identical readouts."""
		now = time.perf_counter()
		if now - self._last_perf_metrics_update_time < SYNC_UPDATE_INTERVAL:
			return
		self._last_perf_metrics_update_time = now

		skipped_pct = 100.0 * self.frames_skipped_final / (self.frames_skipped_final + 1)
		avg_preprocess = self._rolling_average(self._preprocess_ms_samples, self.last_preprocess_ms, now)
		avg_inference = self._rolling_average(self._inference_ms_samples, self.last_inference_ms, now)
		avg_postprocess = self._rolling_average(self._postprocess_ms_samples, self.last_postprocess_ms, now)
		avg_skipped_pct = self._rolling_average(self._skipped_pct_samples, skipped_pct, now)

		try:
			base = self.scriptOp.parent()
			_ensure_perf_par(base)
			base.par.Preprocessms = round(avg_preprocess, 2)
			base.par.Inferencems = round(avg_inference, 2)
			base.par.Postprocessms = round(avg_postprocess, 2)
			base.par.Frameskippedpct = round(avg_skipped_pct, 1)
		except Exception:
			pass

	# ========== Inference ==========

	def run_inference(self, input_tensor):
		"""GPU-bound work ONLY -- H2D copy, execute, D2H copy -- meant to be called from
		the background worker thread, same contract as ONNXInferenceManager.run_inference().
		
		Returns a LIST of output arrays (even if only one output), matching ONNX's
		session.run() behavior so subclasses can use identical postprocess() logic
		regardless of backend (outputs[0], outputs[1], etc.)."""
		input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
		
		# Copy input to device
		err = cudart.cudaMemcpyAsync(
			self._d_input, input_tensor.ctypes.data, input_tensor.nbytes,
			cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)[0]
		_cuda_check(err)

		# Execute
		ok = self.context.execute_async_v3(self.stream)
		if not ok:
			raise RuntimeError("execute_async_v3() returned False")

		# Copy ALL outputs back to host
		for i, (d_output, h_output) in enumerate(zip(self._d_outputs, self._h_outputs)):
			err = cudart.cudaMemcpyAsync(
				h_output.ctypes.data, d_output, h_output.nbytes,
				cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)[0]
			_cuda_check(err)

		# Wait for everything to complete
		err = cudart.cudaStreamSynchronize(self.stream)[0]
		_cuda_check(err)
		
		# Return list of output copies (matches ONNX session.run() behavior)
		return [h_output.copy() for h_output in self._h_outputs]

	def infer_sync(self, input_tensor):
		"""Synchronous convenience path (no worker thread) -- for the POC test harness
		and any one-off/non-realtime use. Real per-frame TD scripts should use the
		threaded worker path instead (see ONNXInferenceManager.onCook()'s docstring for
		the pattern this class is meant to eventually mirror)."""
		t0 = time.perf_counter()
		output = self.run_inference(input_tensor)
		self.last_inference_ms = (time.perf_counter() - t0) * 1000
		return output

	# ========== Worker Thread Infrastructure ==========

	def _ensure_worker_started(self):
		"""Start the persistent worker thread if it hasn't been started yet."""
		if self._worker_thread is None or not self._worker_thread.is_alive():
			self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
			self._worker_thread.start()

	def _worker_loop(self):
		"""Persistent background thread that processes inference requests from the queue."""
		while True:
			work_item = self._work_queue.get()
			if work_item is None:  # Shutdown sentinel
				break
			input_tensor = work_item
			try:
				t0 = time.perf_counter()
				output = self.run_inference(input_tensor)
				elapsed_ms = (time.perf_counter() - t0) * 1000
				self.pending_result = (output, elapsed_ms)
			except Exception:
				self.pending_result = (None, 0.0)
			finally:
				self.is_inferencing = False
				self.frames_skipped_final = self.frames_skipped
				self.frames_skipped = 0
	# ========== TouchDesigner Callbacks ==========

	def onSetupParameters(self, scriptOp):
		"""Base parameter setup - subclasses should call super().onSetupParameters() first."""
		# Ensure Script TOP cooks continuously (not just when viewed)
		scriptOp.render = True
		scriptOp.viewer = True
		
		page = scriptOp.appendCustomPage('TRT Engine')
		
		p = page.appendPulse('Loadengine', label='Load Engine')
		p.help = 'Trigger TensorRT engine loading (deserialization + device buffer allocation)'
		
		p = page.appendStr('Loadstatus', label='Load Status')
		p.readOnly = True
		scriptOp.par.Loadstatus = 'not loaded'
		
		# Performance timing stats (matching ONNXInferenceManager)
		p = page.appendStr('Preprocessms', label='Preprocess (ms)')
		p.readOnly = True
		scriptOp.par.Preprocessms = '0.00'
		
		p = page.appendStr('Inferencems', label='Inference (ms)')
		p.readOnly = True
		scriptOp.par.Inferencems = '0.00'
		
		p = page.appendStr('Postprocessms', label='Postprocess (ms)')
		p.readOnly = True
		scriptOp.par.Postprocessms = '0.00'
		
		p = page.appendStr('Totalms', label='Total (ms)')
		p.readOnly = True
		scriptOp.par.Totalms = '0.00'

	def onPulse(self, par):
		"""Handle button presses."""
		if par.name == 'Loadengine':
			self.loadTRT(par.owner)

	def onCook(self, scriptOp):
		"""Main per-frame callback for Script TOP - handles model loading, inference dispatch, and result copy."""
		self.scriptOp = scriptOp
		
		# Update status display (only if parameters exist - onSetupParameters may not have run yet)
		if hasattr(scriptOp.par, 'Loadstatus'):
			status = self.get_loading_status()
			scriptOp.par.Loadstatus = status
		if hasattr(scriptOp.par, 'Preprocessms'):
			scriptOp.par.Preprocessms = f"{self.last_preprocess_ms:.2f}"
		if hasattr(scriptOp.par, 'Inferencems'):
			scriptOp.par.Inferencems = f"{self.last_inference_ms:.2f}"
		if hasattr(scriptOp.par, 'Postprocessms'):
			scriptOp.par.Postprocessms = f"{self.last_postprocess_ms:.2f}"
		if hasattr(scriptOp.par, 'Totalms'):
			total_ms = self.last_preprocess_ms + self.last_inference_ms + self.last_postprocess_ms
			scriptOp.par.Totalms = f"{total_ms:.2f}"
		
		# Auto-load on first cook if not loaded/loading
		if self.engine is None and not self.is_loading and self.load_error is None:
			self.loadTRT(scriptOp)
			if hasattr(scriptOp.par, 'Loadstatus'):
				scriptOp.par.Loadstatus = 'loading'
		
		# Wait for loading to complete
		if self.engine is None:
			return
		
		# Skip work while TD is paused (but keep loading)
		if not scriptOp.time.play:
			return
		
		# Check if previous inference completed
		if self.pending_result is not None:
			output, elapsed_ms = self.pending_result
			self.pending_result = None
			self.last_inference_ms = elapsed_ms
			if output is not None:
				t0 = time.perf_counter()
				# output is already a list from run_inference() - pass it directly
				final_output = self.postprocess(output)
				self.last_postprocess_ms = (time.perf_counter() - t0) * 1000
				scriptOp.copyNumpyArray(final_output)
				# Hook for subclass post-publish work (e.g. table writes)
				if hasattr(self, 'on_result_published'):
					self.on_result_published()
				# Pipeline latency measurement (capture->output, in TD frames)
				if self._capture_abs_frame is not None:
					try:
						self.last_pipeline_frames = td.absTime.frame - self._capture_abs_frame
					except Exception:
						pass
					self._update_sync_estimate()
					self._update_perf_metrics()
				# Record effective FPS sample (once per second)
				self._record_perf_sample()
			return
		
		# If inference is still running, skip this frame (natural frame skipping via threading)
		if self.is_inferencing:
			self.frames_skipped += 1
			return
		
		# Start new inference
		# Get input TOP
		input_top = scriptOp.inputs[0] if len(scriptOp.inputs) > 0 else None
		if input_top is None:
			return
		
		# Extract numpy array from input TOP
		nA = input_top.numpyArray(delayed=True)
		if nA is None or nA.size == 0:
			return
		
		# Capture absTime.frame for pipeline latency measurement
		try:
			self._capture_abs_frame = td.absTime.frame
		except Exception:
			self._capture_abs_frame = None
			
		# Preprocess on main thread
		t0 = time.perf_counter()
		input_tensor = self.preprocess(nA)
		self.last_preprocess_ms = (time.perf_counter() - t0) * 1000
		
		# Dispatch to worker thread
		self._ensure_worker_started()
		self.is_inferencing = True
		self._work_queue.put_nowait(input_tensor)

	# ========== Utility Methods ==========

	def _par_or_default(self, name, default):
		"""Read a live custom par by name if it exists on scriptOp, else fall back to default."""
		if self.scriptOp and hasattr(self.scriptOp.par, name):
			return getattr(self.scriptOp.par, name).eval()
		return default

	def _auto_build_engine(self):
		"""Auto-build missing .engine file via subprocess calling build_trt_engine.py."""
		import subprocess
		import sys
		
		try:
			onnx_path = self.get_onnx_path()
			if not os.path.isfile(onnx_path):
				raise FileNotFoundError(
					f"Cannot auto-build: ONNX source not found: {onnx_path}")
			
			model_name = self.get_build_model_name()
			build_script = os.path.join(project.folder, 'python', 'standalone', 'build_trt_engine.py')
			venv_python = os.path.join(project.folder, '.venv', 'Scripts', 'python.exe')
			
			if not os.path.isfile(venv_python):
				raise FileNotFoundError(
					f"Cannot auto-build: venv python not found: {venv_python}")
			
			self.printTRT(f"[TRT] Auto-building engine from {onnx_path}...")
			self.printTRT(f"[TRT] This will take 6+ minutes. Check TextPort for progress.")
			
			# Run build script with live output
			cmd = [venv_python, build_script, '--model', model_name]
			process = subprocess.Popen(
				cmd,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				bufsize=1
			)
			
			# Stream output to TextPort
			for line in process.stdout:
				self.printTRT(line.rstrip())
			
			return_code = process.wait()
			if return_code != 0:
				raise RuntimeError(f"Build script failed with exit code {return_code}")
				
			self.printTRT(f"[TRT] Auto-build complete!")
			
		except NotImplementedError:
			# Subclass doesn't implement get_onnx_path/get_build_model_name - skip auto-build
			raise FileNotFoundError(
				f"Engine file not found: {self.get_engine_path()} -- build it first with "
				"python/standalone/build_trt_engine.py")
		except Exception as e:
			self.printTRT(f"[TRT] Auto-build failed: {e}")
			raise

	def shutdown(self):
		"""Cleanly release this manager's GPU resources and stop its background worker thread."""
		if self._worker_thread is not None and self._worker_thread.is_alive():
			try:
				self._work_queue.put_nowait(None)  # Shutdown sentinel
			except queue.Full:
				pass
			self._worker_thread.join(timeout=5.0)
		
		# Free CUDA resources
		if self._d_input is not None:
			cudart.cudaFree(self._d_input)
			self._d_input = None
		
		# Free all output buffers
		if self._d_outputs is not None:
			for d_output in self._d_outputs:
				if d_output is not None:
					cudart.cudaFree(d_output)
			self._d_outputs = None
		
		if self.stream is not None:
			cudart.cudaStreamDestroy(self.stream)
			self.stream = None
		
		self.context = None
		self.engine = None
