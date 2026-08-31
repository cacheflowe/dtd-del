"""
Standalone (no TouchDesigner) webcam -> YOLO26-Pose ONNX -> skeleton overlay demo.

Purpose: measure how much latency/FPS headroom exists once TD's render/cook pipeline
(GPU staging-buffer readback, Script TOP cook dependency chain, vsync-bound onCook)
is removed entirely. Reuses the exact model + postprocess/tracker logic from
python/scripts/onnx_yolo26_pose.py and python/util/object_tracker.py (KeypointTracker
is pure numpy, zero TD dependencies, so it imports unmodified here).

Uses this project's own .venv (built from TD's Python 3.11 binary via TDPyEnvManager --
see .ai/skills/td-vscode-python-environment.md), which already has numpy and onnxruntime-gpu
installed from requirements.txt. Run with:
    .venv\\Scripts\\python.exe python\\standalone\\webcam_yolo_pose.py [--camera 0] [--model yolo26s-pose] [--no-window]

cv2 is deliberately NOT in requirements.txt. TD's own bundled Python (a completely
separate install at TouchDesigner/bin, sharing nothing with .venv by default -- see
.venv/pyvenv.cfg's include-system-site-packages=false) ships a full GUI build of cv2
(confirmed live: 4.11.0, imshow works) that TDPyEnvManager never copies into .venv. This
script appends TD's site-packages to sys.path -- same append-only pattern
TDPyEnvManager itself uses in reverse -- so `import cv2` falls through to TD's copy
without adding a pip dependency or touching .venv/requirements.txt at all. numpy/
onnxruntime/tensorrt still resolve from .venv first since it's earlier on sys.path.

Press 'q' or ESC (window focused) to quit. Prints rolling pre/infer/post/total timing
+ effective FPS to the console every ~1s, same cadence as the TD Perf table.
"""

import argparse
import glob
import os
import sys
import threading
import time

import numpy as np

# TD's own bundled Python ships a full GUI build of cv2 (unlike the opencv-python-headless
# this project used to pip-install) -- append it to sys.path so `import cv2` below falls
# through to it, since .venv doesn't have its own copy. Appended, not inserted first, so
# it never shadows anything .venv DOES have (numpy, onnxruntime, tensorrt all still
# resolve from .venv). Safe if TD isn't installed at this path: cv2 import below will
# just fail with a clear ModuleNotFoundError instead of silently misbehaving.
_TD_SITE_PACKAGES = r'C:\Program Files\Derivative\TouchDesigner\bin\Lib\site-packages'
if os.path.isdir(_TD_SITE_PACKAGES) and _TD_SITE_PACKAGES not in sys.path:
	sys.path.append(_TD_SITE_PACKAGES)

import cv2

# The pip nvidia-*-cu12 wheels drop their DLLs under site-packages/nvidia/*/bin instead of
# on PATH -- TD itself never needs this (its own bin/ dir already has matching cudnn/cuda
# DLLs alongside python.exe), but a standalone script using this same .venv does. Without
# this, onnxruntime silently falls back to CPUExecutionProvider (confirmed live: "Error
# loading ... which depends on cudnn64_9.dll which is missing", ~10x slower inference).
# The `tensorrt` pip package (requirements.txt) follows the same pattern but drops its
# DLLs under site-packages/tensorrt_libs/ instead, so that's globbed too.
if sys.platform == 'win32':
	# .venv's OWN site-packages specifically -- NOT derived from cv2.__file__, since cv2 now
	# resolves from TD's install (appended above), not .venv, and would point this at the
	# wrong directory entirely.
	_SITE_PACKAGES = os.path.join(sys.prefix, 'Lib', 'site-packages')
	_nvidia_bin_dirs = glob.glob(os.path.join(_SITE_PACKAGES, 'nvidia', '*', 'bin'))
	_nvidia_bin_dirs += glob.glob(os.path.join(_SITE_PACKAGES, 'tensorrt_libs'))
	for _bin_dir in _nvidia_bin_dirs:
		os.add_dll_directory(_bin_dir)
	# Belt-and-suspenders: onnxruntime's own CUDA provider loader doesn't reliably honor
	# os.add_dll_directory()-registered paths (confirmed live -- cudnn64_9.dll still
	# reported missing with only add_dll_directory set), so also prepend to PATH, the
	# mechanism onnxruntime's own GPU docs recommend.
	os.environ['PATH'] = os.pathsep.join(_nvidia_bin_dirs) + os.pathsep + os.environ.get('PATH', '')

import onnxruntime as ort

# ---- make python/util importable (KeypointTracker has no TD dependency) ----
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_UTIL_DIR = os.path.join(_PROJECT_ROOT, 'python', 'util')
if _UTIL_DIR not in sys.path:
	sys.path.insert(0, _UTIL_DIR)

import object_tracker  # noqa: E402
KeypointTracker = object_tracker.KeypointTracker

# ==================== MODEL / OUTPUT LAYOUT (matches onnx_yolo26_pose.py) ====================
KEYPOINT_NAMES = [
	'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
	'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
	'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
	'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
NUM_KEYPOINTS = len(KEYPOINT_NAMES)

SKELETON_EDGES = [
	(0, 1), (0, 2), (1, 3), (2, 4),
	(0, 5), (0, 6), (5, 6),
	(5, 7), (7, 9), (6, 8), (8, 10),
	(5, 11), (6, 12), (11, 12),
	(11, 13), (13, 15), (12, 14), (14, 16),
]
_UNSTABLE_KEYPOINT_KEYWORDS = ('elbow', 'wrist', 'knee', 'ankle')
DISTANCE_KEYPOINT_INDICES = [
	i for i, name in enumerate(KEYPOINT_NAMES)
	if not any(kw in name for kw in _UNSTABLE_KEYPOINT_KEYWORDS)
]

MODEL_INPUT_SIZE = 640
# 0.005 (onnx_yolo26_pose.py's default) was tuned for one specific TD scene where real
# signal ran at near-zero confidence -- a normal well-lit webcam doesn't need that, and
# leaving it that low here lets dozens of low-confidence noise boxes through, which
# measurably slowed postprocess/tracking (confirmed live: ~30-40ms -> <1ms once raised).
# Vectorized postprocess() below makes this less critical than it used to be, but a
# realistic threshold is still both faster and cleaner. Override with --conf.
CONF_THRESHOLD = 0.5
MIN_BOX_WIDTH = 0.1
MIN_BOX_HEIGHT = 0.1
MAX_MATCH_DIST = 0.5
DUP_DIST_FACTOR = 0.5
TRACKER_MAX_AGE = 15
TRACKER_MIN_HITS = 3

BOX_COLOR = (0, 255, 0)
SKELETON_COLOR = (0, 255, 255)
KEYPOINT_COLOR = (0, 128, 255)


def get_model_path(variant):
	return os.path.join(_PROJECT_ROOT, 'data', 'ml', 'yolo26', f'{variant}.onnx')


def make_session(model_path, use_trt=False):
	"""Same CUDA EP config as onnx_inference_manager.providers() -- HEURISTIC conv algo
	search avoids the periodic ~200ms EXHAUSTIVE-search stalls documented there.

	use_trt=True prepends TensorrtExecutionProvider (falls back to CUDA/CPU if TRT can't
	handle a node). Fixed 640x640 input here means only one engine build ever, but that
	first build (and every subsequent process launch, unless the cache dir below
	survives) is a real one-time cost -- tens of seconds, not milliseconds -- before the
	first frame; the trt_engine_cache_path below persists it across runs so only the
	very first run after a model/options change pays that cost."""
	cuda_options = {'device_id': 0, 'cudnn_conv_algo_search': 'HEURISTIC'}
	providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
	if use_trt:
		cache_dir = os.path.join(_PROJECT_ROOT, 'data', 'ml', 'trt_cache')
		os.makedirs(cache_dir, exist_ok=True)
		trt_options = {
			'device_id': 0,
			'trt_fp16_enable': True,
			'trt_engine_cache_enable': True,
			'trt_engine_cache_path': cache_dir,
		}
		providers.insert(0, ('TensorrtExecutionProvider', trt_options))
		print(f"[ONNX] TensorRT requested -- engine cache: {cache_dir} (first run after any "
			"change here builds a fresh engine, tens of seconds, not milliseconds)")
	so = ort.SessionOptions()
	so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
	session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
	active = session.get_providers()
	print(f"[ONNX] Active providers: {active}")
	if use_trt and 'TensorrtExecutionProvider' not in active:
		# ORT's own fallback when TRT fails to load drops straight to CPU-only, skipping
		# CUDA entirely (confirmed live) -- rebuild explicitly without TRT rather than
		# silently running 10x+ slower on CPU for the rest of the session.
		print("[ONNX] WARNING: TensorRT unavailable (needs the separate NVIDIA TensorRT "
			"SDK/pip package, not just onnxruntime-gpu's nvidia-*-cu12 wheels) -- retrying without it.")
		session = ort.InferenceSession(model_path, sess_options=so, providers=providers[1:])
		active = session.get_providers()
		print(f"[ONNX] Active providers: {active}")
	if 'CUDAExecutionProvider' not in active:
		print("[ONNX] WARNING: running on CPU only -- install onnxruntime-gpu + matching CUDA/cuDNN.")
	return session


_padded_buf = None
_padded_buf_shape = None


def letterbox(frame, size=MODEL_INPUT_SIZE):
	"""Resize+pad to a square `size x size` input, preserving aspect ratio (standard
	Ultralytics letterbox). Returns the padded RGB float32 CHW tensor plus the
	scale/pad needed to map model-space (0-1) coords back to original pixel space."""
	global _padded_buf, _padded_buf_shape
	h, w = frame.shape[:2]
	r = min(size / w, size / h)
	new_w, new_h = round(w * r), round(h * r)
	pad_w, pad_h = size - new_w, size - new_h
	left, top = pad_w // 2, pad_h // 2

	resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
	# Reuse one padded buffer across calls (camera resolution is constant) -- only the
	# border pixels need the 114 fill once; every later call just overwrites the inner
	# region, avoiding a fresh np.full() allocation+fill every frame.
	needed_shape = (size, size, 3)
	if _padded_buf is None or _padded_buf_shape != needed_shape:
		_padded_buf = np.full(needed_shape, 114, dtype=np.uint8)
		_padded_buf_shape = needed_shape
	padded = _padded_buf
	padded[top:top + new_h, left:left + new_w] = resized

	rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
	# Model's 'pixel_values' input expects normalized 0-1 floats -- onnx_yolo26_pose.py
	# never divides because TD's own TOP numpyArray() is already 0-1 float; a raw uint8
	# OpenCV frame is 0-255, so skipping this made every frame look blown-out/saturated
	# to the model (confirmed live: this was why real, well-lit people went undetected
	# while incidental background noise occasionally still fired).
	tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]  # (1, 3, size, size)
	return np.ascontiguousarray(tensor), r, left, top


def postprocess(outputs, r, pad_left, pad_top, orig_w, orig_h, conf_threshold):
	"""Fully vectorized (no per-detection/per-keypoint Python loop) -- an early version
	unletterboxed each box/keypoint with a plain Python function call per value, which
	dominated postprocess time once more than a handful of candidates cleared
	conf_threshold (measured live: ~50-80ms for ~dozens of candidates vs <1ms vectorized)."""
	pred = outputs[0][0]  # (300, 57)
	boxes_n = pred[:, 0:4].copy()
	confidences = pred[:, 4].copy()
	keypoints_n = pred[:, 6:6 + NUM_KEYPOINTS * 3].reshape(-1, NUM_KEYPOINTS, 3).copy()

	valid = confidences > conf_threshold
	valid &= (boxes_n[:, 2] - boxes_n[:, 0] >= MIN_BOX_WIDTH) & (boxes_n[:, 3] - boxes_n[:, 1] >= MIN_BOX_HEIGHT)
	boxes_n, confidences, keypoints_n = boxes_n[valid], confidences[valid], keypoints_n[valid]
	boxes_n = np.clip(boxes_n, 0.0, 1.0)

	size = MODEL_INPUT_SIZE
	# Map model-space (0-1) -> letterboxed-pixel -> original-frame pixel, box+keypoints at once.
	boxes_px = boxes_n.copy()
	boxes_px[:, [0, 2]] = (boxes_px[:, [0, 2]] * size - pad_left) / r
	boxes_px[:, [1, 3]] = (boxes_px[:, [1, 3]] * size - pad_top) / r

	kpts_px = keypoints_n.copy()
	kpts_px[:, :, 0] = (kpts_px[:, :, 0] * size - pad_left) / r
	kpts_px[:, :, 1] = (kpts_px[:, :, 1] * size - pad_top) / r

	boxes_norm = boxes_px / np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
	kpts_norm = kpts_px.copy()
	kpts_norm[:, :, 0] /= orig_w
	kpts_norm[:, :, 1] /= orig_h

	boxes_px_list = boxes_px.tolist()
	kpts_px_list = kpts_px.tolist()
	boxes_norm_list = boxes_norm.tolist()
	kpts_norm_list = kpts_norm.tolist()
	scores_list = confidences.tolist()

	detections = []
	for i in range(len(boxes_px_list)):
		detections.append({
			# Normalized by original frame size so KeypointTracker's distance metric
			# stays scale-invariant, same convention onnx_yolo26_pose.py relies on.
			'box': boxes_norm_list[i],
			'score': float(scores_list[i]),
			'keypoints': kpts_norm_list[i],
			'keypoints_px': kpts_px_list[i],
			'box_px': boxes_px_list[i],
		})
	return detections


class CvPreviewWindow:
	"""Real cv2.imshow/waitKey window -- usable now that cv2 resolves from TD's own
	bundled full-GUI build (see module docstring) instead of the headless pip build
	this project used to install. Preferred over TkPreviewWindow when available:
	no PIL round-trip per frame, and imshow/waitKey are what the rest of this
	project's OpenCV-based tooling already expects."""

	def __init__(self, title='YOLO26 Pose (standalone)'):
		self.title = title
		self.quit_requested = False
		cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)

	def show(self, frame_bgr):
		cv2.imshow(self.title, frame_bgr)
		key = cv2.waitKey(1) & 0xFF
		if key in (ord('q'), 27) or cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
			self.quit_requested = True

	def close(self):
		cv2.destroyWindow(self.title)


class TkPreviewWindow:
	"""Plain tkinter + PIL preview window -- fallback for when cv2 doesn't have
	highgui built in (e.g. .venv's own opencv-python-headless taking priority over
	TD's full-GUI build again, since .venv resolves first on sys.path). Adds no new
	dependency: tkinter ships with TD's own CPython build, Pillow is already in
	requirements.txt."""

	def __init__(self, title='YOLO26 Pose (standalone)'):
		import tkinter as tk
		self._tk = tk
		self.root = tk.Tk()
		self.root.title(title)
		self.label = tk.Label(self.root)
		self.label.pack()
		self.quit_requested = False
		self.root.protocol('WM_DELETE_WINDOW', self._on_close)
		self.root.bind('<Key>', self._on_key)

	def _on_close(self):
		self.quit_requested = True

	def _on_key(self, event):
		if event.keysym in ('q', 'Escape'):
			self.quit_requested = True

	def show(self, frame_bgr):
		from PIL import Image, ImageTk
		rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		photo = ImageTk.PhotoImage(Image.fromarray(rgb))
		self.label.configure(image=photo)
		self.label.image = photo  # keep a reference, Tk drops it otherwise
		self.root.update_idletasks()
		self.root.update()

	def close(self):
		self.root.destroy()


class ThreadedCapture:
	"""Runs cv2.VideoCapture.read() in a background thread and always hands the main
	loop the newest frame immediately, instead of blocking for a new one -- inference
	is free to run faster than the camera's own delivery rate (confirmed live: TensorRT
	pushes this well past it). Re-inferring on the same underlying frame more than once
	is wasted GPU work, but harmless -- it does NOT cause duplicate/ghosted skeletons on
	its own; that bug (see draw_tracks() call site in main()) was actually from drawing
	directly onto the shared, reused frame buffer across iterations, layering multiple
	near-identical skeletons on top of each other. The real fix is drawing onto a fresh
	`frame.copy()` each iteration instead of blocking capture -- see main()'s comment."""

	def __init__(self, cap):
		self.cap = cap
		self._lock = threading.Lock()
		self._frame = None
		self._ok = False
		self._stop = False
		ok, frame = cap.read()  # block once so the first main-loop iteration has a frame
		self._ok, self._frame = ok, frame
		self._thread = threading.Thread(target=self._run, daemon=True)
		self._thread.start()

	def _run(self):
		while not self._stop:
			ok, frame = self.cap.read()
			with self._lock:
				self._ok, self._frame = ok, frame

	def read(self):
		with self._lock:
			return self._ok, self._frame

	def stop(self):
		self._stop = True
		self._thread.join(timeout=1.0)


def draw_tracks(frame, tracks, conf_threshold):
	for t in tracks:
		if t.score < conf_threshold or not t.confirmed:
			continue
		box_px = t.payload.get('box_px')
		kpts_px = t.payload.get('keypoints_px')
		if box_px:
			x1, y1, x2, y2 = [int(v) for v in box_px]
			cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 1)
			cv2.putText(frame, f"#{t.track_id}", (x1, max(0, y1 - 6)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)
		if kpts_px:
			for a, b in SKELETON_EDGES:
				xa, ya, ca = kpts_px[a]
				xb, yb, cb = kpts_px[b]
				cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), SKELETON_COLOR, 2, cv2.LINE_AA)
			for x, y, c in kpts_px:
				cv2.circle(frame, (int(x), int(y)), 3, KEYPOINT_COLOR, -1, cv2.LINE_AA)
	return frame


def open_camera(index, width=1280, height=720, target_fps=60):
	"""Opens the camera and tries to negotiate up to target_fps -- most UVC webcams
	default to 30fps until asked for more. cv2 reports whatever the driver actually
	settled on (not necessarily what was requested), so always re-read the property
	after set() rather than trusting the request."""
	cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
	if not cap.isOpened():
		return cap
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # driver hint only -- ThreadedCapture is what actually guarantees freshness

	before_fps = cap.get(cv2.CAP_PROP_FPS)
	cap.set(cv2.CAP_PROP_FPS, target_fps)
	after_fps = cap.get(cv2.CAP_PROP_FPS)
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print(f"[cam] {w}x{h} -- driver-reported FPS: {before_fps:.1f} -> requested {target_fps} -> {after_fps:.1f} "
		f"(some DSHOW drivers report -1/0 regardless; ThreadedCapture's own [perf] eff_fps is the real number)")
	return cap


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--camera', type=int, default=0)
	parser.add_argument('--model', default='yolo26s-pose', choices=[
		'yolo26n-pose', 'yolo26s-pose',
	])
	parser.add_argument('--conf', type=float, default=CONF_THRESHOLD)
	parser.add_argument('--fps', type=int, default=60, help='Requested camera FPS (driver may not honor it).')
	parser.add_argument('--trt', action='store_true', help='Add TensorrtExecutionProvider ahead of CUDA (see make_session docstring for first-run engine-build cost).')
	parser.add_argument('--no-window', action='store_true', help='Run headless, just print timing.')
	args = parser.parse_args()

	model_path = get_model_path(args.model)
	if not os.path.isfile(model_path):
		print(f"[ONNX] Model not found: {model_path}")
		sys.exit(1)
	session = make_session(model_path, use_trt=args.trt)
	input_name = session.get_inputs()[0].name
	output_name = session.get_outputs()[0].name

	tracker = KeypointTracker(
		max_match_dist=MAX_MATCH_DIST, distance_keypoint_indices=DISTANCE_KEYPOINT_INDICES,
		dup_dist_factor=DUP_DIST_FACTOR, track_buffer=TRACKER_MAX_AGE, min_hits=TRACKER_MIN_HITS,
	)

	cap = open_camera(args.camera, target_fps=args.fps)
	if not cap.isOpened():
		print(f"[cam] Could not open camera index {args.camera}")
		sys.exit(1)
	capture = ThreadedCapture(cap)

	def _make_window():
		if hasattr(cv2, 'imshow'):
			return CvPreviewWindow()
		return TkPreviewWindow()

	window = None if args.no_window else _make_window()

	last_log = time.perf_counter()
	frame_count = 0
	sum_pre = sum_infer = sum_post = 0.0

	try:
		while True:
			t0 = time.perf_counter()
			ok, frame = capture.read()
			if not ok or frame is None:
				print("[cam] Frame grab failed, stopping.")
				break
			h, w = frame.shape[:2]

			t1 = time.perf_counter()
			tensor, r, pad_left, pad_top = letterbox(frame)
			t2 = time.perf_counter()

			outputs = session.run([output_name], {input_name: tensor})
			t3 = time.perf_counter()

			detections = postprocess(outputs, r, pad_left, pad_top, w, h, args.conf)
			tracks = tracker.update(detections)
			t4 = time.perf_counter()

			sum_pre += (t2 - t1) * 1000
			sum_infer += (t3 - t2) * 1000
			sum_post += (t4 - t3) * 1000
			frame_count += 1

			if window is not None:
				# Draw onto a COPY, never the shared frame buffer from capture.read()
				# directly -- ThreadedCapture can hand back the SAME underlying frame
				# object across multiple loop iterations when inference outruns the
				# camera's real delivery rate (confirmed live with TensorRT: ~110-120fps
				# vs a ~60fps camera). Drawing in place on that shared buffer meant each
				# extra pass layered another near-identical skeleton on top of the last
				# (tiny GPU-level nondeterminism between passes on identical input made
				# them not quite overlap) -- ghosted/duplicate-looking skeletons. A fresh
				# copy every iteration means every draw starts from clean pixels, so
				# re-inferring on a stale frame is still wasted GPU work, but no longer
				# visually corrupts the output -- this is the fix, not slowing inference
				# down to match the camera.
				display_frame = frame.copy()
				draw_tracks(display_frame, tracks, args.conf)
				total_ms = (t4 - t0) * 1000
				cv2.putText(display_frame, f"{1000.0 / max(total_ms, 1e-3):.1f} fps", (10, 24),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
				window.show(display_frame)
				if window.quit_requested:
					break

			now = time.perf_counter()
			if now - last_log >= 1.0 and frame_count > 0:
				print(f"[perf] frames={frame_count} avg_pre={sum_pre / frame_count:.2f}ms "
					f"avg_infer={sum_infer / frame_count:.2f}ms avg_post={sum_post / frame_count:.2f}ms "
					f"avg_total={(sum_pre + sum_infer + sum_post) / frame_count:.2f}ms "
					f"eff_fps={frame_count / (now - last_log):.1f}")
				last_log = now
				frame_count = 0
				sum_pre = sum_infer = sum_post = 0.0
	finally:
		capture.stop()
		cap.release()
		if window is not None:
			window.close()


if __name__ == '__main__':
	main()
