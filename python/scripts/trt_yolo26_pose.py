"""
TensorRT-accelerated YOLO26 Pose Estimation for TouchDesigner.

Identical to onnx_yolo26_pose.py in tracking/postprocessing logic, but uses raw
TensorRT (tensorrt + cuda-python) instead of onnxruntime. This avoids the onnxruntime
TensorRT EP crash we hit (twice) inside TD's own process — see trt_poc_test.py and
the conversation history for the full investigation.

Requires:
- Pre-built TensorRT engine: data/ml/yolo26/yolo26s-pose.fp16.engine
  (build offline with python/standalone/build_trt_engine.py)
- cuda-python==12.6.0 with pywin32 post-install completed
- Manual sys.path setup for TD (TD doesn't process .pth files) — handled at module level below

Usage:
Same as onnx_yolo26_pose.py — drop a Script TOP in TD, set its callback DAT to this file,
wire a Video Device In TOP as input, and it outputs pose detection results to
table_output/table_joints/table_bones for DebugSkeleton visualization.
"""

import sys
import os
import math
import numpy as np
import cv2

# ==================== TOUCHDESIGNER .VENV SETUP ====================
# TD's embedded Python doesn't process .pth files (like pywin32.pth), so we manually
# add the paths pywin32 needs. This must happen before any cuda-python imports.
_VENV_SITE = os.path.join(project.folder, '.venv', 'Lib', 'site-packages')
_VENV_ROOT = os.path.join(project.folder, '.venv')
if _VENV_SITE not in sys.path:
	sys.path.insert(0, _VENV_SITE)
# Add win32 subdirectories (pywin32.pth does this for venv python, we do it manually for TD)
for _subdir in ['win32', 'win32/lib', 'pythonwin']:
	_p = os.path.join(_VENV_SITE, _subdir)
	if _p not in sys.path:
		sys.path.insert(0, _p)
# Add .venv root to DLL search path for pywin32 DLLs (pythoncom311.dll, pywintypes311.dll)
os.add_dll_directory(_VENV_ROOT)
# pywin32.pth runs this import to handle environments where post_install wasn't run
import pywin32_bootstrap  # noqa: E402

# custom util imports (now safe to import after sys.path is fixed)
import numpy as npu  # noqa: E402
import trt_inference_manager  # noqa: E402
import object_tracker  # noqa: E402

# Import the base inference manager
TRTInferenceManager = trt_inference_manager.TRTInferenceManager
KeypointTracker = object_tracker.KeypointTracker

# ==================== MODEL OUTPUT FORMAT ====================
# See onnx_yolo26_pose.py's identical comment block for the full layout spec.
KEYPOINT_NAMES = [
	'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
	'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
	'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
	'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
NUM_KEYPOINTS = len(KEYPOINT_NAMES)  # 17

SKELETON_EDGES = [
	(0, 1), (0, 2), (1, 3), (2, 4),          # face
	(0, 5), (0, 6), (5, 6),                  # shoulders/neck
	(5, 7), (7, 9), (6, 8), (8, 10),         # arms
	(5, 11), (6, 12), (11, 12),              # torso
	(11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

_UNSTABLE_KEYPOINT_KEYWORDS = ('elbow', 'wrist', 'knee', 'ankle')
DISTANCE_KEYPOINT_INDICES = [
	i for i, name in enumerate(KEYPOINT_NAMES)
	if not any(kw in name for kw in _UNSTABLE_KEYPOINT_KEYWORDS)
]

# ==================== CONFIGURATION ====================
MODEL_VARIANT = 'yolo26s-pose'
CONF_THRESHOLD = 0.005
TRACKER_MAX_AGE = 15
TRACKER_MIN_HITS = 3
MAX_MATCH_DIST = 0.5
DUP_DIST_FACTOR = 0.5
MIN_BOX_WIDTH = 0.02
MIN_BOX_HEIGHT = 0.02
OUTPUT_SMOOTHING = 0.23
DRAW_BOXES = False

PERSON_BOX_COLOR_BGR = (0, 255, 0)      # Green
SKELETON_COLOR_BGR = (0, 255, 255)      # Yellow
KEYPOINT_COLOR_BGR = (0, 128, 255)      # Orange


# ==================== YOLO26 POSE ESTIMATION (TensorRT) ====================

class YOLO26PoseInferenceTRT(TRTInferenceManager):
	"""TensorRT-accelerated YOLO26 Pose Estimation with temporal tracking.
	
	Identical API and behavior to onnx_yolo26_pose.YOLO26PoseInference, but uses raw
	TensorRT instead of onnxruntime. See onnx_yolo26_pose.py's YOLO26PoseInference
	docstring for the full tracking/smoothing architecture explanation.
	"""

	def __init__(self):
		super().__init__()
		self.opOutputTableDAT = parent().op('table_output')
		self.opJointsTableDAT = parent().op('table_joints')
		self.opBonesTableDAT = parent().op('table_bones')
		self.conf_threshold = CONF_THRESHOLD
		self.tracker = KeypointTracker(
			max_match_dist=MAX_MATCH_DIST, distance_keypoint_indices=DISTANCE_KEYPOINT_INDICES,
			dup_dist_factor=DUP_DIST_FACTOR, track_buffer=TRACKER_MAX_AGE,
			min_hits=TRACKER_MIN_HITS,
		)
		self._kpt_state = {}
		self.tracked_objects = []
		self._output_buf = None
		self._output_buf_shape = None
		self._input_tensor_buf = None
		self._input_buf_shape = None

	def onSetupParameters(self, scriptOp):
		"""Add YOLO26-Pose-specific parameters alongside base class params."""
		super().onSetupParameters(scriptOp)
		page = scriptOp.appendCustomPage('YOLO26')
		p: Page = page.appendFloat('Confthreshold', label='Confidence Threshold', size=1)
		p[0].default = CONF_THRESHOLD
		p[0].min = 0.0
		p[0].max = 0.05
		p[0].clampMin = True
		p[0].clampMax = False
		p[0].help = ("Minimum detection confidence for a person to be admitted to the tracker at all.")
		scriptOp.par.Confthreshold = CONF_THRESHOLD
		
		p = page.appendFloat('Outputsmoothing', label='Output Smoothing', size=1)
		p[0].default = OUTPUT_SMOOTHING
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		p[0].help = object_tracker.par_help('Outputsmoothing', what='keypoint position')
		scriptOp.par.Outputsmoothing = OUTPUT_SMOOTHING
		
		p = page.appendFloat('Tracklossframes', label='Track Loss Frames', size=1)
		p[0].default = TRACKER_MAX_AGE
		p[0].min = 0.0
		p[0].max = 90.0
		p[0].clampMin = True
		p[0].clampMax = False
		p[0].help = object_tracker.par_help('Tracklossframes')
		scriptOp.par.Tracklossframes = TRACKER_MAX_AGE
		
		p = page.appendFloat('Maxmatchdist', label='Max Match Distance', size=1)
		p[0].default = MAX_MATCH_DIST
		p[0].min = 0.0
		p[0].max = 2.0
		p[0].clampMin = True
		p[0].clampMax = False
		p[0].help = object_tracker.par_help('Maxmatchdist', subject='person')
		scriptOp.par.Maxmatchdist = MAX_MATCH_DIST
		
		p = page.appendFloat('Trackconfirmframes', label='Track Confirm Frames', size=1)
		p[0].default = TRACKER_MIN_HITS
		p[0].min = 1.0
		p[0].max = 30.0
		p[0].clampMin = True
		p[0].clampMax = False
		p[0].help = object_tracker.par_help('Trackconfirmframes', subject='person')
		scriptOp.par.Trackconfirmframes = TRACKER_MIN_HITS
		
		p = page.appendFloat('Minboxwidth', label='Min Box Width', size=1)
		p[0].default = MIN_BOX_WIDTH
		p[0].min = 0.0
		p[0].max = 0.2
		p[0].clampMin = True
		p[0].clampMax = False
		p[0].help = object_tracker.par_help('Minboxwidth', shape_note=(
			"a standing person is naturally much narrower than tall, so one shared threshold high "
			"enough to reject noise ends up rejecting real, legitimately tall-but-narrow people."
		))
		scriptOp.par.Minboxwidth = MIN_BOX_WIDTH
		
		p = page.appendFloat('Minboxheight', label='Min Box Height', size=1)
		p[0].default = MIN_BOX_HEIGHT
		p[0].min = 0.0
		p[0].max = 0.2
		p[0].clampMin = True
		p[0].clampMax = False
		p[0].help = object_tracker.par_help('Minboxheight')
		scriptOp.par.Minboxheight = MIN_BOX_HEIGHT

	def get_engine_path(self):
		"""Return path to the pre-built TensorRT engine."""
		model_dir = os.path.join(project.folder, 'data', 'ml', 'yolo26')
		return os.path.join(model_dir, f'{MODEL_VARIANT}.fp16.engine')

	def get_onnx_path(self):
		"""Return path to source ONNX file for auto-build."""
		model_dir = os.path.join(project.folder, 'data', 'ml', 'yolo26')
		return os.path.join(model_dir, f'{MODEL_VARIANT}.onnx')

	def get_build_model_name(self):
		"""Return --model argument for build_trt_engine.py."""
		return MODEL_VARIANT

	def on_engine_loaded(self, engine, context):
		"""Log engine I/O shapes and sanity-check the expected end2end pose output."""
		self.printTRT(f"[TRT] Engine loaded: {self.get_engine_path()}")
		self.printTRT(f"[TRT] Num I/O tensors: {engine.num_io_tensors}")
		for i in range(engine.num_io_tensors):
			name = engine.get_tensor_name(i)
			shape = engine.get_tensor_shape(name)
			dtype = engine.get_tensor_dtype(name)
			mode = engine.get_tensor_mode(name)
			self.printTRT(f"[TRT]   [{i}] {name}: shape={tuple(shape)} dtype={dtype} mode={mode}")

		expected_width = 4 + 1 + 1 + NUM_KEYPOINTS * 3  # box + conf + class_id + keypoints
		output_shape = engine.get_tensor_shape(engine.get_tensor_name(1))
		if output_shape[-1] != expected_width:
			self.printTRT(f"[TRT] WARNING: expected last output dim {expected_width}, got {output_shape}")

	def preprocess(self, nA):
		"""Preprocess input for the pose model.
		Assumes TD has already resized input to 640x640."""
		self.original_h, self.original_w = nA.shape[:2]
		num_channels = nA.shape[2] if len(nA.shape) == 3 else 1

		if num_channels >= 3:
			h, w = self.original_h, self.original_w
			needed = (1, 3, h, w)
			if self._input_buf_shape != needed:
				self._input_tensor_buf = np.empty(needed, dtype=np.float32)
				self._input_buf_shape = needed
			flipped = nA[::-1, :, :3]
			self._input_tensor_buf[0, 0] = flipped[:, :, 0]
			self._input_tensor_buf[0, 1] = flipped[:, :, 1]
			self._input_tensor_buf[0, 2] = flipped[:, :, 2]
		else:
			nA = self.npu.flip_v(nA)
			nA = self.npu.grayscale_to_rgb(nA)
			self._input_tensor_buf = np.ascontiguousarray(nA.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)
			self._input_buf_shape = self._input_tensor_buf.shape

		return self._input_tensor_buf

	def postprocess(self, outputs):
		"""Postprocess YOLO26 pose outputs: end2end format, already NMS'd by the graph.
		
		pred columns: [0:4]=box xyxy (0-1), [4]=conf, [5]=class_id (always 0),
		[6:57]=17 keypoints x (x, y, conf), all normalized 0-1.
		"""
		pred = outputs[0][0]  # (300, 57)

		boxes_xyxy = pred[:, 0:4].copy()
		confidences = pred[:, 4].copy()
		keypoints = pred[:, 6:6 + NUM_KEYPOINTS * 3].reshape(-1, NUM_KEYPOINTS, 3).copy()

		# Read thresholds/smoothing from custom parameters
		self.conf_threshold = self._par_or_default('Confthreshold', CONF_THRESHOLD)
		smoothing = self._par_or_default('Outputsmoothing', OUTPUT_SMOOTHING)
		self.tracker.max_match_dist = self._par_or_default('Maxmatchdist', MAX_MATCH_DIST)
		self.tracker.track_buffer = self._par_or_default('Tracklossframes', TRACKER_MAX_AGE)
		self.tracker.min_hits = int(self._par_or_default('Trackconfirmframes', TRACKER_MIN_HITS))
		min_box_width = self._par_or_default('Minboxwidth', MIN_BOX_WIDTH)
		min_box_height = self._par_or_default('Minboxheight', MIN_BOX_HEIGHT)

		valid = confidences > self.conf_threshold
		valid &= (boxes_xyxy[:, 2] - boxes_xyxy[:, 0] >= min_box_width) & (boxes_xyxy[:, 3] - boxes_xyxy[:, 1] >= min_box_height)
		boxes_xyxy = boxes_xyxy[valid]
		confidences = confidences[valid]
		keypoints = keypoints[valid]

		# Clip boxes to [0, 1]
		boxes_xyxy = np.clip(boxes_xyxy, 0.0, 1.0)

		# Flip Y-axis for TouchDesigner (model uses top-down, TD uses bottom-up)
		boxes_xyxy[:, 1], boxes_xyxy[:, 3] = 1.0 - boxes_xyxy[:, 3], 1.0 - boxes_xyxy[:, 1]
		keypoints[:, :, 1] = 1.0 - keypoints[:, :, 1]

		# Build detection list for the tracker
		detections = []
		for i in range(len(boxes_xyxy)):
			detections.append({
				'box': boxes_xyxy[i].tolist(),
				'score': float(confidences[i]),
				'keypoints': keypoints[i].tolist(),
			})

		# Update tracker
		active_tracks = self.tracker.update(detections)

		# Build structured data for output
		active_ids = {t.track_id for t in active_tracks}
		self.tracked_objects = []
		for t in active_tracks:
			if t.score < self.conf_threshold or not t.confirmed:
				continue
			
			box_raw = t.box
			state = self._kpt_state.get(t.track_id)
			raw_kpts = t.payload.get('keypoints')
			
			if state is None:
				state = {
					'smoothed': [list(kp) for kp in raw_kpts] if raw_kpts else [[0.0, 0.0, 0.0]] * NUM_KEYPOINTS,
					'lost_frames': [0] * NUM_KEYPOINTS,
					'box': list(box_raw),
				}
				self._kpt_state[t.track_id] = state
			elif t.lost_frames == 0 and raw_kpts is not None:
				for k, (x, y, conf) in enumerate(raw_kpts):
					sx, sy, _ = state['smoothed'][k]
					state['smoothed'][k] = [
						sx * smoothing + x * (1.0 - smoothing),
						sy * smoothing + y * (1.0 - smoothing),
						conf,
					]
					state['lost_frames'][k] = 0
				state['box'] = [
					state['box'][i] * smoothing + box_raw[i] * (1.0 - smoothing)
					for i in range(4)
				]
			if t.lost_frames > 0:
				for k in range(NUM_KEYPOINTS):
					state['lost_frames'][k] += 1

			box = state['box']
			cx = (box[0] + box[2]) / 2
			cy = (box[1] + box[3]) / 2
			w = box[2] - box[0]
			h = box[3] - box[1]

			self.tracked_objects.append({
				'track_id': t.track_id,
				'score': t.score,
				'cx': cx, 'cy': cy, 'w': w, 'h': h,
				'x_left': box[0],
				'x_right': box[2],
				'y_top': box[3],
				'y_bottom': box[1],
				'keypoints': state['smoothed'],
				'keypoints_visible': [lost <= self.tracker.track_buffer for lost in state['lost_frames']],
				'vx': float(t.mean[4]), 'vy': float(t.mean[5]),
				'lost_frames': t.lost_frames,
				'total_frames': t.total_frames,
			})

		# Prune keypoint smoothing state for dropped tracks
		object_tracker.prune_stale(active_ids, self._kpt_state)

		# Draw output image
		if DRAW_BOXES:
			output_img = self.npu.flip_v(self.draw_tracked_skeletons())
		else:
			needed_shape = (self.original_h, self.original_w, 3)
			if self._output_buf is None or self._output_buf_shape != needed_shape:
				self._output_buf = np.zeros(needed_shape, dtype=np.float32)
				self._output_buf_shape = needed_shape
			output_img = self._output_buf

		return output_img

	def on_result_published(self):
		"""Flush tables after this frame's texture publishes."""
		self.write_tracks_to_table()
		self.write_joints_bones_to_tables()

	def draw_tracked_skeletons(self):
		"""Render boxes + skeletons for tracked people onto a blank image."""
		output_img = np.zeros((self.original_h, self.original_w, 3), dtype=np.float32)

		if not self.tracked_objects:
			return output_img

		draw_img = np.zeros((self.original_h, self.original_w, 3), dtype=np.uint8)
		w, h = self.original_w, self.original_h

		def to_px(td_x, td_y):
			return object_tracker.td_to_px(td_x, td_y, w, h)

		for obj in self.tracked_objects:
			if obj['lost_frames'] > 0 and obj['score'] < self.conf_threshold * 0.5:
				continue

			fade = object_tracker.track_fade(obj['lost_frames'], self.tracker.track_buffer)

			box_color = tuple(int(c * fade) for c in PERSON_BOX_COLOR_BGR)
			kpt_color = tuple(int(c * fade) for c in KEYPOINT_COLOR_BGR)
			skel_color = tuple(int(c * fade) for c in SKELETON_COLOR_BGR)

			px1, py_bottom = to_px(obj['x_left'], obj['y_bottom'])
			px2, py_top = to_px(obj['x_right'], obj['y_top'])
			cv2.rectangle(draw_img, (px1, py_top), (px2, py_bottom), box_color, 2)

			label = f"#{obj['track_id']} {obj['score']:.0%}"
			font_scale = 0.5
			(tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
			cv2.rectangle(draw_img, (px1, py_top - th - 6), (px1 + tw + 4, py_top), box_color, -1)
			cv2.putText(draw_img, label, (px1 + 2, py_top - 4),
				cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

			kpts = obj['keypoints']
			visible = obj['keypoints_visible']
			pts_px = [to_px(kp[0], kp[1]) if vis else None for kp, vis in zip(kpts, visible)]

			for a, b in SKELETON_EDGES:
				if pts_px[a] is not None and pts_px[b] is not None:
					cv2.line(draw_img, pts_px[a], pts_px[b], skel_color, 2, cv2.LINE_AA)

			for pt in pts_px:
				if pt is not None:
					cv2.circle(draw_img, pt, 3, kpt_color, -1, cv2.LINE_AA)

		return cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

	def write_tracks_to_table(self):
		"""Write current tracking data to a Table DAT."""
		tbl = self.opOutputTableDAT
		if tbl is None:
			return

		kpt_header = []
		for name in KEYPOINT_NAMES:
			kpt_header += [f'{name}_x', f'{name}_y', f'{name}_conf']

		tbl.clear()
		tbl.appendRow([
			*object_tracker.label_header(),
			*object_tracker.box_header(),
			*kpt_header,
			*object_tracker.color_header(),
		])
		for obj in self.tracked_objects:
			flat_kpts = [v for kp in obj['keypoints'] for v in kp]
			tbl.appendRow([
				*object_tracker.label_row(obj['track_id'], obj['score']),
				*object_tracker.box_row(obj),
				*[f"{v:.4f}" for v in flat_kpts],
				*object_tracker.color_row(obj['track_id']),
			])

	def write_joints_bones_to_tables(self):
		"""Write flattened per-visible-keypoint / per-visible-bone Table DATs."""
		if self.opJointsTableDAT is not None:
			tbl = self.opJointsTableDAT
			tbl.clear()
			tbl.appendRow(object_tracker.joints_header())
			for obj in self.tracked_objects:
				track_id = obj['track_id']
				for name, kp, vis in zip(KEYPOINT_NAMES, obj['keypoints'], obj['keypoints_visible']):
					if vis:
						tbl.appendRow(object_tracker.joints_row(track_id, name, kp[0], kp[1], 0.0, kp[2]))

		if self.opBonesTableDAT is not None:
			tbl = self.opBonesTableDAT
			tbl.clear()
			tbl.appendRow(object_tracker.bones_header())
			for obj in self.tracked_objects:
				track_id = obj['track_id']
				kpts = obj['keypoints']
				vis = obj['keypoints_visible']
				for a, b in SKELETON_EDGES:
					if vis[a] and vis[b]:
						ax, ay, aconf = kpts[a]
						bx2, by2, bconf2 = kpts[b]
						dx = bx2 - ax
						dy = by2 - ay
						mx = (ax + bx2) / 2.0
						my = (ay + by2) / 2.0
						angle = math.degrees(math.atan2(dy, dx))
						length = math.hypot(dx, dy)
						conf = min(aconf, bconf2)
						tbl.appendRow(object_tracker.bones_row(track_id, mx, my, angle, length, conf))


# Create global instance
inference_manager = YOLO26PoseInferenceTRT()
trt_inference_manager.shutdown_and_register(parent().path, inference_manager)

# Kick off engine loading without waiting for downstream wiring - nothing wires this COMP's
# output by default, so script1 would never cook until something visits/wires it.
# See TRTInferenceManager.schedule_prewarm_cook() docstring for why this uses deferred td.run().
try:
	inference_manager.schedule_prewarm_cook(op('script1'), me)
except Exception:
		pass  # Prewarm scheduling is optional, don't block module load

# TouchDesigner callback wrappers
def onSetupParameters(scriptOp):
	return inference_manager.onSetupParameters(scriptOp)


def onPulse(par):
	return inference_manager.onPulse(par)


def onCook(scriptOp):
	inference_manager.onCook(scriptOp)
	global DRAW_BOXES
	DRAW_BOXES = parent().par.Drawdebug.eval() == 1


def onGetCookLevel(scriptOp: scriptTOP) -> CookLevel:
	"""
	Sets the scriptOp's cook level, the conditions necessary to cause a cook.

	Return one of the following:
		CookLevel.AUTOMATIC - inputs changed and output being used. TD default behavior.
		CookLevel.ON_CHANGE - inputs changed, output used or not.
		CookLevel.WHEN_USED - every frame when output is being used
		CookLevel.ALWAYS - every frame

	AUTOMATIC alone can't drive this pipeline reliably: anything reading inference_manager
	via a raw Python module reference (not a wire/parameter) is invisible to TD's "is the
	output being used" dependency check, so AUTOMATIC can stop cooking this even while
	something downstream still depends on it. Worse, once AUTOMATIC settles into "not
	cooking" nothing prompts it to re-check later -- resuming play isn't a registered
	dependency of this op, so it never recovers on its own. Always returning ALWAYS keeps
	this op eligible to cook every frame; the play/pause skip instead lives in
	TRTInferenceManager.onCook() itself (checks scriptOp.time.play and returns early), so
	the very next real cook after resuming naturally picks back up.
	"""
	return CookLevel.ALWAYS
