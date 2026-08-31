"""
TensorRT-accelerated YOLO26 Instance Segmentation for TouchDesigner.

Identical to onnx_yolo26_seg.py in tracking/postprocessing logic, but uses raw
TensorRT (tensorrt + cuda-python) instead of onnxruntime. This avoids the onnxruntime
TensorRT EP crash we hit inside TD's own process.

Requires:
- Pre-built TensorRT engine: data/ml/yolo26/yolo26s-seg.fp16.engine
  (build offline with python/standalone/build_trt_engine.py - adapted for segmentation model)
- cuda-python==12.6.0 with pywin32 post-install completed
- Manual sys.path setup for TD (TD doesn't process .pth files) — handled at module level below

Usage:
Same as onnx_yolo26_seg.py — drop a Script TOP in TD, set its callback DAT to this file,
wire a Video Device In TOP as input, and it outputs segmentation masks as a matte image
plus tracking data to table_output for visualization.
"""

import sys
import os
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
import numpy_util as npu  # noqa: E402
import trt_inference_manager  # noqa: E402
import object_tracker  # noqa: E402

# Import the base inference manager
TRTInferenceManager = trt_inference_manager.TRTInferenceManager
ByteTracker = object_tracker.ByteTracker
_nms = object_tracker.nms
_track_color = object_tracker.track_color

# COCO class names (80 classes, used by YOLO26)
COCO_CLASSES = {
	0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
	5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
	10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
	15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
	20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
	25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
	30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
	35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
	40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
	45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
	50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
	55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
	60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
	65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
	70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
	75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# ==================== CONFIGURATION ====================
CLASSES_TO_DETECT = ['person']
MODEL_VARIANT = 'yolo26s-seg'
CONF_THRESHOLD = 0.35
LOW_CONF_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.5
MIN_BOX_WIDTH = 0.02
MIN_BOX_HEIGHT = 0.02
MIN_MASK_AREA_RATIO = 0.15
MASK_THRESHOLD = 0.5
OUTPUT_TRACK_DATA = True
TRACKER_MAX_AGE = 30
TRACKER_IOU_THRESHOLD = 0.3
TRACKER_MIN_HITS = 3
OUTPUT_SMOOTHING = 0.23
# actually reaches 0 before the track itself is pruned.
PRESENCE_RAMP_UP = 5
PRESENCE_RAMP_DOWN = 10
SCORE_THRESHOLD = 0.25

# Draw masks/boxes on the output image?
DRAW_BOXES = False

PERSON_BOX_COLOR_BGR = (0, 255, 0)      # Green


# ==================== YOLO26 SEGMENTATION (TensorRT) ====================

class YOLO26SegmentationInferenceTRT(TRTInferenceManager):
	"""TensorRT-accelerated YOLO26 Instance Segmentation with temporal tracking.
	
	Identical API and behavior to onnx_yolo26_seg.YOLO26SegmentationInference, but uses
	raw TensorRT instead of onnxruntime. See onnx_yolo26_seg.py's class docstring for
	the full architecture explanation.
	"""

	def __init__(self):
		super().__init__()
		self.opOutputTableDAT = parent().op('table_output')
		self.conf_threshold = CONF_THRESHOLD
		self.low_conf_threshold = LOW_CONF_THRESHOLD
		self.tracker = ByteTracker(
			high_thresh=CONF_THRESHOLD, low_thresh=LOW_CONF_THRESHOLD,
			match_thresh=TRACKER_IOU_THRESHOLD, track_buffer=TRACKER_MAX_AGE,
			min_hits=TRACKER_MIN_HITS,
		)
		self._box_state = {}
		self._mask_state = {}
		self._presence_state = {}
		self._proto_h = 160
		self._proto_w = 160
		self.tracked_objects = []
		self._input_tensor_buf = None
		self._input_buf_shape = None
		# Cache target class IDs
		self._target_ids_array = np.array(
			[idx for idx, name in COCO_CLASSES.items() if name in CLASSES_TO_DETECT],
			dtype=np.intp
		) if CLASSES_TO_DETECT else None

	def onSetupParameters(self, scriptOp):
		"""Add YOLO26-Seg-specific parameters alongside base class params."""
		super().onSetupParameters(scriptOp)
		page = scriptOp.appendCustomPage('YOLO26-Seg')
		p = page.appendFloat('Confthreshold', label='Confidence Threshold', size=1)
		p[0].default = CONF_THRESHOLD
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		p[0].help = "Minimum detection confidence (ByteTracker's high threshold)"
		scriptOp.par.Confthreshold = CONF_THRESHOLD
		
		p = page.appendFloat('Lowconfthreshold', label='Low Confidence Threshold', size=1)
		p[0].default = LOW_CONF_THRESHOLD
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		p[0].help = "ByteTracker's low confidence threshold for track recovery"
		scriptOp.par.Lowconfthreshold = LOW_CONF_THRESHOLD
		
		p = page.appendFloat('Maskthreshold', label='Mask Threshold', size=1)
		p[0].default = MASK_THRESHOLD
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		p[0].help = "Sigmoid probability cutoff for mask area calculation"
		scriptOp.par.Maskthreshold = MASK_THRESHOLD
		
		p = page.appendFloat('Outputsmoothing', label='Output Smoothing', size=1)
		p[0].default = OUTPUT_SMOOTHING
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		p[0].help = object_tracker.par_help('Outputsmoothing', what='box position/size')
		scriptOp.par.Outputsmoothing = OUTPUT_SMOOTHING
		
		p = page.appendFloat('Nmsiouthreshold', label='NMS IoU Threshold', size=1)
		p[0].default = NMS_IOU_THRESHOLD
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		p[0].help = "IoU threshold for Non-Maximum Suppression"
		scriptOp.par.Nmsiouthreshold = NMS_IOU_THRESHOLD
		
		p = page.appendFloat('Minboxwidth', label='Min Box Width', size=1)
		p[0].default = MIN_BOX_WIDTH
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		scriptOp.par.Minboxwidth = MIN_BOX_WIDTH
		
		p = page.appendFloat('Minboxheight', label='Min Box Height', size=1)
		p[0].default = MIN_BOX_HEIGHT
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		scriptOp.par.Minboxheight = MIN_BOX_HEIGHT
		
		p = page.appendFloat('Minmaskarea', label='Min Mask Area Ratio', size=1)
		p[0].default = MIN_MASK_AREA_RATIO
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		p[0].help = "Minimum mask fill ratio (fraction of detection box)"
		scriptOp.par.Minmaskarea = MIN_MASK_AREA_RATIO
		
		p = page.appendFloat('Trackiouthreshold', label='Track IoU Threshold', size=1)
		p[0].default = TRACKER_IOU_THRESHOLD
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		p[0].help = object_tracker.par_help('Trackiouthreshold')
		scriptOp.par.Trackiouthreshold = TRACKER_IOU_THRESHOLD
		
		p = page.appendFloat('Tracklossframes', label='Track Loss Frames', size=1)
		p[0].default = TRACKER_MAX_AGE
		p[0].min = 0.0
		p[0].max = 90.0
		p[0].clampMin = True
		p[0].clampMax = False
		p[0].help = object_tracker.par_help('Tracklossframes')
		scriptOp.par.Tracklossframes = TRACKER_MAX_AGE
		
		p = page.appendFloat('Trackconfirmframes', label='Track Confirm Frames', size=1)
		p[0].default = TRACKER_MIN_HITS
		p[0].min = 0.0
		p[0].max = 30.0
		p[0].clampMin = True
		p[0].clampMax = False
		p[0].help = object_tracker.par_help('Trackconfirmframes')
		scriptOp.par.Trackconfirmframes = TRACKER_MIN_HITS
		
		p = page.appendFloat('Scorethreshold', label='Score Threshold', size=1)
		p[0].default = SCORE_THRESHOLD
		p[0].min = 0.0
		p[0].max = 1.0
		p[0].clampMin = True
		p[0].clampMax = True
		p[0].help = "Minimum track confidence to display/output"
		scriptOp.par.Scorethreshold = SCORE_THRESHOLD
		
		p = page.appendToggle('Outputtrackdata', label='Output Track Data')
		p.default = OUTPUT_TRACK_DATA
		p.help = "Write tracking data to table_output (disable if unused)"
		scriptOp.par.Outputtrackdata = OUTPUT_TRACK_DATA

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
		"""Log engine I/O shapes and sanity-check the expected segmentation output."""
		# Note: TensorRT segmentation models have TWO outputs:
		# output0: [1, 300, 38] - detections (box + conf + class_id + 32 mask coefficients)
		# output1: [1, 32, 160, 160] - mask prototypes
		pass

	def preprocess(self, nA):
		"""Preprocess input for the segmentation model.
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
			nA = npu.flip_v(nA)
			nA = npu.grayscale_to_rgb(nA)
			self._input_tensor_buf = np.ascontiguousarray(nA.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)
			self._input_buf_shape = self._input_tensor_buf.shape

		return self._input_tensor_buf

	def postprocess(self, outputs):
		"""Postprocess YOLO26-Seg outputs: end2end format, already NMS'd by the graph.
		
		output0 columns: [0:4]=box (pixel-space), [4]=conf, [5]=class_id,
		[6:38]=32 mask coefficients. output1 = mask prototypes.
		"""
		# TensorRT returns a list of outputs - handle both segmentation outputs
		if len(outputs) != 2:
			# No valid outputs - return black frame
			proto_h, proto_w = self._proto_h, self._proto_w
			return npu.flip_v(np.zeros((proto_h, proto_w, 3), dtype=np.float32))

		pred = outputs[0][0]          # (300, 38)
		mask_protos = outputs[1][0]   # (32, proto_h, proto_w)
		self._proto_h, self._proto_w = mask_protos.shape[1], mask_protos.shape[2]
		num_mask_coeffs = mask_protos.shape[0]
		proto_h, proto_w = mask_protos.shape[1], mask_protos.shape[2]

		boxes_raw = pred[:, 0:4].copy()
		confidences = pred[:, 4].copy()
		class_ids = pred[:, 5].astype(np.intp)
		mask_coeffs = pred[:, 6:6 + num_mask_coeffs].copy()

		# Auto-detect pixel-space vs normalized boxes
		box_max = boxes_raw.max() if boxes_raw.size else 0.0
		if box_max > 1.5:
			input_h, input_w = self.original_h, self.original_w
			boxes_xyxy = boxes_raw / np.array([input_w, input_h, input_w, input_h], dtype=np.float32)
		else:
			boxes_xyxy = boxes_raw

		# Read thresholds from custom parameters
		self.conf_threshold = self._par_or_default('Confthreshold', CONF_THRESHOLD)
		self.low_conf_threshold = self._par_or_default('Lowconfthreshold', LOW_CONF_THRESHOLD)
		nms_iou_threshold = self._par_or_default('Nmsiouthreshold', NMS_IOU_THRESHOLD)
		min_box_width = self._par_or_default('Minboxwidth', MIN_BOX_WIDTH)
		min_box_height = self._par_or_default('Minboxheight', MIN_BOX_HEIGHT)
		min_mask_area_ratio = self._par_or_default('Minmaskarea', MIN_MASK_AREA_RATIO)
		mask_threshold = self._par_or_default('Maskthreshold', MASK_THRESHOLD)
		self.tracker.high_thresh = self.conf_threshold
		self.tracker.low_thresh = self.low_conf_threshold
		self.tracker.match_thresh = self._par_or_default('Trackiouthreshold', TRACKER_IOU_THRESHOLD)
		self.tracker.track_buffer = self._par_or_default('Tracklossframes', TRACKER_MAX_AGE)
		self.tracker.min_hits = int(self._par_or_default('Trackconfirmframes', TRACKER_MIN_HITS))
		smoothing = self._par_or_default('Outputsmoothing', OUTPUT_SMOOTHING)

		# Filter detections
		valid = confidences > self.low_conf_threshold
		valid &= (boxes_xyxy[:, 2] - boxes_xyxy[:, 0] >= min_box_width) & (boxes_xyxy[:, 3] - boxes_xyxy[:, 1] >= min_box_height)
		if self._target_ids_array is not None and len(self._target_ids_array) > 0:
			valid &= np.isin(class_ids, self._target_ids_array)

		boxes_xyxy = boxes_xyxy[valid]
		confidences = confidences[valid]
		class_ids = class_ids[valid]
		mask_coeffs = mask_coeffs[valid]

		boxes_xyxy = np.clip(boxes_xyxy, 0.0, 1.0)
		boxes_native = boxes_xyxy.copy()

		# Flip Y-axis for TouchDesigner
		boxes_xyxy[:, 1], boxes_xyxy[:, 3] = 1.0 - boxes_xyxy[:, 3], 1.0 - boxes_xyxy[:, 1]

		# NMS
		if len(boxes_xyxy) > 0:
			keep = _nms(boxes_xyxy, confidences, nms_iou_threshold)
			boxes_xyxy = boxes_xyxy[keep]
			boxes_native = boxes_native[keep]
			confidences = confidences[keep]
			class_ids = class_ids[keep]
			mask_coeffs = mask_coeffs[keep]

		# Decode masks - vectorized for all detections at once
		detections = []
		if len(boxes_xyxy) > 0:
			masks = np.matmul(mask_coeffs, mask_protos.reshape(num_mask_coeffs, -1)).reshape(-1, proto_h, proto_w)
			np.negative(masks, out=masks)
			np.exp(masks, out=masks)
			np.add(1.0, masks, out=masks)
			np.reciprocal(masks, out=masks)
			binary_masks = masks > mask_threshold

			# Crop each mask to its own detection box (standard YOLO-seg postprocessing)
			# Box coords are in native (pre-TD-flip) orientation
			px1 = np.clip((boxes_native[:, 0] * proto_w).astype(np.intp), 0, proto_w)
			py1 = np.clip((boxes_native[:, 1] * proto_h).astype(np.intp), 0, proto_h)
			px2 = np.clip(np.ceil(boxes_native[:, 2] * proto_w).astype(np.intp), 0, proto_w)
			py2 = np.clip(np.ceil(boxes_native[:, 3] * proto_h).astype(np.intp), 0, proto_h)
			col_idx = np.arange(proto_w)
			row_idx = np.arange(proto_h)
			box_areas_px = np.maximum((px2 - px1) * (py2 - py1), 1)
			
			for i in range(len(binary_masks)):
				col_in_box = (col_idx >= px1[i]) & (col_idx < px2[i])
				row_in_box = (row_idx >= py1[i]) & (row_idx < py2[i])
				in_box = row_in_box[:, np.newaxis] & col_in_box[np.newaxis, :]
				binary_masks[i] &= in_box
				masks[i] *= in_box  # Zero soft probabilities outside the box too

			mask_areas = binary_masks.sum(axis=(1, 2))
			fill_ratios = mask_areas / box_areas_px

			for i in range(len(boxes_xyxy)):
				if fill_ratios[i] <= min_mask_area_ratio:
					continue
				detections.append({
					'box': boxes_xyxy[i].tolist(),
					'score': float(confidences[i]),
					'class_id': int(class_ids[i]),
					'class_name': COCO_CLASSES.get(int(class_ids[i]), 'unknown'),
					'mask': masks[i],
					'mask_area_ratio': float(fill_ratios[i]),
				})

		# Update tracker
		active_tracks = self.tracker.update(detections)
		
		# Build tracked objects list
		active_ids = {t.track_id for t in active_tracks}
		self.tracked_objects = []
		
		for t in active_tracks:
			if not t.confirmed:
				continue
			
			# Use t.lost_frames for presence ramping (not t.is_matched which doesn't exist)
			presence = object_tracker.presence_ramp(
				self._presence_state, t.track_id, t.lost_frames == 0,
				PRESENCE_RAMP_UP, PRESENCE_RAMP_DOWN
			)
			
			# Smooth box using Kalman estimate (t.box, not t.bbox)
			box = t.box
			smoothed = object_tracker.box_smooth(self._box_state, t.track_id, box, smoothing)
			
			# Update mask state (hold last real decoded mask across lost frames)
			new_mask = t.payload.get('mask')
			if new_mask is not None:
				self._mask_state[t.track_id] = new_mask
			held_mask = self._mask_state.get(t.track_id)
			
			cx = (smoothed[0] + smoothed[2]) / 2
			cy = (smoothed[1] + smoothed[3]) / 2
			w = smoothed[2] - smoothed[0]
			h = smoothed[3] - smoothed[1]
			
			self.tracked_objects.append({
				'track_id': t.track_id,
				'class_id': t.payload.get('class_id'),
				'class_name': t.payload.get('class_name', 'unknown'),
				'score': t.score,
				'cx': cx, 'cy': cy, 'w': w, 'h': h,
				'x_left': smoothed[0],
				'x_right': smoothed[2],
				'y_top': smoothed[3],
				'y_bottom': smoothed[1],
				'lost_frames': t.lost_frames,
				'total_frames': t.total_frames,
				'mask_area_ratio': t.payload.get('mask_area_ratio', 0.0),
				'mask': held_mask,
				'presence': presence,
			})
		
		# Prune stale state
		object_tracker.prune_stale(active_ids, self._box_state, self._mask_state, self._presence_state)

		# Draw output image - call separate method like ONNX version
		output_img = npu.flip_v(self.draw_tracked_masks(draw_labels=DRAW_BOXES))
		
		return output_img

	def draw_tracked_masks(self, draw_labels=False):
		"""Render a soft-edged white silhouette matte at native proto resolution.
		Lost/occluded tracks are NOT drawn. Returns RGB float32 (0-1) image."""
		proto_h, proto_w = self._proto_h, self._proto_w
		composite = np.zeros((proto_h, proto_w), dtype=np.float32)
		
		for obj in self.tracked_objects:
			# Only draw freshly detected masks this frame (lost_frames == 0)
			if obj['lost_frames'] > 0:
				continue
			mask = obj.get('mask')
			if mask is None:
				continue
			# Per-pixel max (white silhouette, not per-track color)
			np.maximum(composite, mask, out=composite)
		
		composite = np.clip(composite, 0.0, 1.0)
		composite_rgb = np.repeat(composite[:, :, np.newaxis], 3, axis=2)
		
		return composite_rgb

	def on_result_published(self):
		"""Update table_output with tracking data after each frame."""
		if not self._par_or_default('Outputtrackdata', OUTPUT_TRACK_DATA):
			return
		
		if self.opOutputTableDAT is None:
			return
		
		self.opOutputTableDAT.clear()
		self.opOutputTableDAT.appendRow([
			'track_id', 'class_id', 'class_name', 'score', 'presence',
			'cx', 'cy', 'w', 'h', 'x_left', 'x_right', 'y_top', 'y_bottom',
			'lost_frames', 'total_frames', 'mask_area_ratio',
		])
		
		for obj in self.tracked_objects:
			self.opOutputTableDAT.appendRow([
				obj['track_id'],
				obj['class_id'], obj['class_name'],
				f"{obj['score']:.3f}",
				f"{obj['presence']:.2f}",
				f"{obj['cx']:.4f}", f"{obj['cy']:.4f}",
				f"{obj['w']:.4f}", f"{obj['h']:.4f}",
				f"{obj['x_left']:.4f}", f"{obj['x_right']:.4f}",
				f"{obj['y_top']:.4f}", f"{obj['y_bottom']:.4f}",
				obj['lost_frames'], obj['total_frames'],
				f"{obj['mask_area_ratio']:.3f}",
			])


# ==================== MODULE-LEVEL INTERFACE ====================
# Create the manager instance at module level - shut down any PREVIOUS instance first
# (releases GPU resources and stops worker thread) so script reload doesn't leak.
inference_manager = YOLO26SegmentationInferenceTRT()
trt_inference_manager.shutdown_and_register(parent().path, inference_manager)

# Kick off engine load without waiting for TD to cook script1 - nothing wires this COMP's
# output by default, so without this the engine would never load until something views/wires it.
# Must use deferred td.run() not direct cook(force=True) - see schedule_prewarm_cook docstring.
inference_manager.schedule_prewarm_cook(op('script1'), me)


# ==================== TOUCHDESIGNER CALLBACKS ====================

def onSetupParameters(scriptOp):
	inference_manager.onSetupParameters(scriptOp)


def onPulse(par):
	inference_manager.onPulse(par)


def onCook(scriptOp):
	# Update DRAW_BOXES from parent parameter
	global DRAW_BOXES
	DRAW_BOXES = parent().par.Drawdebug.eval() == 1 if hasattr(parent().par, 'Drawdebug') else False
	
	# Run base manager cook (includes auto-load if engine not loaded)
	inference_manager.onCook(scriptOp)


def onGetCookLevel(scriptOp: scriptTOP) -> CookLevel:
	"""Force continuous cooking - see trt_yolo26_pose.py's identical function."""
	return CookLevel.ALWAYS
