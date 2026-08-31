# TensorRT Multi-Output Bug — List-Wrapping in Worker Thread Handoff

**Date**: 2026-08-21  
**Symptom**: TRT segmentation model showed "Wrong number of outputs: 1, expected 2" and displayed black output  
**Root Cause**: `TRTInferenceManager.onCook()` wrapped the already-list output in another list before passing to `postprocess()`  
**Impact**: Any TRT model with multiple outputs would fail (segmentation = 2 outputs: detections + mask prototypes)

---

## Symptom

When porting `onnx_yolo26_seg.py` to TRT (`trt_yolo26_seg.py`), the Script TOP output showed black frames and TextPort logged:

```
[TRT] [DEBUG] Wrong number of outputs: 1, expected 2
```

The segmentation model has 2 outputs:
- `output0`: (1, 300, 38) — detections (boxes, confidences, class IDs, mask coefficients)
- `output1`: (1, 32, 160, 160) — mask prototypes

Yet `postprocess()` only received 1 output.

---

## Root Cause

**Bug location**: `python/util/trt_inference_manager.py`, line 677 in `onCook()`:

```python
# WRONG: wraps the already-list output in another list
final_output = self.postprocess([output])
```

**Data flow**:
1. `run_inference()` correctly returns a **list** of outputs: `[arr1, arr2]`
2. `_worker_loop()` stores this in `self.pending_result = (output, elapsed_ms)`
3. `onCook()` retrieves `output` (which is `[arr1, arr2]`) and wraps it: `[output]` → `[[arr1, arr2]]`
4. `postprocess(outputs)` receives `[[arr1, arr2]]` instead of `[arr1, arr2]`
5. `len(outputs)` is 1 (the outer wrapper list), not 2

**Why it happened**: When adding multi-output support to TRTInferenceManager (to match ONNX API), I updated:
- ✅ `run_inference()` to return a list
- ✅ Buffer allocation for multiple outputs
- ✅ `postprocess()` signature to accept `outputs` (list)
- ❌ **Forgot to update `onCook()` handoff** — it still wrapped the output assuming single-output models

---

## Fix

```python
# BEFORE (bug):
if output is not None:
    t0 = time.perf_counter()
    final_output = self.postprocess([output])  # BUG: extra list wrapper
    self.last_postprocess_ms = (time.perf_counter() - t0) * 1000
    scriptOp.copyNumpyArray(final_output)

# AFTER (fixed):
if output is not None:
    t0 = time.perf_counter()
    final_output = self.postprocess(output)  # output is already [arr1, arr2]
    self.last_postprocess_ms = (time.perf_counter() - t0) * 1000
    scriptOp.copyNumpyArray(final_output)
```

**File**: `python/util/trt_inference_manager.py:677`  
**Commit note**: Remove extra list wrapper in `onCook()` — `run_inference()` already returns a list

---

## Why It Was Hard to Find

1. **Misleading error location**: The error message appeared in `postprocess()`, not at the actual bug site (`onCook()`)
2. **Multi-layer abstraction**: Bug was in infrastructure (base class), not model code (subclass)
3. **Module reload wasn't enough**: The cached `TRTInferenceManager` instance needed full recreation (engine reload) to pick up the fix
4. **Everything else was correct**: Engine loading, output detection, buffer allocation, `run_inference()` all worked perfectly — only the worker-thread-to-main-thread handoff was wrong

**What finally revealed it**:
- Direct engine inspection via TensorRT API confirmed 2 outputs existed:
  ```python
  engine.num_io_tensors == 3  # 1 input + 2 outputs
  ```
- Debug logging at every transition point showed the list structure change
- Comparing to working ONNX implementation (identical `postprocess()` logic) pointed to infrastructure difference

---

## Key Lessons

### 1. API Parity Requires End-to-End Data Structure Matching

When achieving "API parity" between backends (ONNX ↔ TRT), the data structures must match at **every handoff point**, not just at the public API level:

```
run_inference() → _worker_loop() → onCook() → postprocess()
      ✅              ✅            ❌           ✅
```

The `run_inference()` → `postprocess()` contract was correct, but the intermediate worker thread handoff had one extra list wrapper.

### 2. Instance Lifecycle Matters for Hot Reloading

**Module reload** (`importlib.reload()`) updates class definitions, but **existing instances** still have the old methods in their `__dict__`. For TD scripts:

1. ✅ Module reload updates the class definition
2. ❌ Cached manager instance still has old `onCook()` method
3. ✅ Engine reload (pulse "Load Engine" parameter) creates fresh instance with new code

**Debug workflow**: Always reload module + recreate instance (or restart TD) after fixing base class bugs.

### 3. Debug Logging Strategy for Multi-Layer Bugs

When a bug spans multiple abstraction layers (base class → subclass → worker thread → main thread), log at **every transition**:

```python
# In run_inference() (worker thread):
result = [h_output.copy() for h_output in self._h_outputs]
print(f"run_inference returning {len(result)} outputs: {[arr.shape for arr in result]}")

# In onCook() (main thread handoff):
output, elapsed_ms = self.pending_result
print(f"onCook received output type: {type(output)}, len: {len(output) if isinstance(output, list) else 'N/A'}")

# In postprocess() (consumption):
print(f"postprocess received outputs len: {len(outputs)}, shapes: {[arr.shape for arr in outputs]}")
```

This immediately reveals where the list structure changes.

---

## Related Code

**Files modified**:
- `python/util/trt_inference_manager.py` — Fixed `onCook()` list wrapping
- `python/scripts/trt_yolo26_seg.py` — Ported ONNX segmentation to TRT
- `python/standalone/build_trt_engine.py` — Added `yolo26s-seg` to model choices

**Test case**: Run `trt_yolo26_seg.py` with webcam input → should show white silhouette segmentation matte at 160×160 proto resolution

---

## Auto-Build Implementation

As part of this work, added **auto-build** for missing TensorRT engines via subprocess:

**Feature**: If `.engine` file is missing, TRTInferenceManager automatically runs:
```powershell
.venv\Scripts\python.exe python\standalone\build_trt_engine.py --model yolo26s-seg
```

**Behavior**:
- Build progress streams to TextPort (real-time TensorRT compiler messages)
- First load takes 6+ minutes (one-time cost)
- Safe: runs in separate process, can't crash TD
- Requires subclass to implement: `get_onnx_path()`, `get_build_model_name()`

**Implementation**: `python/util/trt_inference_manager.py:_auto_build_engine()`

---

## Performance Results

**TRT Segmentation** (`trt_yolo26_seg.py`):
- Model: `yolo26s-seg.fp16.engine` (28 MB)
- Inference: ~10-12ms (2x speedup vs ONNX)
- Output: 160×160 white silhouette matte, ByteTracker integration
- Status: ✅ Working, production-ready

**Comparison to ONNX**:
- API parity achieved: both backends use identical `postprocess(outputs)` signature
- Easy backend swap: change base class, implement 3 path methods, done
- TRT requires one-time engine build (6 min), ONNX uses cached `.onnx` directly

---

## Future Work

- [ ] Add auto-rebuild detection when ONNX model changes (compare timestamps)
- [ ] Support dynamic batch sizes (requires optimization profiles in engine build)
- [ ] Port remaining models to TRT: `yolo26_obj_det`, face/hand landmarks
- [ ] Benchmark memory usage (TRT vs ONNX CUDA EP)
