"""Simple import test for debugging."""
import sys
import os

project_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
venv_site = os.path.join(project_folder, '.venv', 'Lib', 'site-packages')
venv_root = os.path.join(project_folder, '.venv')

print(f"[TEST] venv_site in sys.path? {venv_site in sys.path}")

if venv_site not in sys.path:
	sys.path.insert(0, venv_site)
	print(f"[TEST] Added site-packages to sys.path")

# Add win32 subdirectories (pywin32.pth does this for venv python, we do it manually for TD)
win32_paths = [
	os.path.join(venv_site, 'win32'),
	os.path.join(venv_site, 'win32', 'lib'),
	os.path.join(venv_site, 'pythonwin'),
]
for p in win32_paths:
	if p not in sys.path:
		sys.path.insert(0, p)
print(f"[TEST] Added win32 subdirectories to sys.path")

# Add .venv root to DLL search path for pywin32 DLLs (pythoncom311.dll, pywintypes311.dll)
print(f"[TEST] Adding {venv_root} to DLL search path...")
os.add_dll_directory(venv_root)
print(f"[TEST] DLL directory added")

# pywin32.pth runs this import to handle environments where post_install wasn't run
import pywin32_bootstrap
print(f"[TEST] pywin32_bootstrap imported")

print(f"[TEST] First 3 sys.path entries:")
for p in sys.path[:3]:
	print(f"  {p}")

print("[TEST] Attempting: import tensorrt")
try:
	import tensorrt as trt
	print(f"[TEST] SUCCESS - tensorrt {trt.__version__}")
except Exception as e:
	print(f"[TEST] FAILED - {type(e).__name__}: {e}")

print("[TEST] Attempting: from cuda import cudart")
try:
	from cuda import cudart
	print(f"[TEST] SUCCESS - cuda.cudart imported")
except Exception as e:
	print(f"[TEST] FAILED - {type(e).__name__}: {e}")

print("[TEST] Attempting: import win32api")
try:
	import win32api
	print(f"[TEST] SUCCESS - win32api imported")
except Exception as e:
	print(f"[TEST] FAILED - {type(e).__name__}: {e}")
