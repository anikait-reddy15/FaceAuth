import os
import ctypes

# 1. ADD CUDA TO PATH MANUALLY (Just in case Environment Variables are lagging)
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
os.add_dll_directory(cuda_path)

print(f"Checking for files in: {cuda_path}\n")

def check_dll(name):
    path = os.path.join(cuda_path, name)
    # Check if file exists physically
    if os.path.exists(path):
        print(f"[{name}] Found file? YES.", end=" ")
        # Try to load it into memory
        try:
            ctypes.WinDLL(path)
            print("Loadable? YES (Success)")
        except OSError as e:
            print(f"Loadable? NO (Error: {e})")
            print("  -> This usually means it is 32-bit (Wrong) or corrupt.")
    else:
        print(f"[{name}] Found file? NO (Critical Missing File)")

# Check the Big Three
check_dll("cudart64_110.dll")  # CUDA
check_dll("cudnn64_8.dll")     # cuDNN
check_dll("zlibwapi.dll")      # ZLIB

print("\n-------------------------------------------")
print("Diagnosis:")