import os
from PIL import Image
import time

# 1. Point to a folder you know exists
BASE_DIR = r"C:\Projects\FaceAuth\Model\Datasets\lfw-deepfunneled"

print(f"Checking {BASE_DIR}...")

# 2. Find the first valid image
found_image = None
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(".jpg"):
            found_image = os.path.join(root, file)
            break
    if found_image:
        break

if not found_image:
    print("ERROR: Could not find any .jpg files!")
else:
    print(f"Found image: {found_image}")
    
    # 3. Time how long it takes to open
    print("Attempting to open file...")
    start = time.time()
    try:
        img = Image.open(found_image)
        img.load() # Force actual read from disk
        end = time.time()
        print(f"SUCCESS! Image opened in {end - start:.4f} seconds.")
        print(f"Image Size: {img.size}")
    except Exception as e:
        print(f"FAILED to open image: {e}")