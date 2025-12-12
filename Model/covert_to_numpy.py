import numpy as np
import os
from PIL import Image

# 1. SETUP
index_file = "lfw_index.txt"
output_file = "lfw_data.npz"
target_shape = (105, 105)

if not os.path.exists(index_file):
    print("Error: lfw_index.txt not found!")
    exit()

print("Starting One-Time Conversion...")
print("Reading images and packing them into a single file.")

# 2. LOAD PATHS
image_paths = []
image_classes = [] # The person's name

with open(index_file, "r") as f:
    for line in f:
        parts = line.strip().split("|")
        person_name = parts[0]
        paths = parts[1:]
        
        for p in paths:
            image_paths.append(p)
            image_classes.append(person_name)

print(f"Found {len(image_paths)} total images to process.")

# 3. PROCESS IMAGES
# Initialize huge arrays
# Shape: (Total_Images, 105, 105, 3)
X_data = np.zeros((len(image_paths), 105, 105, 3), dtype='float32')
y_data = np.array(image_classes)

count = 0
bad_indices = []

for i, path in enumerate(image_paths):
    try:
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((target_shape[1], target_shape[0]))
        img_array = np.array(img).astype('float32') / 255.0
        
        X_data[i] = img_array
        
        count += 1
        if count % 200 == 0:
            print(f"Processed {count}/{len(image_paths)} images...")
            
    except Exception as e:
        print(f"Failed to load: {path}")
        bad_indices.append(i)

# 4. CLEAN UP (Remove empty slots if any failed)
if bad_indices:
    print(f"Removing {len(bad_indices)} failed images...")
    X_data = np.delete(X_data, bad_indices, axis=0)
    y_data = np.delete(y_data, bad_indices, axis=0)

# 5. SAVE
print("Saving to 'lfw_data.npz'... This may take a moment.")
np.savez_compressed(output_file, images=X_data, names=y_data)
print("SUCCESS! Data saved. You can now delete the raw image folder if you want.")