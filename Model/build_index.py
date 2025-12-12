import os
import time

# 1. SETUP PATHS
dataset_dir = r"C:\Projects\FaceAuth\Model\Datasets\lfw-deepfunneled"
output_file = "lfw_index1.txt"

print(f"Scanning {dataset_dir}...")
print("Scanning in batches of 500 with cool-down pauses...")

# 2. OPEN OUTPUT FILE
count = 0
with open(output_file, "w") as f:
    # 3. SCAN
    for root, dirs, files in os.walk(dataset_dir):
        
        # Filter for image files
        image_files = [x for x in files if x.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Only save if person has at least 2 images
        if len(image_files) >= 2:
            person_name = os.path.basename(root)
            full_paths = [os.path.join(root, img) for img in image_files]
            
            # Create a single line string separated by pipes "|"
            line = person_name + "|" + "|".join(full_paths)
            f.write(line + "\n")
            
            count += 1
            
            # --- THE BATCH LOGIC ---
            if count % 500 == 0:
                print(f"--> Indexed {count} people. Pausing for 5 seconds to cool down...")
                
                # Force write to disk so we don't lose data if it crashes later
                f.flush()
                os.fsync(f.fileno())
                
                # Sleep to let the OS/Disk catch up
                time.sleep(5) 
                print("--> Resuming...")

print("------------------------------------------------")
print(f"SUCCESS! Index created for {count} people.")
print(f"Saved to {output_file}")