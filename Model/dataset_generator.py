import os
import random
import numpy as np
from PIL import Image

class LFWTripletGenerator:
    def __init__(self, index_file="lfw_index.txt", batch_size=32, target_shape=(105, 105)):
        self.index_file = index_file
        self.batch_size = batch_size
        self.target_shape = target_shape
        
        # Load Index
        self.people_dict = self.load_index()
        self.people_names = list(self.people_dict.keys())

    def load_index(self):
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Index file '{self.index_file}' not found.")

        people = {}
        print("Loading data from index file...")
        with open(self.index_file, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                person_name = parts[0]
                image_paths = parts[1:]
                people[person_name] = image_paths
                
        print(f"Done! Loaded {len(people)} people from index.")
        return people
    
    def preprocess_image(self, image_path):
        """
        Loads image using PIL with DEBUG printing.
        """
        try:
            # DEBUG PRINT: Verify the path looks correct
            # print(f"DEBUG: Loading {image_path}...") 
            
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize((self.target_shape[1], self.target_shape[0]))
            img_array = np.array(img).astype('float32') / 255.0
            return img_array
            
        except Exception as e:
            print(f"\n[ERROR] Failed to load file!")
            print(f"  > Path: {image_path}")
            print(f"  > Error: {e}")
            return None
    
    def get_batch(self):
        h, w = self.target_shape
        anchor = np.zeros((self.batch_size, h, w, 3))
        positive = np.zeros((self.batch_size, h, w, 3))
        negative = np.zeros((self.batch_size, h, w, 3))

        for i in range(self.batch_size):
            fail_count = 0
            while True:
                # Emergency Brake
                if fail_count > 5:
                    raise RuntimeError("Too many image load failures! Check your paths in lfw_index.txt")

                try:
                    # 1. Random Selection
                    anchor_person = random.choice(self.people_names)
                    anchor_path, pos_path = random.sample(self.people_dict[anchor_person], 2)

                    negative_person = random.choice(self.people_names)
                    while negative_person == anchor_person:
                        negative_person = random.choice(self.people_names)
                    neg_path = random.choice(self.people_dict[negative_person])

                    # 2. Load Images
                    anc = self.preprocess_image(anchor_path)
                    pos = self.preprocess_image(pos_path)
                    neg = self.preprocess_image(neg_path)

                    # 3. Check Validity
                    if anc is None or pos is None or neg is None:
                        fail_count += 1
                        continue 

                    anchor[i] = anc
                    positive[i] = pos
                    negative[i] = neg
                    break 

                except Exception as e:
                    print(f"Batch generation error: {e}")
                    fail_count += 1
                    continue

        return [anchor, positive, negative]

if __name__ == "__main__":
    try:
        gen = LFWTripletGenerator(index_file="lfw_index.txt")
        print("Attempting to generate ONE batch...")
        (a, p, n) = gen.get_batch()
        print("\n--- Success! ---")
        print(f"Batch Generated: {a.shape}")
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")