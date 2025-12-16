import os
import random
import numpy as np
from PIL import Image

class LFWTripletGenerator:
    def __init__(self, dataset_path, batch_size=32, target_shape=(100, 100)):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.target_shape = target_shape
        
        # This dictionary will hold the ACTUAL IMAGE ARRAYS, not just paths
        # Format: { 'PersonName': [ numpy_array_1, numpy_array_2, ... ] }
        self.data = {}
        self.people_names = []
        
        # Load everything into RAM immediately
        self._load_dataset_to_ram()

    def _load_dataset_to_ram(self):
        print("------------------------------------------------")
        print("LOADING DATASET INTO RAM (This takes 1-2 mins)...")
        print("------------------------------------------------")
        
        # Statistics for progress
        total_people = 0
        total_images = 0
        
        with os.scandir(self.dataset_path) as entries:
            for entry in entries:
                if entry.is_dir():
                    person_name = entry.name
                    person_dir = entry.path
                    
                    person_images = []
                    
                    try:
                        for img_name in os.listdir(person_dir):
                            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(person_dir, img_name)
                                
                                # Process immediately
                                img_array = self._process_image(img_path)
                                if img_array is not None:
                                    person_images.append(img_array)
                                    
                    except OSError:
                        continue

                    # Only keep people with 2+ images
                    if len(person_images) >= 2:
                        self.data[person_name] = person_images
                        total_people += 1
                        total_images += len(person_images)
                        
                        # Print progress every 200 people so you know it's not frozen
                        if total_people % 200 == 0:
                            print(f"Loaded {total_people} classes so far...")

        self.people_names = list(self.data.keys())
        print("------------------------------------------------")
        print(f"DONE! Loaded {total_images} images from {total_people} people.")
        print("Training will now be blindingly fast.")
        print("------------------------------------------------")

    def _process_image(self, image_path):
        """Helper to load and resize images"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.target_shape[1], self.target_shape[0]))
            img_array = np.array(img).astype('float32') / 255.0
            return img_array
        except:
            return None

    def get_batch(self):
        h, w = self.target_shape
        anchors = np.zeros((self.batch_size, h, w, 3))
        positives = np.zeros((self.batch_size, h, w, 3))
        negatives = np.zeros((self.batch_size, h, w, 3))

        for i in range(self.batch_size):
            # 1. Random Anchor
            person = random.choice(self.people_names)
            
            # 2. Grab images FROM RAM (Instant)
            # We don't need 'try/except' here because we filtered bad images during loading
            anc_img, pos_img = random.sample(self.data[person], 2)
            
            # 3. Random Negative
            neg_person = random.choice(self.people_names)
            while neg_person == person:
                neg_person = random.choice(self.people_names)
            
            neg_img = random.choice(self.data[neg_person])
            
            anchors[i] = anc_img
            positives[i] = pos_img
            negatives[i] = neg_img

        return [anchors, positives, negatives]