import tensorflow as tf
import numpy as np
import os

# Import your custom modules
from dataset_generator import LFWTripletGenerator
from model import make_embedding_model, SiameseModel

# --- CONFIGURATION ---
LFW_PATH = r"C:\Projects\FaceAuth\Model\Datasets\lfw-deepfunneled"
BATCH_SIZE = 32
TOTAL_TRIPLETS_REQUIRED = 150000
EPOCHS = 50 

# Steps calculation
STEPS_PER_EPOCH = (TOTAL_TRIPLETS_REQUIRED // EPOCHS) // BATCH_SIZE

print(f"Plan: Train on {TOTAL_TRIPLETS_REQUIRED} triplets.")
print(f"Execution: {EPOCHS} Epochs x {STEPS_PER_EPOCH} Steps x {BATCH_SIZE} Batch Size")

# 1. SETUP GENERATOR
data_gen = LFWTripletGenerator(dataset_path=LFW_PATH, batch_size=BATCH_SIZE, target_shape=(100, 100))

def tf_data_generator():
    """
    Wraps the generator. 
    Yields a TUPLE of inputs ((anc, pos, neg), label)
    """
    while True:
        try:
            # X is a list [anchors, positives, negatives]
            X = data_gen.get_batch()
            
            # We explicitly unpack it into a tuple
            inputs = (X[0], X[1], X[2])
            
            # Dummy labels
            dummy_y = np.zeros((BATCH_SIZE, 1))
            
            yield (inputs, dummy_y)
        except Exception as e:
            print(f"Skipping bad batch: {e}")
            continue

# 2. CREATE TENSORFLOW DATASET (The Fix for TypeError)
# We define the shape structure explicitly so TF doesn't have to guess
train_dataset = tf.data.Dataset.from_generator(
    tf_data_generator,
    output_signature=(
        (
            tf.TensorSpec(shape=(None, 100, 100, 3), dtype=tf.float32), # Anchor
            tf.TensorSpec(shape=(None, 100, 100, 3), dtype=tf.float32), # Positive
            tf.TensorSpec(shape=(None, 100, 100, 3), dtype=tf.float32), # Negative
        ),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32) # Label
    )
)

# 3. BUILD MODEL
print("Building Siamese Network...")
embedding_net = make_embedding_model(input_shape=(100, 100, 3))
siamese_model = SiameseModel(embedding_net, margin=0.5)

siamese_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

# 4. START TRAINING
print("Starting Training Loop...")
# Note: We pass the 'train_dataset' object here, not the function
siamese_model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS
)

# 5. SAVE
embedding_net.save("faceauth_model_128d.h5")
print("-----------------------------------")
print("SUCCESS! Model trained and saved.")
print("-----------------------------------")