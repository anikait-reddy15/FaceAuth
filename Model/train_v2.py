import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, Model, metrics

# --- 1. MEMORY FIX (MUST BE FIRST) ---
# This stops the "Allocator" crash by allowing memory to grow slowly
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("SUCCESS: GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)

# --- CONFIGURATION ---
LFW_PATH = r"C:\Projects\FaceAuth\Model\Datasets\lfw-deepfunneled"
BATCH_SIZE = 16       
EPOCHS = 40 
SAVE_PATH = "faceauth_model_128d_v2.h5"

# --- 2. DATASET GENERATOR ---
from dataset_generator import LFWTripletGenerator

# --- 3. THE LIGHTWEIGHT MODEL (Global Average Pooling) ---
def make_embedding_model(input_shape=(100, 100, 3)):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(64, (2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(64, (2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(64, (2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    
    # --- THE MEMORY SAVER ---
    # GlobalAveragePooling reduces 12x12x256 (36k) -> 1x1x256 (256)
    # This reduces VRAM usage by 99%
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense Layer
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    # Output
    outputs = layers.Dense(128, activation='sigmoid')(x)
    return Model(inputs, outputs, name="Embedding")

class SiameseModel(Model):
    def __init__(self, embedding_model, margin=1.0):
        super(SiameseModel, self).__init__()
        self.embedding_model = embedding_model
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.embedding_model(inputs)

    def train_step(self, data):
        triplets, _ = data
        anchors, positives, negatives = triplets[0], triplets[1], triplets[2]

        with tf.GradientTape() as tape:
            anchor_embedding = self.embedding_model(anchors)
            positive_embedding = self.embedding_model(positives)
            negative_embedding = self.embedding_model(negatives)

            ap_distance = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), -1)
            an_distance = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding), -1)

            loss = ap_distance - an_distance + self.margin
            loss = tf.maximum(loss, 0.0)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.embedding_model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.embedding_model.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]

# --- 4. AUGMENTATION & DATA PIPELINE ---
data_gen = LFWTripletGenerator(dataset_path=LFW_PATH, batch_size=BATCH_SIZE, target_shape=(100, 100))

def tf_data_generator():
    while True:
        try:
            X = data_gen.get_batch()
            yield ((X[0], X[1], X[2]), np.zeros((BATCH_SIZE, 1)))
        except Exception as e:
            continue

def augment_data(inputs, labels):
    anchors, positives, negatives = inputs
    
    def apply_aug(img):
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        return tf.clip_by_value(img, 0.0, 1.0)
    
    return (apply_aug(anchors), apply_aug(positives), negatives), labels

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    TOTAL_TRIPLETS = 150000
    STEPS = (TOTAL_TRIPLETS // EPOCHS) // BATCH_SIZE

    # Create Dataset
    dataset = tf.data.Dataset.from_generator(
        tf_data_generator,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 100, 100, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 100, 100, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 100, 100, 3), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    )
    
    dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Build & Train
    print("\n--- STARTING FINAL TRAINING (MemSafe) ---")
    embedding_net = make_embedding_model()
    model = SiameseModel(embedding_net, margin=1.0)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
    
    model.fit(dataset, steps_per_epoch=STEPS, epochs=EPOCHS)
    
    embedding_net.save(SAVE_PATH)
    print("SUCCESS: Saved merged model.")