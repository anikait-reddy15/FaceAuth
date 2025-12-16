import tensorflow as tf
from tensorflow.keras import layers, Model, metrics

def make_embedding_model(input_shape=(100, 100, 3)):
    inputs = layers.Input(shape=input_shape)

    # --- Block 1 ---
    x = layers.Conv2D(64, (10, 10), padding='same')(inputs)
    x = layers.BatchNormalization()(x)  
    x = layers.Activation('relu')(x)    # Activation comes after BN usually
    x = layers.MaxPooling2D(64, (2, 2), padding='same')(x)

    # --- Block 2 ---
    x = layers.Conv2D(128, (7, 7), padding='same')(x)
    x = layers.BatchNormalization()(x)  
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(64, (2, 2), padding='same')(x)

    # --- Block 3 ---
    x = layers.Conv2D(128, (4, 4), padding='same')(x)
    x = layers.BatchNormalization()(x)  
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(64, (2, 2), padding='same')(x)

    # --- Block 4 ---
    x = layers.Conv2D(256, (4, 4), padding='same')(x)
    x = layers.BatchNormalization()(x)  
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    
    # --- Optimization ---
    # We use BatchNormalization first to organize the data
    x = layers.BatchNormalization()(x)
    
    # Then we use Dropout (0.4 is often the sweet spot)
    # This prevents the final dense layer from memorizing specific faces
    x = layers.Dropout(0.4)(x)
    
    # --- Final Embedding ---
    outputs = layers.Dense(128, activation=None)(x)
    
    return Model(inputs, outputs, name="Embedding")

class SiameseModel(Model):
    # This part stays exactly the same as before
    def __init__(self, embedding_model, margin=0.5):
        super(SiameseModel, self).__init__()
        self.embedding_model = embedding_model
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.embedding_model(inputs)

    def train_step(self, data):
        triplets, _ = data
        anchors = triplets[0]
        positives = triplets[1]
        negatives = triplets[2]

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