import tensorflow as tf
print("GPU Available: ", len(tf.config.list_physical_devices('GPU')) > 0)