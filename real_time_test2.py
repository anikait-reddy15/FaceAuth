import tensorflow as tf
import numpy as np
import cv2
import os

# --- Configuration ---
MODEL_PATH = "faceauth_model_128d_v2.h5" 

# Must match the input shape used for training
INPUT_SHAPE = (100, 100) 

# --- THRESHOLD EXPLAINED (Sigmoid Model) ---
# - Same Person distance is usually: 0.0 to 0.3
# - Different Person distance is usually: 0.8 to 2.0+
# Therefore, 0.5 is a perfect "Cutoff" point.
DISTANCE_THRESHOLD = 0.5

# --- Helper Functions ---

def load_embedding_model():
    """Loads the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        print("Did you run train_final.py to generate it?")
        exit()
        
    try:
        # We use compile=False because we don't need the TripletLoss function for testing,
        # we only need the model to generate embeddings.
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("SUCCESS: Model loaded.")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load model. Reason: {e}")
        exit()

def preprocess_face(face_image):
    """Resizes and normalizes the face image for the model."""
    # 1. Resize to 100x100 
    face_image = cv2.resize(face_image, INPUT_SHAPE)
    
    # 2. Convert to float32 and normalize to 0-1
    face_array = np.asarray(face_image).astype('float32') / 255.0
    
    # 3. Expand dimensions to fit model input: (1, 100, 100, 3)
    return np.expand_dims(face_array, axis=0)

def detect_face(frame):
    """
    Simple dummy face detection. 
    Assumes the face is the largest object in the center.
    """
    h, w, _ = frame.shape
    
    # Define a central box for the face (approx 1/3rd of screen)
    size = min(h, w) // 3
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    x2 = x1 + size
    y2 = y1 + size

    face_roi = frame[y1:y2, x1:x2]
    return face_roi, (x1, y1, x2, y2)

def calculate_distance(emb1, emb2):
    """Calculates the Euclidean distance between two embeddings."""
    return np.sum(np.square(emb1 - emb2))

# --- Main Application Logic ---

def real_time_test():
    embedding_model = load_embedding_model()
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    anchor_embedding = None
    test_embedding = None
    current_mode = "ANCHOR"

    print("\n--- TEST MODE STARTED ---")
    print(f"Threshold set to: {DISTANCE_THRESHOLD}")
    print("1. Press 'A' to capture ANCHOR (You).")
    print("2. Press 'T' to capture TEST (To verify).")
    print("3. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Get face from center box
        face_img, (x1, y1, x2, y2) = detect_face(frame)
        
        # --- UI Logic ---
        color = (0, 255, 0) if current_mode == "ANCHOR" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"MODE: {current_mode}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # --- Compare ---
        if anchor_embedding is not None and test_embedding is not None:
            distance = calculate_distance(anchor_embedding, test_embedding)
            
            if distance < DISTANCE_THRESHOLD:
                result_text = "MATCH (UNLOCKED)"
                result_color = (0, 255, 0) # Green
            else:
                result_text = "NO MATCH (LOCKED)"
                result_color = (0, 0, 255) # Red
            
            # Display results
            cv2.putText(frame, f"Diff: {distance:.4f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, result_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)

        # Show Window
        cv2.imshow('FaceAuth Test', frame)
        
        # Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            if face_img.size > 0:
                # Capture Anchor
                processed = preprocess_face(face_img)
                anchor_embedding = embedding_model.predict(processed, verbose=0)[0]
                current_mode = "TEST"
                test_embedding = None
                print("Anchor Captured.")
        elif key == ord('t'):
            if face_img.size > 0:
                # Capture Test
                processed = preprocess_face(face_img)
                test_embedding = embedding_model.predict(processed, verbose=0)[0]
                # Keep mode as TEST so you can keep spamming 'T' to test different angles
                print("Test Captured.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_test()