import tensorflow as tf
import numpy as np
import cv2
import os

# --- Configuration ---
MODEL_PATH = r"C:\Projects\FaceAuth\faceauth_model_128d_v1.h5"
INPUT_SHAPE = (100, 100) # Must match the input shape used for training
# We use this threshold to decide if two faces are the same or different.
# If Distance < THRESHOLD: SAME PERSON
# If Distance >= THRESHOLD: DIFFERENT PERSON
# For Triplet Loss (margin=0.5), a good starting threshold is 0.7-0.9
DISTANCE_THRESHOLD = 0.15

# --- Helper Functions ---

def load_embedding_model():
    """Loads the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        print("Please ensure faceauth_model_128d.h5 is in the root project folder.")
        exit()
        
    # We must load the custom class to load the model properly
    from Model.model_v1 import SiameseModel
    
    # Load the embedding network which is stored inside the SiameseModel structure
    # Since we saved the 'embedding_net', we can load it directly.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("SUCCESS: Embedding model loaded.")
    return model

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
    Simple dummy face detection. In a real app, you would use a dedicated library
    like OpenCV's Haarcascades or MTCNN here.
    
    For now, we assume the largest object in the center is the face.
    """
    # This is a placeholder: it assumes your face is in the center
    h, w, _ = frame.shape
    
    # Define a central box for the face (e.g., 150x150 pixels in the middle)
    size = min(h, w) // 3
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    x2 = x1 + size
    y2 = y1 + size

    face_roi = frame[y1:y2, x1:x2]
    
    # Return the cropped image and its coordinates for drawing a box
    return face_roi, (x1, y1, x2, y2)

def calculate_distance(emb1, emb2):
    """Calculates the Euclidean distance between two embeddings."""
    return np.sum(np.square(emb1 - emb2))


# --- Main Application Logic ---

def real_time_test():
    """Runs the real-time comparison test."""
    
    embedding_model = load_embedding_model()
    cap = cv2.VideoCapture(0) # 0 is the default webcam

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # Two variables to hold the embeddings of the faces we capture
    anchor_embedding = None
    test_embedding = None
    
    current_mode = "ANCHOR"

    print("\n--- TEST MODE STARTED ---")
    print("Instructions:")
    print("1. Press 'A' to capture the **ANCHOR** face.")
    print("2. Press 'T' to capture the **TEST** face (for comparison).")
    print("3. Press 'Q' to quit.")


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame for a natural feel
        frame = cv2.flip(frame, 1)
        
        # Attempt to detect a face (using placeholder logic)
        face_img, (x1, y1, x2, y2) = detect_face(frame)
        
        # --- Drawing and Display ---
        color = (0, 255, 0) if current_mode == "ANCHOR" else (255, 0, 0)
        
        # Draw the target box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Display the current mode
        cv2.putText(frame, f"MODE: {current_mode}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # --- Distance Display ---
        distance = "N/A"
        result_text = ""
        
        if anchor_embedding is not None and test_embedding is not None:
            distance = calculate_distance(anchor_embedding, test_embedding)
            
            if distance < DISTANCE_THRESHOLD:
                result_text = "MATCH (SUCCESS)"
                result_color = (0, 255, 0)
            else:
                result_text = "NO MATCH (FAILURE)"
                result_color = (0, 0, 255)
            
            cv2.putText(frame, f"Distance: {distance:.4f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, result_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2, cv2.LINE_AA)
        
        
        cv2.imshow('Face Recognition Test (Press A, T, or Q)', frame)
        
        # --- Keyboard Input Handling ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('a'):
            # Capture the Anchor face
            if face_img.size > 0:
                preprocessed_face = preprocess_face(face_img)
                anchor_embedding = embedding_model.predict(preprocessed_face)[0]
                current_mode = "TEST"
                test_embedding = None # Reset test embedding for new comparison
                print(f"Anchor Captured. Distance will update when TEST is captured.")
        
        elif key == ord('t'):
            # Capture the Test face
            if face_img.size > 0:
                preprocessed_face = preprocess_face(face_img)
                test_embedding = embedding_model.predict(preprocessed_face)[0]
                current_mode = "ANCHOR" # Back to anchor mode for easy resetting
                print(f"Test Captured. Distance: {calculate_distance(anchor_embedding, test_embedding):.4f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_test()