import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

class SemanticDetector:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        print("Loading pre-trained Keras MobileNetV2 model...")
        # Load the model, pre-trained on ImageNet
        self.model = MobileNetV2(weights='imagenet')
        self.target_size = (224, 224)
        print("Semantic model loaded successfully.")

    def detect(self, frame):
        """
        Runs the model on the frame and returns a label if confidence > threshold.
        For this MVP, it acts as a frame-level object detector/classifier.
        """
        # Preprocess the frame for MobileNetV2
        # Convert BGR (OpenCV) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized_frame = cv2.resize(rgb_frame, self.target_size)
        
        # Expand dimensions to create a batch of 1
        input_batch = np.expand_dims(resized_frame, axis=0)
        
        # Preprocess input (scaling pixels)
        processed_input = preprocess_input(input_batch)
        
        # Run inference
        preds = self.model.predict(processed_input, verbose=0)
        
        # Decode predictions
        decoded = decode_predictions(preds, top=1)[0]
        
        # top prediction
        _, label, confidence = decoded[0]
        
        if confidence > self.threshold:
            # We assume the object is roughly in the center for this frame-level MVP
            h, w = frame.shape[:2]
            bbox = (int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75))
            return label, confidence, bbox
            
        return None, 0.0, None
