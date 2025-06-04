import cv2
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for the model
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values
    img = img / 255.0
    
    return img

def detect_deepfake(model, image_path, threshold=0.5):
    """
    Detect if an image is a deepfake
    Returns probability and classification
    """
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Get prediction
    prediction = model.predict(img)[0][0]
    
    # Classify based on threshold
    is_fake = prediction >= threshold
    
    return prediction, is_fake