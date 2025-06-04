import cv2
import numpy as np

def extract_face_and_features(image_path):
    """
    Extract face from image and analyze for inconsistencies
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    face_data = []
    for (x, y, w, h) in faces:
        # Extract face
        face = img[y:y+h, x:x+w]
        
        # Store face information
        face_data.append({
            'face': face,
            'position': (x, y, w, h)
        })
    
    return img, face_data

def analyze_face_inconsistencies(face):
    """
    Analyze facial features for inconsistencies that might indicate manipulation
    Returns a score and dictionary of analysis results
    """
    # Convert to grayscale
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
    # Calculate the noise level (can indicate tampering)
    noise_level = np.std(gray_face)
    
    # Calculate local binary pattern for texture analysis
    def get_lbp(img):
        lbp = np.zeros_like(img)
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                center = img[i, j]
                binary = (img[i-1:i+2, j-1:j+2] >= center).flatten()
                binary[4] = 0  # exclude center
                lbp[i, j] = np.sum(binary * 2**np.arange(8))
        return lbp
    
    lbp = get_lbp(gray_face)
    lbp_std = np.std(lbp)
    
    # Look for color inconsistencies
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_std = np.std(h)
    s_std = np.std(s)
    
    # Combine metrics
    analysis = {
        'noise_level': noise_level,
        'texture_consistency': lbp_std,
        'hue_variation': h_std,
        'saturation_variation': s_std
    }
    
    # Calculate inconsistency score
    score = (noise_level / 50) + (lbp_std / 40) + (h_std / 30) + (s_std / 50)
    
    return score, analysis