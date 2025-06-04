from .preprocessing import preprocess_image, detect_deepfake
from .face_analysis import extract_face_and_features, analyze_face_inconsistencies

def deepfake_detection_pipeline(image_path, model=None):
    """
    Complete pipeline that combines model prediction with face inconsistency analysis
    """
    # First, detect faces
    img, face_data = extract_face_and_features(image_path)
    
    results = {
        'faces_detected': len(face_data),
        'faces': [],
        'model_prediction': None,
        'overall_assessment': ''
    }
    
    # If no faces detected, only rely on the model
    if len(face_data) == 0:
        if model is not None:
            prob, is_fake = detect_deepfake(model, image_path)
            results['model_prediction'] = {
                'probability': float(prob),
                'is_fake': bool(is_fake)
            }
            results['overall_assessment'] = f"No faces detected. Model predicts this is {'a deepfake' if is_fake else 'authentic'} with {prob:.2f} confidence."
        else:
            results['overall_assessment'] = "No faces detected and no model provided. Unable to make an assessment."
        return results
    
    # Analyze each face for inconsistencies
    total_inconsistency_score = 0
    for i, face_info in enumerate(face_data):
        face = face_info['face']
        inconsistency_score, analysis = analyze_face_inconsistencies(face)
        
        face_results = {
            'position': face_info['position'],
            'inconsistency_score': float(inconsistency_score),
            'analysis': {k: float(v) for k, v in analysis.items()}
        }
        
        results['faces'].append(face_results)
        total_inconsistency_score += inconsistency_score
    
    # Average inconsistency score
    avg_inconsistency = total_inconsistency_score / len(face_data)
    
    # Combine with model prediction if available
    if model is not None:
        prob, is_fake = detect_deepfake(model, image_path)
        results['model_prediction'] = {
            'probability': float(prob),
            'is_fake': bool(is_fake)
        }
        
        # Combined assessment
        if is_fake and avg_inconsistency > 1.5:
            confidence = "high"
        elif is_fake or avg_inconsistency > 1.5:
            confidence = "moderate"
        else:
            confidence = "low"
        
        results['overall_assessment'] = f"Image is {'likely a deepfake' if is_fake or avg_inconsistency > 1.5 else 'likely authentic'} with {confidence} confidence."
    else:
        # Assessment based only on inconsistency
        if avg_inconsistency > 1.5:
            results['overall_assessment'] = f"Based on facial analysis alone, this image shows signs of manipulation with inconsistency score of {avg_inconsistency:.2f}."
        else:
            results['overall_assessment'] = f"Based on facial analysis alone, this image appears authentic with inconsistency score of {avg_inconsistency:.2f}."
    
    return results