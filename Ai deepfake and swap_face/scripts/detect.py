import tensorflow as tf
import argparse
import os
import sys
import json

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import deepfake_detection_pipeline

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect deepfakes in images')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, default='../models/deepfake_detector.h5', help='Path to the model file')
    parser.add_argument('--output', type=str, help='Path to save the results JSON file')
    parser.add_argument('--no_model', action='store_true', help='Use only face analysis without model')
    
    args = parser.parse_args()
    
    # Load the model if not using only face analysis
    model = None
    if not args.no_model:
        try:
            model = tf.keras.models.load_model(args.model)
            print(f"Model loaded from {args.model}")
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Running with face analysis only.")
    
    # Run detection
    print(f"Analyzing image: {args.image}")
    results = deepfake_detection_pipeline(args.image, model)
    
    # Print results
    print("\n--- DETECTION RESULTS ---")
    print(results['overall_assessment'])
    print(f"Faces detected: {results['faces_detected']}")
    
    if model is not None and 'model_prediction' in results:
        print(f"Model confidence: {results['model_prediction']['probability']:.2f}")
    
    # Save results if output path is provided
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    print("\nDetailed analysis:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()