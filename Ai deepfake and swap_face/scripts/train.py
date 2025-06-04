import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import build_deepfake_detector
from src.evaluation import plot_training_history, evaluate_model

def train_model(model, train_dir, validation_dir, batch_size=32, epochs=10):
    """
    Train the deepfake detection model
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data should only be rescaled
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow from directory
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    
    return model, history

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--validation_dir', type=str, required=True, help='Path to validation data directory')
    parser.add_argument('--test_dir', type=str, help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--output_model', type=str, default='../models/deepfake_detector.h5', help='Path to save the model')
    
    args = parser.parse_args()
    
    # Build the model
    model = build_deepfake_detector()
    print("Model built successfully.")
    
    # Train the model
    print(f"Training model with {args.epochs} epochs and batch size {args.batch_size}...")
    model, history = train_model(model, args.train_dir, args.validation_dir, 
                                 batch_size=args.batch_size, epochs=args.epochs)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set if provided
    if args.test_dir:
        print("\nEvaluating on test data...")
        evaluate_model(model, args.test_dir, batch_size=args.batch_size)
    
    # Save the model
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model)
    print(f"Model saved to {args.output_model}")

if __name__ == "__main__":
    main()