import torch
from digit_classifier import MNISTClassification

# Try loading the model
try:
    model = MNISTClassification()
    model.load_state_dict(torch.load('mnist_classifier.pth'))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")