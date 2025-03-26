import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import seaborn as sns

from digit_classifier import MNISTClassification


def load_mnist_test_data():
    # Define transforms (same as in training script)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load test data
    testset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False
    )

    return testloader


def detailed_model_evaluation(model, testloader):
    # Set model to evaluation mode
    model.eval()

    # Prepare lists to store predictions and true labels
    all_preds = []
    all_labels = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for inputs, labels in testloader:
            # Move inputs and labels to the same device as the model
            device = next(model.parameters()).device
            inputs, labels = inputs.to(device), labels.to(device)

            # Get model predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute key metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )

    # Generate classification report
    class_report = classification_report(all_labels, all_preds)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def main():
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and load the model
    model = MNISTClassification().to(device)
    model.load_state_dict(torch.load('mnist_classifier.pth', map_location=device))

    # Load test data
    testloader = load_mnist_test_data()

    # Perform detailed evaluation
    evaluation_results = detailed_model_evaluation(model, testloader)

    # Print evaluation metrics
    print("Model Evaluation Results:")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"Precision: {evaluation_results['precision']:.4f}")
    print(f"Recall: {evaluation_results['recall']:.4f}")
    print(f"F1 Score: {evaluation_results['f1_score']:.4f}")

    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(evaluation_results['classification_report'])

    # Plot confusion matrix
    plot_confusion_matrix(evaluation_results['confusion_matrix'])

    print("\nConfusion matrix has been saved to 'confusion_matrix.png'")


if __name__ == '__main__':
    main()