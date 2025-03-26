import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"On {device}")


class MNISTClassification(nn.Module):
    def __init__(self):
        super(MNISTClassification, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Loading training data
    training_data = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        training_data,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    # Loading testing data
    testing_data = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testing_data,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    return trainloader, testloader


def model_train(model, trainloader, criterion, optimiser, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimiser.zero_grad()

            # Forward pass, backwards pass, and optimisation
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch [{epoch + 1}/{epochs}, Step [{i + 1}], Loss: {running_loss / 200:.4f}')
                running_loss = 0.0
    print("Training complete")


def model_evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int32)

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    accuracy_score = 100 * correct / total
    print(f"Accuracy score: {accuracy_score:.2f}%")

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix.numpy(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    return accuracy_score


def visualise_predictions(model, testloader):
    model.eval()
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            if i >= 1:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for j in range(min(9, inputs.size(0))):
                ax = axes[j // 3, j % 3]
                img = inputs[j].cpu().squeeze().numpy()
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Pred: {predicted[j].item()}, True: {labels[j].item()}')
                ax.axis('off')
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    plt.close()


def main():
    # Ensure data directory exists
    os.makedirs('./data', exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize model
    model = MNISTClassification().to(device)

    # Load data
    trainloader, testloader = load_mnist_data()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    model_train(model, trainloader, criterion, optimiser)

    # Evaluate model
    accuracy = model_evaluate(model, testloader)

    # Visualize predictions
    visualise_predictions(model, testloader)

    # Save model
    torch.save(model.state_dict(), 'mnist_classifier.pth')
    print('Model saved to mnist_classifier.pth')

    # Verify model can be loaded
    try:
        test_model = MNISTClassification().to(device)
        test_model.load_state_dict(torch.load('mnist_classifier.pth'))
        print("Model verification successful!")
    except Exception as e:
        print(f"Model verification failed: {e}")


if __name__ == "__main__":
    main()