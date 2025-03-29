import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)


#Model arch
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # Calculate the correct input size for the fully connected layer
        # MNIST images are 28x28. After two 2x2 max pooling operations, they become 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Add debug prints to track tensor shapes
        # print(f"Input shape: {x.shape}")

        x = self.conv1(x)
        x = F.relu(x)
        # print(f"After conv1: {x.shape}")

        x = self.conv2(x)
        x = F.relu(x)
        # print(f"After conv2: {x.shape}")

        x = self.pool(x)
        # print(f"After first pool: {x.shape}")

        x = self.pool(x)  # Apply pooling twice to get from 28x28 to 7x7
        # print(f"After second pool: {x.shape}")

        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(f"After flatten: {x.shape}")

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_model(epochs=5, batch_size=64, learning_rate=0.001):
    # Prepare data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    #Load ds
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #Init. model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTNet().to(device)

    #Loss func, optimiser
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Training loop
    train_losses = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        #Training phase
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        #Testing
        model.eval()
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        test_accuracies.append(accuracy)

        print(f'Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    #Save model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print(f'Model saved to mnist_model.pth')

    #Plot TL/TA
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    return model, test_accuracies[-1]


if __name__ == "__main__":
    model, accuracy = train_model(epochs=5)
    print(f"Final model accuracy: {accuracy:.2f}%")