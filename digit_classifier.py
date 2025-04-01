import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)


# Model arch
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        #Calculate the correct input size for the fully connected layer
        #MNIST images are 28x28. After two 2x2 max pooling operations, they become 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Extract base features
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool(x)
        x = self.pool(x)  #Apply pooling twice to get from 28x28 to 7x7

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        # Extract final features
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


#Custom dataset for digit-specific augmentation
class AugmentedMNIST(Dataset):
    def __init__(self, original_dataset, digit, transforms_list):
        self.original_dataset = original_dataset
        self.indices = [i for i, (_, label) in enumerate(original_dataset) if label == digit]
        self.transforms_list = transforms_list
        self.digit = digit

    def __len__(self):
        return len(self.indices) * len(self.transforms_list)

    def __getitem__(self, idx):
        transform_idx = idx % len(self.transforms_list)
        sample_idx = self.indices[idx // len(self.transforms_list)]

        image, label = self.original_dataset[sample_idx]

        #Convert tensor to PIL for augmentations
        from torchvision.transforms import ToPILImage, ToTensor, Normalize
        image_pil = ToPILImage()(image)

        #Apply augmentation
        aug_image = self.transforms_list[transform_idx](image_pil)

        #Convert back to tensor and normalize
        aug_tensor = ToTensor()(aug_image)
        aug_tensor = Normalize((0.1307,), (0.3081,))(aug_tensor)

        return aug_tensor, label


def train_model(epochs=5, batch_size=64, learning_rate=0.001, focus_digits=None):
    #Prepare data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    #Load ds
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    #Apply data augmentation for specific digits if specified
    if focus_digits is not None:
        from torch.utils.data import ConcatDataset
        import torchvision.transforms as T

        #Define augmentation transforms
        transform_list = [
            T.RandomRotation(15),  # Rotate by up to 15 degrees
            T.RandomAffine(0, translate=(0.1, 0.1)),  # Small translations
            T.RandomAffine(0, scale=(0.8, 1.2)),  # Scale variations
            T.RandomAffine(0, shear=10),  # Shear transformations
        ]

        #Create datasets for each digit and augmentation
        augmented_datasets = []
        for digit in focus_digits:
            #Create dataset applying all augmentations to the digit
            aug_dataset = AugmentedMNIST(train_dataset, digit, transform_list)
            augmented_datasets.append(aug_dataset)

        # Combine original and augmented datasets
        combined_dataset = ConcatDataset([train_dataset] + augmented_datasets)
        train_dataset = combined_dataset
        print(f"Added {sum(len(ds) for ds in augmented_datasets)} augmented samples for digits {focus_digits}")

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
        digit_correct = {digit: 0 for digit in range(10)}
        digit_total = {digit: 0 for digit in range(10)}

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                # Track accuracy for each digit
                for digit in range(10):
                    is_digit = target == digit
                    if is_digit.any():
                        digit_correct[digit] += pred[is_digit].eq(target[is_digit].view_as(pred[is_digit])).sum().item()
                        digit_total[digit] += is_digit.sum().item()

        accuracy = 100. * correct / len(test_loader.dataset)
        test_accuracies.append(accuracy)

        #Print per-digit accuracies
        print(f'Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
        for digit in range(10):
            if digit_total[digit] > 0:
                digit_acc = 100. * digit_correct[digit] / digit_total[digit]
                print(f'  Digit {digit} Accuracy: {digit_acc:.2f}%')

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
    #Train with special focus on digit 6 to improve distinction from 9
    model, accuracy = train_model(epochs=5, focus_digits=[6])
    print(f"Final model accuracy: {accuracy:.2f}%")