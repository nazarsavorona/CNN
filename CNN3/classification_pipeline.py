import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchmetrics import Metric
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class Accuracy(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        self.correct += torch.sum(preds == target).cpu()
        self.total += target.size(0)

    def compute(self):
        return self.correct.float() / self.total.float()

    def reset(self):
        self.correct = torch.tensor(0)
        self.total = torch.tensor(0)

class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)  # Replaced view with flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Test loop
def test(model, criterion, test_loader, device='cuda'):
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        accuracy = Accuracy()
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            total_loss += criterion(outputs, targets)

            accuracy.update(outputs, targets)

    return total_loss / len(test_loader), accuracy.compute()

def train(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda', test_loader=None):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        accuracy = Accuracy()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            accuracy.update(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 99:  # Print every 100 batches
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], '
                    f'Loss: {running_loss / 100:.4f}, Accuracy: {100 * accuracy.compute().item():.2f}')

                accuracy.reset()

                writer.add_scalar('Loss/train', running_loss / 100, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Accuracy/train', 100 * accuracy.compute().item(), epoch * len(train_loader) + batch_idx)

                running_loss = 0.0

        test_loss, test_acc = test(model, criterion, test_loader, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {100 * test_acc:.2f}')

        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', 100 * test_acc, epoch)

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = 10
    batch_size = 256
    learning_rate = 0.001

    # Transformations for CIFAR-10 dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4822, 0.4466), (0.2463, 0.2428, 0.2607))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4822, 0.4466), (0.2463, 0.2428, 0.2607)),
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    # model = torchvision.models.resnet18(num_classes=10, weights=None).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, criterion, optimizer, num_epochs, device, test_loader)

writer.flush()