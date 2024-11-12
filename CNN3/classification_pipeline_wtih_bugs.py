import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from classification_pipeline import Accuracy


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = self._create_shortcut(in_channels, out_channels, stride)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        x = self.relu(x)

        return x

    def _create_shortcut(self, in_channels, out_channels, stride):
        if in_channels != out_channels or stride != 1:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            return nn.Identity()


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # Reduced initial channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 512, stride=2),
            nn.AdaptiveAvgPool2d((1, 1))  # Add adaptive pooling
        )
        self.classifier = nn.Linear(512, num_classes)  # Adjusted input size

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Training function
def train(model, trainloader, optimizer, criterion, device, accuracy):
    model.train()
    running_loss = 0.0

    log_period = 100
    log_period = 1  # Overfit on one batch

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accuracy.update(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % log_period == (log_period - 1):
            print(f"[{i + 1}] loss: {running_loss / log_period :.4f}, accuracy: {accuracy.compute()}")

            accuracy.reset()

        break  # Overfit on one batch


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':

    # Transformations for CIFAR-10 dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4822, 0.4466), (0.2463, 0.2428, 0.2607)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4822, 0.4466), (0.2463, 0.2428, 0.2607))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

    # Load CIFAR10 train and test datasets
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)  # Overfit on one batch

    # Create the ResNet model
    model = ResNet(num_classes=10)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    accuracy = Accuracy()

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(200):  # Reduce number of epochs
        train(model, trainloader, optimizer, criterion, device, accuracy)
        evaluate(model, testloader, device)
        scheduler.step()
