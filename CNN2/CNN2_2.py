import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.norm1 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(64 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 10)

        self.norm_fc = nn.LayerNorm(128)

        # Metrics
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 3)
        x = self.norm1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.norm_fc(x)
        x = F.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        loss = self.__common_step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.__common_step(batch, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.__common_step(batch, mode="test")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def __common_step(self, batch, mode="train"):
        data, targets = batch
        outputs = self(data)
        loss = F.nll_loss(outputs, targets)

        accuracy = self.accuracy(outputs, targets)

        self.log_dict({
            "{}_loss".format(mode): loss,
            "{}_accuracy".format(mode): accuracy
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=11)

    tensorboard_logger = TensorBoardLogger("logs/", name="cifar10")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")

    model = CIFAR10Classifier()

    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[checkpoint_callback],
        logger=tensorboard_logger
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, val_loader)
