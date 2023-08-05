import os
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Metric

batch_size = 32
input_size = 28 * 28
num_classes = 10

# Load Data
entire_dataset = MNIST(root="dataset/", train=True, transform=ToTensor(), download=True)
train_ds, val_ds = utils.data.random_split(entire_dataset, [55000, 5000])
test_ds = MNIST(root="dataset/", train=False, transform=ToTensor(), download=True)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss




    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'val_loss': loss, 'val_accuracy': accuracy, 'val_f1': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy, 'test_f1': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y


    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    model = NN(input_size=input_size, num_classes=num_classes)
    trainer = pl.Trainer(limit_train_batches=100, min_epochs=1, max_epochs=10, enable_checkpointing=True)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(model, val_loader)
    trainer.test(model, test_loader)