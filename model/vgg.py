import lightning.pytorch as pl
import torch
from torch import nn
import torchvision
    
from collections import OrderedDict
import torchmetrics
import torch
import torch.nn as nn

class VggModule(pl.LightningModule):

    def __init__(self, loss='mse', num_classes=2):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = torchvision.models.vgg16(num_classes=num_classes)
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch (tuple): Input and target batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch
        z = self.model(x)
        y = y.unsqueeze(1).float()
        loss = self.loss(z, y)

        preds = torch.argmax(z, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

        

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch (tuple): Input and target batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch
        z = self.model(x)
        y = y.unsqueeze(1).float()
        loss = self.loss(z, y)

        preds = torch.argmax(z, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return loss

    def test_step(self, batch, batch_idx):
        """test step.

        Args:
            batch (tuple): Input and target batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch
        z = self.model(x)
        y = y.unsqueeze(1).float()
        loss = self.loss(z, y)
        self.log('test_loss', loss)

        preds = torch.argmax(z, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        return optimizer