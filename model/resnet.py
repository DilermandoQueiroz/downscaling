import lightning.pytorch as pl
import torch
import torchvision
from torch import nn

class Resnet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        # self.model = torchvision.models.resnet18(pretrained=True)
        # init a pretrained resnet
        backbone = torchvision.models.resnet18(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        size = x.size()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)

        return x
    
    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch (tuple): Input and target batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch
        z = self(x)
        loss = torch.nn.CrossEntropyLoss()(z, y)
        self.log('train_loss', loss)    
        
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
        z = self(x)
        loss = torch.nn.CrossEntropyLoss()(z, y)
        self.log('val_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer
    
