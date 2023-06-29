import lightning.pytorch as pl

class ModelModule(pl.LightningModule):

    def __init__(self, model, learning_rate=1e-05, loss='mse'):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        
        if loss == 'mse':
            self.loss = torch.nn.MSELoss()
        elif loss == 'l1':
            self.loss = torch.nn.L1Loss()
        elif loss == 'perceptual':
            self.loss = VGGPerceptualLoss()

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
        z = self(x)
        loss = self.loss(z, y)
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
        loss = self.loss(z, y)
        self.log('val_loss', loss)
        
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
        z = self(x)
        loss = self.loss(z, y)
        self.log('test_loss', loss)
        self.log('linear_loss', self.loss(x, y))
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        return optimizer