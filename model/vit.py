# Local application
from model.components.cnn_blocks import PeriodicConv2D
from model.components.pos_embed import get_2d_sincos_pos_embed

# Third party
import lightning.pytorch as pl
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_

class VisionTransfomerModule(pl.LightningModule):

    def __init__(self, learning_rate=4.7e-05, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**kwargs)
        self.loss = torch.nn.MSELoss()
        self.learning_rate = learning_rate
        # init a pretrained resnet

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
        """Test step.

        Args:
            batch (tuple): Input and target batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch
        z = self(x)
        loss = torch.nn.MSELoss()(z, y)
        loss_data = torch.nn.MSELoss()(x, y)
        self.log('test_loss_mse_model', loss)
        self.log('test_loss_mse_data', loss_data)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        history,
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, self.in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    proj_drop=drop_rate,
                    attn_drop=drop_rate,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head = nn.Sequential(*self.head)
        self.final = PeriodicConv2D(
            (self.num_patches * embed_dim) // (img_size[0] * img_size[1]),
            self.out_channels,
            kernel_size=7,
            padding=3,
        )
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, patches):
        b, num_patches, embed_dim = patches.shape
        p = self.patch_size
        h, w = self.img_size
        hh, ww = h // p, w // p
        c = (num_patches * embed_dim) // (h * w)
        if hh * ww != patches.shape[1]:
            raise RuntimeError("Cannot unpatchify")
        x = patches.reshape((b, hh, ww, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape((b, -1, h, w))
        return x

    def forward_encoder(self, x: torch.Tensor):
        # x.shape = [B,C,H,W]
        x = self.patch_embed(x)
        # x.shape = [B,num_patches,embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        # x.shape = [B,num_patches,embed_dim]
        x = self.norm(x)
        return x

    def forward(self, x):
        # x.shape = [B,T,in_channels,H,W]
        x = x.unsqueeze(1)
        x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]
        x = self.forward_encoder(x)
        # x.shape = [B,num_patches,embed_dim]
        x = self.head(x)
        # x.shape = [B,num_patches,embed_dim]
        x = self.unpatchify(x)
        # x.shape = [B,(num_patches*embed_dim)//(H*W),H,W]
        preds = self.final(x)
        # preds.shape = [B,out_channels,H,W]
        return preds