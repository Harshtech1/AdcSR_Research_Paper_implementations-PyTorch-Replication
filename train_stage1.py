import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_adcsr import AdcSRPairedDataset
from models.adcsr_decoder import AdcSRPrunedDecoder

class LightningStage1DecoderPretrainer(pl.LightningModule):
    def __init__(self, lr=1.3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.decoder = AdcSRPrunedDecoder()
        self.pixel_criterion = nn.L1Loss()

    def forward(self, x):
        return self.decoder(x)

    def training_step(self, batch, batch_idx):
        lr_img, hr_img = batch
        
        # CHANGED: The UNet outputs 64x64 after PixelUnshuffle(2) reduces 128x128
        mock_unet_features = torch.randn(lr_img.shape[0], 240, 64, 64, device=self.device)
        
        reconstructed_hr = self.forward(mock_unet_features)
        loss = self.pixel_criterion(reconstructed_hr, hr_img)
        
        self.log("stage1_l1_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    dataset = AdcSRPairedDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    model = LightningStage1DecoderPretrainer()
    
    trainer = pl.Trainer(
        max_epochs=50, 
        accelerator="auto", 
        devices=1,
        logger=True
    )
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
