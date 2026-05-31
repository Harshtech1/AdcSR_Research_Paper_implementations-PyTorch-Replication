import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset_adcsr import AdcSRPairedDataset
from models.adcsr_unet import AdcSRPrunedUNet
from models.adcsr_decoder import AdcSRPrunedDecoder
from models.layers import PixelUnshuffleWrapper

class LightningAdcSRStage2Student(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        
        self.pixel_unshuffle = PixelUnshuffleWrapper(downscale_factor=2)
        self.pruned_unet = AdcSRPrunedUNet()
        self.pruned_decoder = AdcSRPrunedDecoder()
        
        self.distill_loss_fn = nn.L1Loss()
        
    def forward(self, lr_img):
        unshuffled = self.pixel_unshuffle(lr_img)
        unet_feats = self.pruned_unet(unshuffled)
        hr_output = self.pruned_decoder(unet_feats)
        return unet_feats, hr_output

    def training_step(self, batch, batch_idx):
        lr_img, hr_img = batch
        student_feats, hr_out = self.forward(lr_img)
        
        with torch.no_grad():
            mock_teacher_feats = torch.randn_like(student_feats)
            
        loss_distill = self.distill_loss_fn(student_feats, mock_teacher_feats)
        loss_adversarial = torch.mean(torch.abs(hr_out - hr_img)) * 0.05
        total_loss = loss_distill + loss_adversarial
        
        self.log("stage2_total_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def main():
    dataset = AdcSRPairedDataset()
    # 4 workers to feed the A100 faster
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    model = LightningAdcSRStage2Student()
    
    trainer = pl.Trainer(
        max_epochs=50, # 50 Full Epochs!
        accelerator="auto",
        devices=1,
        logger=True
    )
    
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main()
