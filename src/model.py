import torch
import pytorch_lightning as pl

from torch import nn

class AgeGenderModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.5)
        )

        self.fc_age = nn.Linear(73926, 1)  #For age class
        self.fc_gender = nn.Linear(73926, 1)    #For gender class
        
        self.cfg = cfg
        self.criterion_binary= nn.BCELoss()
        self.criterion_regression = nn.MSELoss()
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)

        age = self.fc_age(x)
        gender= torch.sigmoid(self.fc_gender(x))  

        return {'age': age, 'gender': gender}

    def training_step(self, batch, batch_idx):
        image, label_age, label_gender = batch['image'], batch['label_age'], batch['label_gender']
        
        label_hat = self(image)
        label_age_hat = label_hat['age']
        label_gender_hat = label_hat['gender']

        loss_age = self.criterion_regression(label_age_hat, label_age)
        loss_gender = self.criterion_binary(label_gender_hat, label_gender.unsqueeze(-1))
        loss = loss_age + loss_gender

        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_loss_age', loss_age, on_step=False, on_epoch=True)
        self.log('train_loss_gender', loss_gender, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, label_age, label_gender = batch['image'], batch['label_age'], batch['label_gender']
        
        label_hat = self(image)
        label_age_hat = label_hat['age']
        label_gender_hat = label_hat['gender']

        loss_age = self.criterion_regression(label_age_hat, label_age)
        loss_gender = self.criterion_binary(label_gender_hat, label_gender.unsqueeze(-1))
        loss = loss_age + loss_gender

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_loss_age', loss_age, on_step=False, on_epoch=True)
        self.log('val_loss_gender', loss_gender, on_step=False, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        image, label_age, label_gender = batch['image'], batch['label_age'], batch['label_gender']
        
        label_hat = self(image)
        label_age_hat = label_hat['age']
        label_gender_hat = label_hat['gender']

        loss_age = self.criterion_regression(label_age_hat, label_age)
        loss_gender = self.criterion_binary(label_gender_hat, label_gender.unsqueeze(-1))
        loss = loss_age + loss_gender

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_loss_age', loss_age, on_step=False, on_epoch=True)
        self.log('test_loss_gender', loss_gender, on_step=False, on_epoch=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=float(self.cfg["model"]["lr"]), momentum=0.9)

        return optimizer