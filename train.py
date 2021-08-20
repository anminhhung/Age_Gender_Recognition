import pytorch_lightning as pl

from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader 
from pytorch_lightning.callbacks import ModelCheckpoint

from configs.init import init_config
from src.custom_transform import RGBToTensor
from src.dataset import AgeGenderData
from src.model import AgeGenderModel

if __name__ == "__main__":
    cfg = init_config()
    rgb_to_tensor = RGBToTensor()
    
    # dataloader
    transformed_train_data = AgeGenderData(path_dir=cfg["dataset"]["data_path"], \
                                    train_val_ratio=cfg["dataset"]["train_val_ratio"], train=True, \
                                    transform=transforms.Compose([RGBToTensor()]))

    transformed_test_data = AgeGenderData(path_dir=cfg["dataset"]["data_path"], \
                                    train_val_ratio=cfg["dataset"]["train_val_ratio"], train=False, \
                                    transform=transforms.Compose([RGBToTensor()]))
    
    train_dataloader = DataLoader(transformed_train_data, batch_size=cfg["model"]["batch_size"], shuffle=True)
    test_dataloader = DataLoader(transformed_test_data, batch_size=cfg["model"]["batch_size"], shuffle=True)

    checkpoint_callback = ModelCheckpoint(
        monitor = cfg["callback"]["monitor"],
        dirpath = cfg["callback"]["checkpoint_path"],
        filename = cfg["callback"]["filename"],
        save_top_k = cfg["callback"]["save_top_k"],
        mode = cfg["callback"]["mode"]
    )

    # train model
    pl.seed_everything(0)
    model = AgeGenderModel()

    trainer = pl.Trainer(
        callbacks = [checkpoint_callback],
        max_epochs = cfg["model"]["epochs"]
    )
    trainer.fit(model, train_dataloader, test_dataloader)

    print("Best model path: ", checkpoint_callback.best_model_path)