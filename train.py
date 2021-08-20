import argparse
import pytorch_lightning as pl

from torchvision import transforms
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader 
from pytorch_lightning.callbacks import ModelCheckpoint

from configs.init import init_config
from src.custom_transform import RGBToTensor
from src.dataset import AgeGenderData
from src.model import AgeGenderModel

def parse_args():
    parser = argparse.ArgumentParser(description='List the content')
    parser.add_argument("--cfg_path", default="configs/config.ini", help='path of config file')
  
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = init_config(args.cfg_path)
    rgb_to_tensor = RGBToTensor()
    
    # dataloader
    transformed_train_data = AgeGenderData(path_dir=cfg["dataset"]["data_path"], \
                                    train_val_ratio=float(cfg["dataset"]["train_val_ratio"]), train=True, \
                                    transform=transforms.Compose([RGBToTensor()]))

    transformed_test_data = AgeGenderData(path_dir=cfg["dataset"]["data_path"], \
                                    train_val_ratio=float(cfg["dataset"]["train_val_ratio"]), train=True, \
                                    transform=transforms.Compose([RGBToTensor()]))
    
    train_dataloader = DataLoader(transformed_train_data, batch_size=int(cfg["model"]["batch_size"]), shuffle=True)
    test_dataloader = DataLoader(transformed_test_data, batch_size=int(cfg["model"]["batch_size"]), shuffle=True)

    checkpoint_callback = ModelCheckpoint(
        monitor = cfg["callback"]["monitor"],
        dirpath = cfg["callback"]["checkpoint_path"],
        filename = cfg["callback"]["filename"],
        save_top_k = int(cfg["callback"]["save_top_k"]),
        mode = cfg["callback"]["mode"]
    )

    # train model
    pl.seed_everything(0)
    model = AgeGenderModel(cfg)

    trainer = pl.Trainer(
        callbacks = [checkpoint_callback],
        max_epochs = int(cfg["model"]["epochs"])
    )
    trainer.fit(model, train_dataloader, test_dataloader)

    print("Best model path: ", checkpoint_callback.best_model_path)