import cv2 
import torch 
import argparse

from configs.init import init_config
from src.model import AgeGenderModel

def parse_args():
    parser = argparse.ArgumentParser(description='List the content')
    parser.add_argument("--image", require=True, help='path of image')
  
    return parser.parse_args()

def predict(image, loaded_model, max_age, gender_threshold=0.5):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = image.unsqueeze(0) 

    predicted = loaded_model(image)
    age = predicted["age"].item() * max_age
    prob_gender = predicted["gender"].item()
    gender = 1 if prob_gender > gender_threshold else 0
    
    return age, gender

if __name__ == "__main__":
    args = parse_args()
    cfg = init_config()

    # load model
    age_gender_model = AgeGenderModel()
    age_gender_model = loaded_model.load_from_checkpoint(cfg["inference"]["checkpoint_path"])

    # predict
    image_path = args.image
    image = cv2.imread(image_path)
    pred_age, pred_gender = predict(image, loaded_model, \
                    cfg["dataset"]["max_age"], cfg["inference"]["gender_threshold"])

    print("pred_age: {}, pred_gender: {}".format(pred_age, pred_gender))