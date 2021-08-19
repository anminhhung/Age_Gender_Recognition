import os 
import cv2 
import numpy as np 

from sklearn.model_selection import train_test_split
from torchvision import transforms

class MyData(Dataset):
    def __init__(self, path_dir, test_ratio=0.2, train=True, transform=None):
        """
            Input:
                path_dir: train folder
                test_ratio: split size
            formart image_name: <number_id>_A<age>_G<0,1>.png
        """
        list_age = []
        list_gender = []
        list_path = []
        for image_name in os.listdir(path_dir):
            age = ((image_name.split(".")[0]).split("_")[1]).split("A")[-1]
            gender = ((image_name.split(".")[0]).split("_")[2]).split("G")[-1]
            image_path = os.path.join(path_dir, image_name)
            
            list_age.append(int(age))
            list_gender.append(int(gender))
            list_path.append(image_path)

        # #Splitting the data into train and validation set
        X_train, X_test, y_age_train, y_age_test, y_gender_train, y_gender_test = \
        train_test_split(list_path, list_age, list_gender, test_size=test_ratio)
        
        if train:
            self.X = X_train
            self.age_y = y_age_train
            self.gender_y = y_gender_train
        else:
            self.X = X_test
            self.age_y = y_age_test
            self.gender_y = y_gender_test
        
        # apply transformation
        self.transform=transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.X[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float')
        age = np.array(self.age_y[idx]).astype('float')
        gender = np.array(self.gender_y[idx]).astype('float')

        sample={'image': image, 'label_age': age, 'label_gender': gender}

        if self.transform:
            sample = self.transform(sample)
        
        return sample