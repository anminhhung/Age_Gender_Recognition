import torch 

class RGBToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, age, gender = sample['image'], sample['label_age'], sample['label_gender']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        age = torch.from_numpy(age).float()
        gender = torch.from_numpy(gender).float()

        return {'image': image,
                'label_age': age,
                'label_gender': gender}