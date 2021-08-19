from torch import nn

class AgeGenderModel(nn.Module):
    def __init__(self):
        super(AgeGenderModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.5)
        )

        self.fc_age = nn.Linear(73926, 1)  
        self.fc_gender = nn.Linear(73926, 1)    
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)

        age = self.fc_age(x)
        gender= torch.sigmoid(self.fc_gender(x))  

        return {'age': age, 'gender': gender}