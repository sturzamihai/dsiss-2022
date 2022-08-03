import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from tqdm.notebook import tqdm
from tqdm import trange

sns.set_theme()

import os, datetime, random, math, glob, pydicom, warnings
import pickle, cv2
from pathlib import Path
 
import torch, torchviz, torchvision
from torch import nn
from torchsummary import summary
from torchviz import make_dot
from torch.utils.data import Dataset, DataLoader

base_dir = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
os.listdir(base_dir)

train_df = pd.read_csv(os.path.join(base_dir,"train.csv"))
test_df = pd.read_csv(os.path.join(base_dir,"test.csv"))

print(train_df.shape)
train_df.head(10)

class PulmonaryDataset(Dataset):
    def __init__(self, csv_file, base_dir="/kaggle/input/osic-pulmonary-fibrosis-progression/", target='train',transform=None):
        self.csv_data = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.transform = transform
        self.target = target             
        
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx):
        img_folder = os.path.join(self.base_dir,self.target,self.csv_data.iloc[idx].Patient)
        
        ## GETTING ONLY FIRST IMAGE FOR THE PATIENT
        for img_path in os.listdir(img_folder):
            img = pydicom.read_file(os.path.join(img_folder, img_path))
            try:
                img = cv2.resize(img.pixel_array.astype(float)/2**11, (224,224))
                stacked_img = np.stack((img,)*3, axis=-1)
            except Exception as e:
                print("Compressed image found.. retrying")
                img_array = img.decompress('pillow')
                img = cv2.resize(img.pixel_array.astype(float)/2**11, (224,224))
                stacked_img = np.stack((img,)*3, axis=-1)
            
            gender_label = {
                'Male': 1,
                'Female': 0
            }
            
            smoker_label = {
                'Never smoked': -1,
                'Ex-smoker': 0,
                'Currently smokes': 1
            }
            
            tab_data = np.array([self.csv_data.iloc[idx].Age, gender_label[self.csv_data.iloc[idx].Sex],
                        smoker_label[self.csv_data.iloc[idx].SmokingStatus]])
            
            return {'features': tab_data, 'image': stacked_img, 'target': self.csv_data.iloc[idx].FVC}

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super(ResNetBlock, self).__init__()
        
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()
            
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = x + shortcut
        return nn.ReLU()(x)

class PulmonaryResNetModel(nn.Module):
    def __init__(self, in_channels, resblock, outputs=1):
        super(PulmonaryResNetModel, self).__init__()
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU()
        )
        
        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False),
            resblock(256, 256, downsample=False)
        )


        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False),
            resblock(512, 512, downsample=False),
        )
        
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc_resnet = torch.nn.Linear(512, 64)
        
        self.layer_tab = nn.Sequential(
            nn.Linear(3,64),
            nn.ReLU(),
        )
        
        self.fc_model = torch.nn.Linear(128, 1)
        
    def forward(self, x_img, x_tab):
        x_img = self.layer0(x_img)
        x_img = self.layer1(x_img)
        x_img = self.layer2(x_img)
        x_img = self.layer3(x_img)
        x_img = self.layer4(x_img)
        x_img = self.gap(x_img)
        x_img = self.flatten(x_img)
        x_img = self.fc_resnet(x_img)
        
        x_tab = self.layer_tab(x_tab)
        
        x = torch.cat((x_img,x_tab),1)
        x = self.fc_model(x)
        
        return x

pulmonary_dataset = PulmonaryDataset(csv_file=os.path.join(base_dir,"train.csv"), base_dir = base_dir, target='train')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PulmonaryResNetModel(3,ResNetBlock)
model.to(device)

print(model)

x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)
x_tab = torch.zeros(1,3 , dtype=torch.float, requires_grad=False)
out = model(x, x_tab)

dot = make_dot(out)
dot.format = 'png'
dot.render('model_graph')

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

running_loss = 0.
last_loss = 0.

pulmonary_dataloader = DataLoader(pulmonary_dataset, batch_size=32,shuffle=True)

for i, data in enumerate(pulmonary_dataloader):
    images = np.reshape(data['image'].float().to(device), (32, 3,224,224)).type(torch.FloatTensor)
    features = data['features'].float().to(device)
    targets = data['target'].float().to(device)
    
    optimizer.zero_grad()
    preds = model(images, features)
    loss = loss_fn(preds, targets)
    loss.backward()
    optimizer.step()
    
    running_loss += loss.item()
    last_loss = running_loss / 32 # loss per batch
    print('  batch {} loss: {}'.format(i + 1, last_loss))
    running_loss = 0.