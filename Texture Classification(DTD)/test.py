import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import torch.utils.data as split

#reference 1:https://pytorch.org/docs/
#reference 2:https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
#reference 3:https://www.deeplearningbook.org
#resources used: https://cloud.lambdalabs.com/

#new CNN class to load our train.py model state dictionary
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=  3, out_channels = 32, kernel_size=3, padding='same')
        self.conv_2 = nn.Conv2d(in_channels=  32, out_channels= 32, kernel_size=3, padding='same')
        self.pool_1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv_3 = nn.Conv2d(in_channels=  32, out_channels = 32, kernel_size=3, padding='same')
        self.conv_4 = nn.Conv2d(in_channels=  32, out_channels= 32, kernel_size=3, padding='same')
        self.pool_2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv_5 = nn.Conv2d(in_channels=  32, out_channels = 32, kernel_size=3, padding='same')
        self.conv_6 = nn.Conv2d(in_channels=  32, out_channels= 32, kernel_size=3, padding='same')
        self.pool_3 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv_7 = nn.Conv2d(in_channels=  32, out_channels= 32, kernel_size=3, padding='same')
        self.pool_4 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.flatten = nn.Flatten()
        self.forward_1 = nn.Linear(9248, 2334)
        self.forward_2 = nn.Linear(2334, 1162)
        self.forward_3 = nn.Linear(1162, 360)
        self.forward_4 = nn.Linear(360, 47)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.pool_1(F.relu(self.conv_2(x)))
        x = F.relu(self.conv_3(x))
        x = self.pool_1(F.relu(self.conv_4(x)))
        x = F.relu(self.conv_5(x))
        x = self.pool_1(F.relu(self.conv_6(x)))
        x = self.pool_1(F.relu(self.conv_7(x)))
        x = self.flatten(x)
        x = F.relu(self.forward_1(x))
        x = F.relu(self.forward_2(x))
        x = F.relu(self.forward_3(x))
        x = F.relu(self.forward_4(x))
        return x    

if __name__ == '__main__':
    testmodel = CNN()
    test_data_loader = torch.load('unoptimizedtestset.pth')
    #load my model from train.py
    mymodel = torch.load('unoptimized_model.pth')
    #set my new model equal to train.py model
    testmodel.load_state_dict(mymodel)
    #set model in eval mode
    testmodel.eval()
    print("train.py model is loaded successfully\n starting test on train.py test_data_loader")
    #start testing on the testset loaded from train.py
    total_p = 0
    correct_p = 0
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data,(images,labels) in enumerate(test_data_loader):
            X= images
            Y= labels
            test_predict = testmodel(X.float())
            test_loss = loss(test_predict, Y)
            #get testing error
            _,compute = torch.max(test_predict,dim=1, keepdim=True)
            correct_p += (compute ==labels).sum().item()
            total_p += labels.size(0) 
            test_acc = (correct_p/total_p)*100
        print("Testing Accuracy(%):",test_acc, "test loss:",test_loss.item())
        
   