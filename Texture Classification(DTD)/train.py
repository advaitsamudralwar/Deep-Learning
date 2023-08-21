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
from torch.utils.tensorboard import SummaryWriter


#reference 1:https://pytorch.org/docs/
#reference 2:https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
#reference 3:https://www.deeplearningbook.org
#resources used: https://cloud.lambdalabs.com/
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

#preprocess and load dataset
def loadData():
    # Download dataset
    transform = T.Compose([T.Resize((300, 300)), 
    T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3))])
    dataset_download = datasets.DTD(root='./data', download=True, transform=transform)
    print("Preprocessing Data......")
    dataloader = DataLoader(dataset_download, shuffle=True)
    train_size = int(len(dataloader)*0.65)
    validate_size = int(len(dataloader)*0.15)
    test_size  = int(len(dataloader)*0.20)
    print(len(dataloader), train_size,validate_size,test_size)
    #split data into train test and validate data
    train_data , val_data, test_data = split.random_split(dataset_download, [train_size,validate_size, test_size])
    train_data_loader = DataLoader(train_data, batch_size=30, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=30, shuffle=False)
    torch.save(test_data_loader,'unoptimizedtestset.pth')
    val_data_loader = DataLoader(val_data, batch_size=30, shuffle=True)
    trainCnn(train_data_loader)


##Train CNN
def trainCnn(trainset):
    print("Data has been successfully preprocess")
    print("Training now:")
    epoch = 10
    mymodel.train()
    total_p = 0
    correct_p = 0
    loss = nn.CrossEntropyLoss()
    optimize_model = torch.optim.SGD(mymodel.parameters(), 0.0001, momentum=0.9)
    for i in range(epoch):
        for data, (images,labels) in enumerate(trainset):
            X= images
            Y= labels
            train_predict = mymodel(X.float())
            train_loss = loss(train_predict, Y)
            optimize_model.zero_grad()
            train_loss.backward()
            optimize_model.step()
            #get training error
            _,compute = torch.max(train_predict,dim=1, keepdim=True)
            correct_p += (compute == labels).sum().item()
            total_p += labels.size(0) 
            train_acc = (correct_p/total_p)*100
            #add loss and training acc to log
            writer.add_scalar('Loss', train_loss.item(), len(trainset) * i + data)
            writer.add_scalar('Training Accuracy(%)', train_acc, len(trainset) * i + data)
            # acc = compute.eq(Y.view_as(compute)).sum().item()
            # t_acc += acc
            # inputs += input.size(0)
            # t_acc = 100*(t_acc/inputs)
        print("Epoch:", i ,"Training Accuracy(%):",train_acc, "Loss:", train_loss.item())
        #add graphs and images to log
        writer.add_graph(mymodel,images)
    #save my optimized model to be used by test.py
    torch.save(mymodel.state_dict(), 'unoptimized_model.pth')
    print("Training Complete")
    
    #Cnn(X_test,Y_test, batchsize)


if __name__ == '__main__':
    direct = os.getcwd()
    writer = SummaryWriter(log_dir='log/trainlog')
    mymodel = CNN()
    loadData()
    