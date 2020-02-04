# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 21:25:52 2019

@author: GS65 8RF
"""


import os

training_folder='../Resources/training'
classes = sorted(os.listdir(training_folder))

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

#Create Loader
def Dataset_Loader(img_dataset):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
    
    whole_dataset = torchvision.datasets.ImageFolder(
            root = img_dataset,
            transform = transform )
    
    train_size = int(0.7*len(whole_dataset))
    test_size = len(whole_dataset) - train_size
    train_dataset,test_dataset = torch.utils.data.random_split(whole_dataset,[train_size,test_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = 50,
                                               )
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = 50,
                                               )
    return train_loader,test_loader

train_loader,test_loader = Dataset_Loader(training_folder)

#Create Neural Net Class
class Net(nn.Module):
    def __init__(self,num_class = 3):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size= 3, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size= 3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        
        self.flatt = nn.Linear(in_features=32*32*24, out_features=num_class)
        
    def forward(self,x):
        x = F.relu(self.pool(self.conv1(x)))
        
        x = F.relu(self.pool(self.conv2(x)))
        
        x = x.view(-1,32*32*24)
        
        x = self.flatt(x)
        
        return torch.log_softmax(x, dim=1)

device = "cpu"
if(torch.cuda.is_available()):
    device = "cuda"
model = Net(num_class = len(classes)).to(device)
print (model)
    
#Preprocess to train
def train(device,dataset,loss_criteria,model,optimizer,epochs):
    model.train()
    train_loss = 0
    batch_id = 0
    print("Epoch: ",epochs)
    
    for data,target in train_loader:
        batch_id += 1
        
        data,target = data.to(device),target.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = loss_criteria(output,target)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        print('Training on batch {} Loss:{:.6f}'.format(batch_id,loss.item()))

    avg_loss = train_loss / batch_id
    
    print('Training Set:  Average Loss: {:6f}'.format(avg_loss))
    return avg_loss
    
def test(dataset,model,loss_criteria,device):
    model.eval()
    test_loss = 0
    correct = 0
    batch_id = 0
    with torch.no_grad():
        for data,target in test_loader:
            batch_id += 1
       
            data,target = data.to(device),target.to(device)
        
            output = model(data)
        
            test_loss += loss_criteria(output,target).item()
        
            _,predict = torch.max(output.data,1)
            correct += torch.sum(target==predict).item()
        
        avg_loss = test_loss / batch_id
        
        print('Validation Set:   Average Loss: {:.6f}, Accuracy: {}/{} ({:.0f}) \n'.format
             (avg_loss, correct, len(dataset.dataset),
              100. * correct / len(dataset.dataset)))
        
        return avg_loss

#Train the model
optimizer = optim.Adam(model.parameters(),lr=0.01)
loss_criteria = nn.CrossEntropyLoss()

epoch_num = []
train_process = []
test_process = []

epochs = 5
for epoch in range(1,epochs+1):
    train_loss = train(device,train_loader,loss_criteria,model,optimizer,epoch)
    test_loss = test(test_loader,model,loss_criteria,device)
    epoch_num.append(epoch)
    train_process.append(train_loss)
    test_process.append(test_loss)

import matplotlib.pyplot as plt
plt.plot(epoch_num,train_process)   
plt.plot(epoch_num,test_process)     
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training','Validation'],loc='upper right')
plt.show()
        
#Evaluating Model
from sklearn.metrics import confusion_matrix 
import numpy as np

true_value = []
predictions = []
model.eval()

device = "cpu"

for data,target in test_loader:
    model=model.to(device)
    for label in target.data.numpy():
        true_value.append(label)
    for prediction in model(data).data.numpy().argmax(1):
        predictions.append(prediction)

cm = confusion_matrix(true_value,predictions)

plt.imshow(cm,cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks,classes)
plt.yticks(tick_marks,classes)
plt.xlabel("Predicted Shape")
plt.ylabel("True Shape")
plt.show()

#Saving the trained model
import pickle
filestream = open('cnn(pytorch).h3','wb')
pickle.dump(model,filestream)
filestream.close()