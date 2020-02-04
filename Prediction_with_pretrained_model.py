# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 08:04:43 2019

@author: GS65 8RF
"""

#Preprocess for predit image
from PIL import Image
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

def resize(img,size=(128,128),bg_color="white"):
    
    img.thumbnail(size,Image.ANTIALIAS)
    
    new_img = Image.new("RGB", size, bg_color)
    
    new_img.paste( img, (int ((size[0] - img.size[0] ) / 2) , int ((size[1] - img.size[1]) / 2)))
    
    return new_img

def predict(classifier,img_data):
    
    classifier.eval()
    
    classes = ['circle','square','triangle']
    
    transform = transforms.Compose ( [
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ] )
    
    img_tensor = torch.stack([transform(img).float() for img in img_data])
    
    predictions = []
    
    model_predict = classifier(img_tensor)
    
    for prediction in model_predict.data.numpy():
        
        prediction = np.argmax(prediction)
        
        predictions.append(classes[prediction])
        
    return np.array(predictions)

#Predicting new image
filestream = open('cnn(pytorch).h3','rb')
model = pickle.load(filestream)
filestream.close()

test_data='../Resources/test'
img_data = os.listdir(test_data)

size = (128,128)
img_array = []

fig = plt.figure(figsize=(12, 8))

for id_x in range(len(img_data)):
    image = Image.open(os.path.join(test_data, img_data[id_x]))
    resized_img = resize(image,size,"white")
    img_array.append(resized_img)

prediction = predict(model,img_array)
    
for id_x in range(len(prediction)):
    a= fig.add_subplot(1,len(prediction),id_x+1) #+1 doesn't mean increamenting it perform auto.It becz of subplot start from 1
    plt.imshow(img_array[id_x])
    a.set_title(prediction[id_x])