# Authors: Simone Maravigna, Francesco Marotta
# Date: 2023-10
# Project: 2D Object Detection on nuImages Dataset using Faster R-CNN

#-------------------------------------------
# libraries
#-------------------------------------------
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.rpn import AnchorGenerator
from loader import NuImagesDataset

#-------------------------------------------
# hyperparameters
#-------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------
# dataset and dataloader
# ------------------------------------------
id_dict = {}
#eg, id_dict['animal']=1, id_dict['human.pedestrian.adult']=2, etc 0 is background
for i, line in enumerate(open('./classes.txt', 'r')):
    id_dict[line.replace('\n', '')] = i+1 #creating matches class->number
val_dataset = NuImagesDataset('./data/sets/nuimages', id_dict=id_dict)
val_loader = DataLoader(val_dataset, batch_size=1)

#-------------------------------------------
# model
# ------------------------------------------

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

model.to(device)

#-------------------------------------------
# evaluation
# ------------------------------------------
model.eval()
with torch.no_grad():
    for i, data in enumerate(val_loader):
        """ if(i>0):
            break """
        images, boxes, label = data

        images = list((image/255.0).to(device) for image in images)

        """ targets = []
        for i in range(len(images)):
            d = {}
            d['boxes'] = boxes[i]
            d['labels'] = labels[i]
            targets.append(d) """

        output = model(images)
        print(output)