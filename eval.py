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

from loader import NuImagesDataset, collate_fn

#-------------------------------------------
# hyperparameters
#-------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------
# dataset and dataloader
# ------------------------------------------

id_dict = {}
#eg, id_dict['animal']=1, id_dict['human.pedestrian.adult']=2, etc 0 is background
for i, line in enumerate(open('/hpc/home/simone.maravigna/3D-Perception-1/classes.txt', 'r')):
    id_dict[line.replace('\n', '')] = i+1 #creating matches class->number

val_dataset = NuImagesDataset('/hpc/home/simone.maravigna/3D-Perception-1/data/sets/nuimages', id_dict=id_dict, version='val')
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

#-------------------------------------------
# model
# ------------------------------------------

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

model.to(device)

model.load_state_dict(torch.load("/hpc/home/simone.maravigna/3D-Perception-1/checkpoints"))

mAP = MeanAveragePrecision().to(device)

#-------------------------------------------
# evaluation
# ------------------------------------------

print('Start evaluation...')

valaccuracy = 0

model.eval()
with torch.no_grad():
    for i, data in enumerate(val_loader):
        images, boxes, labels = data

        images = list((image/255.0).to(device) for image in images)

        targets = []
        for i in range(len(images)):
            d = {}
            d['boxes'] = boxes[i].to(device)
            d['labels'] = labels[i].to(device)
            targets.append(d)

        output = model(images)
        print(output)

        mean_ap = mAP.forward(output, targets)
        valaccuracy += mean_ap['map'].item()

print('Val accuracy: {}'.format(valaccuracy/len(val_loader)))
