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
epochs = 3

#-------------------------------------------
# dataset and dataloader
# ------------------------------------------
id_dict = {}
#eg, id_dict['animal']=1, id_dict['human.pedestrian.adult']=2, etc 0 is background
for i, line in enumerate(open('./classes.txt', 'r')):
    id_dict[line.replace('\n', '')] = i+1 #creating matches class->number
train_dataset = NuImagesDataset('./data/sets/nuimages', id_dict=id_dict, version='mini')
val_dataset = NuImagesDataset('./data/sets/nuimages', id_dict=id_dict, version='mini')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

#-------------------------------------------
# model
# ------------------------------------------

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

num_classes = 24 # 23 classes + background

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.to(device)

#-------------------------------------------
# loss function, optimizer and metric
# ------------------------------------------

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

mAP = MeanAveragePrecision().to(device)

#-------------------------------------------
# training
# ------------------------------------------

print('Start training...')

train_losses = []
val_accuracy = []

for epoch in range(epochs):
    trainloss = 0
    valaccuracy = 0

    model.train()
    for i, data in enumerate(train_loader):
        if(i>0):
            break
        images, boxes, labels = data

        images = list((image/255.0).to(device) for image in images)

        targets = []
        for i in range(len(images)):
            d = {}
            d['boxes'] = boxes[i].to(device)
            d['labels'] = labels[i].to(device)
            targets.append(d)

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        print(losses)

        trainloss += losses.item()

    print('Epoch: {} - Finished training, starting eval'.format(epoch))
    train_losses.append(trainloss/len(train_loader))

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            """ if(i>0):
                break """
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
        
        val_accuracy.append(valaccuracy/len(val_loader))

    print('Epoch: {} - Finished eval'.format(epoch))
    print('Train loss: {}'.format(train_losses[-1]))
    print('Val accuracy: {}'.format(val_accuracy[-1]))

