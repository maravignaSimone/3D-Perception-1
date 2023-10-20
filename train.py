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
epochs = 10

#-------------------------------------------
# dataset and dataloader
# ------------------------------------------

train_dataset = NuImagesDataset('C:/Users/franc/3DProject1/data/sets/nuimages')
val_dataset = NuImagesDataset('C:/Users/franc/3DProject1/data/sets/nuimages')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

#-------------------------------------------
# model
# ------------------------------------------

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

num_classes = 23 # 23 classes + background

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
train_accuracy = []
val_losses = []
val_accuracy = []

for epoch in range(epochs):
    trainloss = 0
    valloss = 0
    trainaccuracy = 0
    valaccuracy = 0

    model.train()
    for i, data in enumerate(train_loader):
        if(i>0):
            break
        images, targets = data

        images = list(image for image in images)
        optimizer.zero_grad()
        target = []
        target.append(targets)
        target[0]['boxes']= target[0]['boxes'].squeeze(0)
        print(target[0]['boxes'].shape )
        print(target)
        images = [(image / 255.0) for image in images] # transform images to [0,1] range
        images = torch.stack(images).to(device)
        target[0]['boxes'].to(device)
        output = model(images, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        trainloss += loss.item()
        trainaccuracy += mAP(output, target)

    print('Epoch: {} - Finished training, starting eval'.format(epoch))
    train_losses.append(trainloss/len(train_loader))
    train_accuracy.append(trainaccuracy/len(train_loader))

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if(i>0):
                break
            images, targets = data

            images = list(image for image in images)
            targets = {k: v.to(device) for k, v in targets.items()}

            output = model(images, targets)
            loss = criterion(output, targets)

            valloss += loss.item()
            valaccuracy += mAP(output, targets)
        
        val_losses.append(valloss/len(val_loader))
        val_accuracy.append(valaccuracy/len(val_loader))

    print('Epoch: {} - Finished eval'.format(epoch))
    print('Train loss: {} - Train accuracy: {}'.format(train_losses[-1], train_accuracy[-1]))
    print('Val loss: {} - Val accuracy: {}'.format(val_losses[-1], val_accuracy[-1]))

