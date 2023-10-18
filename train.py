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

from loader import NuImagesDataset

#-------------------------------------------
# hyperparameters
#-------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10

#-------------------------------------------
# dataset and dataloader
# ------------------------------------------

train_dataset = NuImagesDataset('data/sets/nuimages')
val_dataset = NuImagesDataset('data/sets/nuimages')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)

#-------------------------------------------
# model
# ------------------------------------------

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

num_classes = 2

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.to(device)

#-------------------------------------------
# loss function, optimizer and metric
# ------------------------------------------

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



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
        input, target = data
        print(target)
        if(i>0):
            break