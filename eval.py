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
import os
from PIL import Image, ImageDraw, ImageFont
from loader import NuImagesDataset, collate_fn, get_id_dict, get_id_dict_rev

#-------------------------------------------
# hyperparameters
#-------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------
# dataset and dataloader
# ------------------------------------------

id_dict = get_id_dict()
id_dict_rev = get_id_dict_rev()
val_dataset = NuImagesDataset('./data/sets/nuimages', id_dict=id_dict, version='mini')
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)

#-------------------------------------------
# model
# ------------------------------------------

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
num_classes = 24 # 23 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.to(device)

model.load_state_dict(torch.load("C:/Users/franc/Documents/GitHub/3D-Perception-1/checkpoints/checkpoint_epoch9_2023-10-23_01-46-43.pth"))

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
        for im in range(len(images)):
            d = {}
            d['boxes'] = boxes[im].to(device)
            d['labels'] = labels[im].to(device)
            targets.append(d)

        output = model(images)

        mean_ap = mAP.forward(output, targets)
        valaccuracy += mean_ap['map'].item()
        for j in range(len(output)):
            print(output[j])
            img = Image.fromarray(images[j].mul(255).permute(1, 2, 0).byte().cpu().numpy())
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", 25)
            
            for k in range(len(output[j]['boxes'])):
                    box = output[j]['boxes'][k]
                    label = output[j]['labels'][k]
                    score = output[j]['scores'][k]
                    score = score.item()
                    
                    if(score>0.7):
                        draw.rectangle(box.tolist(), outline='red', width=2)
                        draw.text((box[0], box[1]),id_dict_rev[label.item()] , font=font, fill='red', stroke_width=1)
            img.show()
            img.save(os.path.join('./output/eval', 'img'+str(i)+'.jpg'))
print('Val accuracy: {}'.format(valaccuracy/len(val_loader)))
