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
from PIL import Image, ImageDraw, ImageFont
import cv2


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
        images, boxes, label = data

        images = list((image/255.0).to(device) for image in images)
        outputs = model(images)

        # Draw bounding boxes on the images
        for j in range(len(outputs)):
            print(outputs[j])
            img = Image.fromarray(images[j].mul(255).permute(1, 2, 0).byte().cpu().numpy())
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", 25)
            
            for k in range(len(outputs[j]['boxes'])):
                    box = outputs[j]['boxes'][k]
                    label = outputs[j]['labels'][k]
                    draw.rectangle(box.tolist(), outline='red')
                    draw.text((box[0], box[1]), str(label.item()), font=font, fill='red', stroke_width=1)
            img.show()

                
            
#outputs[j]['boxes'], outputs[j]['labels']