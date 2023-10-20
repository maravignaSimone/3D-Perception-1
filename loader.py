from nuimages import NuImages
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import glob
import torch
import PIL

id_dict = {}
filenames = []
front_tokens = []
#creating dictionary of classes
#eg, id_dict['animal']=1, id_dict['human.pedestrian.adult']=1, etc 0 is background
for i, line in enumerate(open('classes.txt', 'r')):
    id_dict[line.replace('\n', '')] = i+1 #creating matches class->number
nuim = NuImages(dataroot='C:/Users/franc/3DProject1/data/sets/nuimages', version='v1.0-mini', verbose=True, lazy=True)
for i, sample in  enumerate(nuim.sample): #itero la tabella sample
    """  if(i>0):
        break """
    sample_token = sample['token']  #prendo il token del sample che sto analizzando
    #print(sample_token)
    #sample_data = nuim.__load_table__('sample_data')
    sample_data_tokens = nuim.get_sample_content(sample_token) #prendo i token dalla tabella sample_data delle istanze che hanno sample_token = miosample e saranno più di uno
    #ora devo iterare sulla tabella sample data su tutti quelli che hanno questi token
    for j, token in enumerate(sample_data_tokens):
        sample_data_instance = nuim.get('sample_data', token)
        filename = sample_data_instance['filename']
        if filename.find('samples/CAM_FRONT/')!=-1:
            filenames.append(filename) #in filenames ho i path di tutte le immagini che ci servono
            front_tokens.append(token) #qui ho i tokens di sample_data delle imamgini che mi servono

class NuImagesDataset(Dataset):
    def __init__(self, root, transforms = None):
        self.root = root
        self.transforms = transforms

    def __getitem__(self, idx):
        # load images and masks
        img_path = filenames[idx]
        token = front_tokens[idx]
        img = read_image(self.root + '/' +  img_path)
        annotations = []
        #prendo tutte le annotazioni che hanno sample_data_token = token
        for obj in nuim.object_ann:
            if obj['sample_data_token'] == token:
                annotations.append(obj['token'])
        data = []
        for ann in annotations:
            label = nuim.get('category', nuim.get('object_ann', ann)['category_token'])['name']
            bbox = nuim.get('object_ann', ann)['bbox']
            cl = id_dict[label]
            data.append({
            'bbox': bbox,
            'category_id': cl,
            'category_name': label
            })
        # there is only one class
        labels = torch.as_tensor([d['category_id'] for d in data], dtype=torch.int64)
        image_id = idx
        #each bounding box is in the format xyxy
        #compute the area of each bounding box
        area = torch.as_tensor([d['bbox'][2]*d['bbox'][3] for d in data], dtype=torch.float32) #probabilmente non funziona
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(data),), dtype=torch.int64)
        
        # Wrap sample and targets into torchvision tv_tensors:
        img = torch.Tensor(img)

        target = {}
        target["boxes"] = torch.Tensor([d['bbox'] for d in data])
        area = torch.Tensor([d['bbox'][3]-d['bbox'][1] for d in data])*torch.Tensor([d['bbox'][2]-d['bbox'][0] for d in data])
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(front_tokens)
    