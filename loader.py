import torch
from nuimages import NuImages
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

#defining the dataset class
class NuImagesDataset(Dataset):
    def __init__(self, root, id_dict):
        self.root = root
        self.filenames = []
        self.front_tokens = []
        self.id_dict = id_dict
        self.nuim = NuImages(dataroot=root, version='v1.0-mini', verbose=True, lazy=True)
        #now we need to build the lists of filenames and tokens
        for i, sample in  enumerate(self.nuim.sample):
            sample_token = sample['token'] #this is the token of the sample we are analyzing
            sample_data_tokens = self.nuim.get_sample_content(sample_token) #these are the tokens of the sample_data instances that have sample_token = sample_token
            #now we need to iterate over the sample_data table on all the tokens we have
            for j, token in enumerate(sample_data_tokens):
                sample_data_instance = self.nuim.get('sample_data', token) #this is the instance of sample_data that has token = token
                filename = sample_data_instance['filename'] #this is the filename of the image
                #now we need to check if the image is a front camera image and if it is we add it to the list
                if filename.find('samples/CAM_FRONT/')!=-1:
                    self.filenames.append(filename)
                    self.front_tokens.append(token)


    def __getitem__(self, idx):
        # take the idx-th image and its token
        img_path = self.filenames[idx]
        token = self.front_tokens[idx]
        #read the image as tensor
        img = read_image(self.root + '/' +  img_path)
        annotations = []
        #get the annotations tokens of the image
        for obj in self.nuim.object_ann:
            if obj['sample_data_token'] == token:
                annotations.append(obj['token'])
        data = []
        #get the labels and the bboxes of the annotations
        for ann in annotations:
            label = self.nuim.get('category', self.nuim.get('object_ann', ann)['category_token'])['name']
            bbox = self.nuim.get('object_ann', ann)['bbox']
            cl = self.id_dict[label]
            data.append({
            'bbox': bbox,
            'category_id': cl,
            'category_name': label
            })
        
        
        
        # Wrap sample and targets into torchvision tv_tensors:
        #img = torch.Tensor(img)

        # put boxes and labels into tensors
        boxes = torch.Tensor([d['bbox'] for d in data])
        labels = torch.as_tensor([d['category_id'] for d in data], dtype=torch.int64)

        return img, boxes, labels

    def __len__(self):
        return len(self.front_tokens)
    