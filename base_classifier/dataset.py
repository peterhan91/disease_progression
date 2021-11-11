import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image


class PatchDataset(Dataset):

    def __init__(self, path_to_images, fold, sample=0, transform=None):

        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv("./label/most.csv")
        self.fold = fold
        self.df = self.df[self.df['fold'] == fold]
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(frac=sample, random_state=42)
            print('subsample the training set with ratio %f' % sample)
        self.df = self.df.set_index('name baseline')
        self.PRED_LABEL = ['fast prog']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.index[idx]
        image = Image.open(os.path.join(self.path_to_images, filename))
        image = image.convert('RGB')
        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')
        if self.transform:
            image = self.transform(image)

        return (image, label)