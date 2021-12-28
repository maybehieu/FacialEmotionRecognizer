import torch
from torch.utils.data import dataset
from torchvision.transforms import transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd

import os
from csv import reader


DATA_ROOT = ""  # path to 'datasets' folder
CSV_ROOT = ""  # path to 'label' folder


# function to create dataFrame containing corresponding label to images
def process_csv(data_type="train"):
    if data_type == "train":
        CSV_DIR = os.path.join(CSV_ROOT, "trainLabel.csv")
    elif data_type == "val":
        CSV_DIR = os.path.join(CSV_ROOT, "valLabel.csv")
    elif data_type == "test":
        CSV_DIR = os.path.join(CSV_ROOT, "testLabel.csv")
    origin_df = pd.read_csv(CSV_DIR, header=None)
    origin_df.columns = [
        "img_name", "dims", "0", "1", "2", "3", "4", "5", "6", "7",
        "Unknown", "NF"
    ]

    origin_df.sort_values(by=['img_name'])
    fileName_df = origin_df['img_name'].values

    return origin_df, fileName_df

# function to create dataset for dataloader
# dataset = CustomDataset(mode="train" || "val" || "test")


class CustomDataset(data.Dataset):
    def __init__(self, dataset_type="train") -> None:
        super(CustomDataset, self).__init__()
        self.df, self.fileName_df = process_csv(dataset_type)
        self.mode = dataset_type
        if self.mode == "train":
            self.DATA_DIR = os.path.join(DATA_ROOT, "FER2013Train")
        elif self.mode == "val":
            self.DATA_DIR = os.path.join(DATA_ROOT, "FER2013Valid")
        elif self.mode == "test":
            self.DATA_DIR = os.path.join(DATA_ROOT, "FER2013Test")

    def img_transform(self, img):

        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.mode == "train":
            img = train_transform(img)
        else:
            img = val_transform(img)
        return img

    def get_class(self, file_name):

        row_df = self.df[self.df["img_name"] == file_name]
        init_val = -1
        init_idx = -1
        for x in range(2, 10):
            max_val = max(init_val, row_df.iloc[0].values[x])
            if max_val > init_val:
                init_val = max_val
                init_idx = int(
                    x - 2
                )  # Labels indices start at 0
        return init_idx

    def __getitem__(self, index):
        fileName = self.fileName_df[index]
        filePath = os.path.join(self.DATA_DIR, fileName)
        img = Image.open(filePath)
        img = self.img_transform(img)
        label = self.get_class(fileName)
        return img, torch.tensor(label).to(torch.long)

    def __len__(self):
        return len(self.df)
