import torch
import torchvision
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        self.data = pd.read_csv('../cass_data/train.csv')


    def __getitem__(self, index):
        # PIL.Image.open(fp, mode='r', formats=None)
        with Image.open("../cass_data/train_images/6103.jpg") as im:
            im.rotate().show()
        # inputs = torch.zeros([3, 224, 224])
        # label = 0

        return inputs, label

    def __len__(self):
        return 10000


x = StartingDataset()