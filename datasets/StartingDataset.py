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
        with Image.open("../cass_data/train_images/6103.jpg") as im:
            #x = im.rotate(0).show()
            #pix_val = torch.Tensor(im.getdata()) #have to reshape to smaller size
            
            resize = im.resize((224,224))
            #x = resize.rotate(0).show()
            y = torch.Tensor(resize.getdata())
            z = y.reshape((3,224,224))
       

        inputs = torch.zeros([3, 224, 224])
        label = 0
        a = z.shape
        b = inputs.shape

        #return inputs, label
        return a, b

    def __len__(self):
        return 10000


x = StartingDataset()
print(x.__getitem__(0))
x.data.head()

#finish __getitem__
#finish len
#finish info doc