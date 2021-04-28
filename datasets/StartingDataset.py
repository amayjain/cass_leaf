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
       # x = "../cass_data/" + str(pd[index])
        # y = "../cass_data/" + "1000015157.jpg"
        jpg_str = str((self.data.loc[index])['image_id'])
        with Image.open("../cass_data/train_images/" + jpg_str) as im:
            #x = im.rotate(0).show()
            #pix_val = torch.Tensor(im.getdata()) #have to reshape to smaller size
            
            resize = im.resize((224,224))
            #x = resize.rotate(0).show()
            y = torch.Tensor(resize.getdata())
            z = y.reshape((3,224,224))
       

        #inputs = torch.zeros([3, 224, 224])
       #label = 0
        #a = z.shape
        #b = inputs.shape

        #return inputs, label
        return z

    def __len__(self):
        return 21367


x = StartingDataset()
#print(x.__getitem__(23))
#x.data.head()


"""
for i in range(26):
    print(x.__getitem__(i))
"""

#finish __getitem__
#finish len
#finish info doc