import torch
import torchvision
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, train_check):
        if train_check == true:
            self.data = pd.read_csv('../cass_data/train.csv') # 80%
        else:
            self.data = pd.read_csv('../cass_data/train.csv') # 20%


    def __getitem__(self, index):
       # x = "../cass_data/" + str(pd[index])
        # y = "../cass_data/" + "1000015157.jpg"
        jpg_str = str((self.data.loc[index])['image_id'])
        labels = (self.data.loc[index])['label']
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
        return z, labels

    def __len__(self):
        return 21367


leaf_traindata = StartingDataset(true)
leaf_testdata = StartingDataset(false)
#print(x.__getitem__(23))
#x.data.head()

leaf_traindataload = torch.utils.data.DataLoader(leaf_traindata, batch_size= 16, shuffle=True)
leaf_testdataload = torch.utils.data.DataLoader(leaf_testdata, batch_size= 16, shuffle=True)

