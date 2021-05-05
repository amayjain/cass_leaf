import torch
import torchvision.transforms as transforms 
import pandas as pd
from PIL import Image

class StartingDataset(torch.utils.data.Dataset):
    def __init__(self, df_path="cass_data/train.csv", img_path="cass_data/train_images", train=True):
        df = pd.read_csv(df_path).sample(frac = 1, random_state=42).reset_index(drop=True)

        train_percentage = 0.8
        rows = df.shape[0]
        train_rows = int(rows * train_percentage)

        if train:
            self.df = df.iloc[:train_rows]
        else:
            self.df = df.iloc[train_rows:]
        
        self.df = self.df.reset_index(drop=True)

        self.img_path = img_path

        self.transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        jpg_str = str(self.df.loc[index]['image_id'])
        label = self.df.loc[index]['label']

        with Image.open(f"{self.img_path}/{jpg_str}") as im:
            im = self.transforms(im)

        #print("label:", label)
        return im, label

    def __len__(self):
        return len(self.df)


if __name__ == '__main__':
    df_path = "../cass_data/train.csv"
    img_path = "../cass_data/train_images"

    train = StartingDataset(df_path=df_path, img_path=img_path, train=True)
    test = StartingDataset(df_path=df_path, img_path=img_path, train=False)
    
    # print(train[42])
    # train.df.head()
