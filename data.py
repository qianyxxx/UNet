import os

from torch.utils.data import Dataset
from utils import *
from torchvision import transforms
from torch.nn import functional as F

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        # 初始化数据集
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        # 返回数据集的长度
        return len(self.name)

    def __getitem__(self, index):
        # 返回数据集中索引为index的数据
        segment_name = self.name[index]  # xxx.png
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', segment_name.replace('.png', '.jpg'))  # xxx.jpg
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image), transform(segment_image)


if __name__ == '__main__':
    data = MyDataset('../dataset/VOCdevkit/VOC2012')
    print(data[1][0].shape)
    print(data[0][0].shape)
    out = F.one_hot(data[0][1].long())  # [H, W] -> [H, W, C] c: class
    print(out.shape)
