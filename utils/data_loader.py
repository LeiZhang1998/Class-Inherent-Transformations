from torch.utils import data
from PIL import Image
import numpy as np
import torch


name = {'COVID-19': 0, 'normal': 1, 'pneumonia': 2}


class ClsDataLoader(data.Dataset):
    def __init__(self, txt_path, transform=None):
        super(ClsDataLoader, self).__init__()
        self.transform = transform
        file = open(txt_path, 'r')
        self.data_list = []
        for line in file:
            self.data_list.append(line.strip().rsplit(' ', 1))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        (img_path, label) = self.data_list[item]
        img = Image.open('datasets/' + img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.tensor(int(label))
        return img, img, label


class MulDataLoader(data.Dataset):
    def __init__(self, txt_path, normal=False):
        super(MulDataLoader, self).__init__()
        self.normal = normal
        file = open(txt_path, 'r')
        self.data_list = []
        for line in file:
            self.data_list.append(line.strip())

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data_path = self.data_list[item]
        data = open(data_path, 'r')
        lines = data.readlines()
        cls = lines[0].strip()
        huayan = lines[1].strip().split()
        bo = lines[2].strip().split()
        huayan = list(map(float, huayan))
        bo = list(map(float, bo))
        cls = torch.tensor(int(cls))
        # huayan = torch.tensor(huayan)
        # bo = torch.tensor(bo)[:144]
        huayan = torch.tensor(huayan)
        bo = torch.tensor([bo])[:, :144]
        return bo, huayan, cls


if __name__ == '__main__':
    d = ClsDataLoader('../cross/covid/train_spilt_1.txt')
    print(d.classes)