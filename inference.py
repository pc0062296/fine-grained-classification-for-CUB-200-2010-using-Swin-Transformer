#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from torchvision import transforms
import timm
from PIL import Image
from torch.autograd import Variable


def testoutput(imgids, labels, net, testLoader):
    net.eval()
    i = 0
    submission = []
    for data in testLoader:
        data = data.cuda()
        data = Variable(data, volatile=True).to('cuda')
        output = net(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability

        submission.append([imgids[int(i)], labels[int(pred)]])
        print('{} {}'.format(imgids[int(i)], labels[int(pred)]))
        i += 1
    np.savetxt('answer.txt', submission, fmt='%s')


class bird2Dataset(Dataset):
    def __init__(self, imgs, transform):
        self.transform = transform
        self.imgs = imgs
        self.imgs = [f'dataset/test/{i}' for i in self.imgs]
        print('Total data in {}'.format(len(self.imgs)))

    def __getitem__(self, index):
        imgpath = self.imgs[index]
        img = Image.open(imgpath).convert('RGB')
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imgs)

imgids = []
labels = []
with open('dataset/testing_img_order.txt', "r", encoding="utf-8") as f:
    for line in f.readlines():
        strr = line.split("\n")
        imgids.append(strr[0])

with open('dataset/classes.txt', "r", encoding="utf-8") as f:
    for line in f.readlines():
        strr = line.split("\n")
        labels.append(strr[0])

test_preprocess = transforms.Compose([transforms.Resize((420, 420), Image.BILINEAR),
                                     transforms.CenterCrop((384, 384)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ds_test = bird2Dataset(imgids, transform=test_preprocess)
testLoaderr = DataLoader(ds_test, batch_size=1, shuffle=False)

model = timm.models.swin_large_patch4_window12_384_in22k(pretrained=True)
model.head = torch.nn.Linear(in_features=1536, out_features=200, bias=True)
model.to('cuda')

pretrained_model = torch.load("output/hw1model/hw1model.bin")['model']
model.load_state_dict(pretrained_model)

with torch.no_grad():
    testoutput(imgids, labels, model, testLoaderr)