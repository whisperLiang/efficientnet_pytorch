from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as FUN
import os
from scipy import io
import json
from efficientnet_pytorch.model import EfficientNet
from PIL import Image, ImageDraw, ImageFont

# input_size = 224
class_num = 20
image_dir = './images/test/0/fe803d232e3c959f95e4df9b9b383432.jpg'
use_gpu = torch.cuda.is_available()




def test_model(model):
    model.eval()
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    image = Image.open(image_dir)
    img = tfms(image).unsqueeze(0)
    img = Variable(img.cuda())
    # print(img.shape) # torch.Size([1, 3, 224, 224])
    
    labels_map = json.load(open('examples/simple/underwater.txt'))
    labels_map = [labels_map[str(i)] for i in range(20)]

    with torch.no_grad():
        outputs = model(img)
    # Print predictions
    print('-----')
    cout = 0
    for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
        cout += 1
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))



if __name__ == '__main__':
    # 自动下载到本地预训练
    # model_ft = EfficientNet.from_pretrained('efficientnet-b0')
    model_ft = EfficientNet.from_name('efficientnet-b5')
    # 离线加载预训练，需要事先下载好
    # model_ft = EfficientNet.from_name(net_name)
    # net_weight = 'eff_weights/' + pth_map[net_name]
    # state_dict = torch.load(net_weight)
    # model_ft.load_state_dict(state_dict)

    # 修改全连接层
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, class_num)
    if use_gpu:
        model_ft = model_ft.cuda()
    print('-' * 10)
    print('Test Accuracy:')
    model_ft.load_state_dict(torch.load("./images/model/efficientnet-b5.pth"))
    # criterion = nn.CrossEntropyLoss().cuda()
    test_model(model_ft)
  
