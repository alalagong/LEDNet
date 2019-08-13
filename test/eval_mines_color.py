import numpy as np
import torch
import os
import cv2
import importlib
from dataset import *
from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes

from lednet import Net

from transform import Relabel, ToLabel, Colorize

import visdom

NUM_CHANNELS = 3
NUM_CLASSES = 20

#* *******************测试单张图片****************************

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
    # Normalize([.485, .456, .406], [.229, .224, .225]),
])



def main(args):
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = Net(NUM_CLASSES)

    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    # model.load_state_dict(torch.load(args.state))
    # model.load_state_dict(torch.load(weightspath)) #not working if missing key

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print("Model and weights LOADED successfully")

    model.eval()

    if (not os.path.exists(args.datadir)):
        print("Error: datadir could not be loaded")

    # loader = DataLoader(
    #     cityscapes('/home/liqi/PycharmProjects/LEDNet/4.png', input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
    #     num_workers=args.num_workers, batch_size=1 ,shuffle=False)
    input_transform_cityscapes = Compose([
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    name ="4.png"
    with open(image_path_city('/home/gongyiqun/images', name), 'rb') as f:
        images = load_image(f).convert('RGB')

        images = input_transform_cityscapes(images)
    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if (args.visualize):
        vis = visdom.Visdom()

    if (not args.cpu):
        images = images.cuda()
        # labels = labels.cuda()
    a=torch.unsqueeze(images,0)
    inputs = Variable(a)
    # targets = Variable(labels)
    with torch.no_grad():
        outputs = model(inputs)

    label = outputs[0].max(0)[1].byte().cpu().data
    # label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
    label_color = Colorize()(label.unsqueeze(0))


    filenameSave = "./save_color/"+"Others/"+name
    os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
    # image_transform(label.byte()).save(filenameSave)

    label_save = ToPILImage()(label_color)
    label_save = label_save.resize((1241, 376), Image.BILINEAR)
    # label_save = cv2.resize(label_save, (376, 1224),interpolation=cv2.INTER_AREA)
    label_save.save(filenameSave)

    if (args.visualize):
        vis.image(label_color.numpy())
    # print(step, filenameSave)

    # for step, (images, labels, filename, filenameGt) in enumerate(loader):


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="../save/logs(KITTI)/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="lednet.py")
    parser.add_argument('--subset', default="val")  # can be val, test, train, demoSequence

    parser.add_argument('--datadir', default="")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
