import torch
import os
import importlib
import pycocotools.mask
import numpy as np
import cv2
from ruamel.yaml import YAML
import time
import base64

# from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import *

from lednet import Net

from transform import Relabel, ToLabel, colormap_cityscapes, Colorize

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

NUM_CLASSES = 20 # 类别数目0-18 和 空类型

# yaml的输出格式
inp = """\
#%YAML:1.0
instance:
confidence: 
"""

# 输入图像的尺寸变换
input_transform_cityscapes = Compose([
    Resize((512,1024),Image.BILINEAR),
    ToTensor(),
])

# 处理得到的像素类别信息
class PostProcess:

    def __init__(self, n=20, w=1024, h=512):

        if n < 20:
            n = 20
            print("too little labels and set n = 20")
        self.cmap = colormap_cityscapes(n)  # 构造类别颜色映射
        self.width = w          # 输入的凸显大小
        self.height = h         


    def __call__(self, object):

        output_unit = torch.nn.functional.softmax(object[0], dim=0)  # 使用softmax处理置信度到0-1之间
        preds = output_unit.max(0) # 找到每个像素最可能的类别

        confidence = preds[0].cpu().numpy()     # [0]置信度值，转numpy，下面都在numpy上处理
        label = preds[1].byte().cpu().numpy()   # [1]标签值，转numpy

        # label_resize = cv2.resize(label, (self.width, self.height), cv2.INTER_NEAREST)  # 会增加不存在的类别

        size = label.shape # 1024*512
        color_image = np.zeros([self.height, self.width, 3], np.uint8)      # 彩色图片
        confid_image = np.zeros([self.height, self.width, 3], np.uint8)
        label_resize = np.zeros([self.height, self.width, 1], np.uint8)*19  # add void class

        instance = []

        # for label in range(1, len(self.cmap)):
        for label_index in range(0, len(self.cmap)-1):
            # bool
            mask = label == label_index

            # segmentation
            mask_src = np.zeros([size[0], size[1], 1], np.uint8)
            mask_src[mask] = 1;
            mask_resize = cv2.resize(mask_src, (self.width, self.height), cv2.INTER_AREA)

            # cv2.imshow("mask", mask_resize*255)
            # cv2.waitKey(0)

            if((mask_resize == 0).all()): # 无该类别
                continue
            
            # COCO mask编码
            rle = pycocotools.mask.encode(
                np.asfortranarray(mask_resize.astype(np.uint8))
            )
            rle['counts'] = rle['counts'].decode('ascii')

            # mask_decode = pycocotools.mask.decode(rle)
            # cv2.imshow("decode", mask_decode*255)
            # cv2.waitKey(0)

            # bbox
            mask_loc = np.where(mask_resize == 1)

            left = min(mask_loc[0])     # y
            top = min(mask_loc[1])      # x
            right = max(mask_loc[0])    # y
            down = max(mask_loc[1])     # x
            bbox = [int(left), int(top), int(right-left), int(down-top)]


            mask_resize = mask_resize == 1
            label_resize[mask_resize] = label_index
            
            # rgb
            color_image[:,:,0][mask_resize] = self.cmap[label_index][2]
            color_image[:,:,1][mask_resize] = self.cmap[label_index][1]
            color_image[:,:,2][mask_resize] = self.cmap[label_index][0]
            
            # 添加进去
            instance.append({
                'category_id': int(label_index),
                'box': bbox,
                'segmentation': rle,
            })

        # whole imagel
        # after resize will appear some zero pixel!!!
        bbox = [0, 0, self.width, self.height]  # 图像大小
        img_encode = cv2.imencode('.png', label_resize)[1]  # 图像，opencv编码
        # str_encode = np.array(img_encode).tostring()
        str_encode = str(base64.b64encode(img_encode))[2:-1]    # base64编码
        
        # 整幅图像编码
        instance.append({
            'category_id': int(-1),
            'box':  bbox,
            'segmentation': str_encode,
        })

        # score
        confidence_uint8 = confidence*255
        confidence_uint8 = confidence_uint8.astype(np.uint8)
        confidence_resize = cv2.resize(confidence_uint8, (self.width, self.height), cv2.INTER_AREA)
        # 编码
        img_encode = cv2.imencode('.png', confidence_resize)[1]
        # rle_confi = np.array(img_encode).tostring()
        rle_confi = str(base64.b64encode(img_encode))[2:-1]

        # image
        confid_image[:, :, 0] = confidence_resize
        confid_image[:, :, 1] = confidence_resize
        confid_image[:, :, 2] = confidence_resize

        return instance, rle_confi, confid_image, color_image


def main(args):
    weightspath = args.loadDir + args.loadWeights

    print("Loading weights: " + weightspath)

    model = Net(NUM_CLASSES)

    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    def load_my_state_dict(model, state_dict):  # custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath)) # 加载权重
    print("Model and weights LOADED successfully")

    model.eval()

    if (not os.path.exists(args.datadir)):
        print("Error: datadir could not be loaded")
    
    # 读取数据集目录
    loader = DataLoader(
        KITTI(args.datadir, input_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it

    for step, (images, filename) in enumerate(loader): # 迭代图片

        time_start = time.clock()

        if (not args.cpu):
            images = images.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs) # output 是[1,20,512,1024], 对应于20个类别的置信度，和图像大小

        time_elapsed = (time.clock() - time_start)


        time_start = time.clock()

        instance, confidence, confidence_color, label_color = PostProcess(20, 1226, 370)(outputs)  # 太费时了这个， 需要简化

        time_process = (time.clock() - time_start)

        # out_json = {"instance":instance,
        #        "confidence":confidence
        # }

        yaml = YAML()
        code = yaml.load(inp)
        code['instance'] = instance         # 改变inp格式
        code['confidence'] = confidence
        yaml.indent(mapping=6, sequence=4, offset=2)    # 设置yaml空格格式

        # 设置保存路径名字
        filenameSave = "/home/gongyiqun/project/output/08/" + filename[0].split(args.subset)[1]
        filenameSave_bel = filenameSave.split(".png")[0]+"_bel.png"
        filenameSave_inf = filenameSave.split(".png")[0]+".yaml"
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)

        with open(filenameSave_inf, 'w') as f:
            yaml.dump(code, f)
            # json.dump(out, f)



        # label_save = ToPILImage()(label_color)
        # label_save = label_save.resize((1242, 375), Image.BILINEAR)  # For KITTI only
        # label_save.save(filenameSave)
        # cv2.imwrite(filenameSave, label_color)
        # cv2.imwrite(filenameSave_bel, confidence_color)


        print(step, filenameSave, time_elapsed, time_process)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir', default="../save/logs(KITTI)/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="lednet.py")
    parser.add_argument('--subset', default="image_2")  # can be val, test, train, demoSequence

    parser.add_argument('--datadir', default="/home/common/gongyiqun/kitti/color/color/08/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
