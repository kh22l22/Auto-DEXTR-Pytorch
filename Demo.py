

import argparse
from PIL import Image
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
from nlfd import build_model
#EXT
import os
import torch
from collections import OrderedDict
from matplotlib import pyplot as plt

from torch.nn.functional import upsample

import networks.deeplab_resnet as resnet
from mypath import Path
from dataloaders import helpers as helpers
from skimage.transform import resize

import cv2

#Ext Model
modelName = 'dextr_pascal-sbd'
pad = 50
thres = 0.9
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

img_path = 'Messi.jpg'
out_path = 'Out.png'

#  Create the network and load the weights
net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
print("Initializing weights from: {}".format(os.path.join(Path.models_dir(), modelName + '.pth')))
state_dict_checkpoint = torch.load(os.path.join(Path.models_dir(), modelName + '.pth'),
                                   map_location=lambda storage, loc: storage)
# Remove the prefix .module from the model when it is trained using DataParallel
if 'module.' in list(state_dict_checkpoint.keys())[0]:
    new_state_dict = OrderedDict()
    for k, v in state_dict_checkpoint.items():
        name = k[7:]  # remove `module.` from multi-gpu training
        new_state_dict[name] = v
else:
    new_state_dict = state_dict_checkpoint
net.load_state_dict(new_state_dict)
net.eval()
net.to(device)

#  Read image and click the points
#image = np.array(Image.open('ims/MSRA10K_Imgs_GT/Imgs/124.jpg'))
image = np.array(Image.open(img_path))
plt.ion()
plt.axis('off')
plt.imshow(image)
plt.title('Click the four extreme points of the objects\nHit enter when done (do not close the window)')

results = []

###################################################################################################################
## NLFD

def NLFD(model_path, img_path, cuda):
    transform = transforms.Compose([transforms.Resize((352, 352)), transforms.ToTensor()])
    img = Image.open(img_path)
    shape = img.size
    ori_H = shape[0]
    ori_W = shape[1]
    img = transform(img) - torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255
    img = Variable(img.unsqueeze(0), volatile=True)
    net = build_model()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    if cuda: img, net = img.cuda(), net.cuda()
    prob = net(img)
    prob = (prob.cpu().data[0][0].numpy() * 255).astype(np.uint8)

    # kernel = np.ones((5, 5), np.uint8)
    # prob = cv2.dilate(prob, kernel, iterations=1)
    #print (prob)
    #prob = cv2.morphologyEx(prob, cv2.MORPH_CLOSE, kernel)
    # p_img = Image.fromarray(prob, mode='L').resize(shape)
    # p_img.show()

    prob = resize(prob, (ori_W, ori_H), order=1)
    #prob = cv2.cvtColor(prob.copy(), cv2.COLOR_GRAY2BGR)
    return prob
###################################################################################################################

#NLFD Model
model_path = './weights/best.pth'

parser = argparse.ArgumentParser()

parser.add_argument('--demo_img', type=str, default=img_path)
parser.add_argument('--trained_model', type=str, default=model_path)
parser.add_argument('--cuda', type=bool, default=True)
config = parser.parse_args()
ext = ['.jpg', '.png']
if not os.path.splitext(config.demo_img)[-1] in ext:
    raise IOError('illegal image path')

prob = NLFD(config.trained_model, config.demo_img, config.cuda)

prob = prob*255
prob = (prob).astype(np.uint8)


###################################################################################################################

# Extreme Cut
with torch.no_grad():
    while 1:

        extreme_points_ori_ = np.array(plt.ginput(1, timeout=0)).astype(np.int)
        ##without NLFD ########################################################
        #ext_point_arr = helpers.calc_extreme_points(image)
        ##with NLFD ###########################################################
        ext_point_arr = helpers.calc_extreme_points_NLFD(image, prob)
        extreme_points_ori = np.array(ext_point_arr)
        print (extreme_points_ori)

        #  Crop image to the bounding box from the extreme points and resize
        bbox = helpers.get_bbox(image, points=extreme_points_ori, pad=pad, zero_pad=True)
        crop_image = helpers.crop_from_bbox(image, bbox, zero_pad=True)
        resize_image = helpers.fixed_resize(crop_image, (512, 512)).astype(np.float32)

        #  Generate extreme point heat map normalized to image values
        extreme_points = extreme_points_ori - [np.min(extreme_points_ori[:, 0]), np.min(extreme_points_ori[:, 1])] + [pad,
                                                                                                                      pad]
        extreme_points = (512 * extreme_points * [1 / crop_image.shape[1], 1 / crop_image.shape[0]]).astype(np.int)
        extreme_heatmap = helpers.make_gt(resize_image, extreme_points, sigma=10)
        extreme_heatmap = helpers.cstm_normalize(extreme_heatmap, 255)

        #  Concatenate inputs and convert to tensor
        input_dextr = np.concatenate((resize_image, extreme_heatmap[:, :, np.newaxis]), axis=2)
        inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[np.newaxis, ...])

        # Run a forward pass
        inputs = inputs.to(device)
        outputs = net.forward(inputs)
        outputs = upsample(outputs, size=(512, 512), mode='bilinear', align_corners=True)
        outputs = outputs.to(torch.device('cpu'))

        pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        result = helpers.crop2fullmask(pred, bbox, im_size=image.shape[:2], zero_pad=True, relax=pad) > thres

        results.append(result)

        # Plot the results
        plt.imshow(helpers.overlay_masks(image / 255, results))
        #plt.plot(extreme_points_ori[:, 0], extreme_points_ori[:, 1], 'ro')

        plt.plot(extreme_points_ori[0, 0], extreme_points_ori[0, 1], 'ro')
        plt.plot(extreme_points_ori[1, 0], extreme_points_ori[1, 1], 'go')
        plt.plot(extreme_points_ori[2, 0], extreme_points_ori[2, 1], 'bo')
        plt.plot(extreme_points_ori[3, 0], extreme_points_ori[3, 1], 'yo')
