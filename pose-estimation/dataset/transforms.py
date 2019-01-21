import numpy as np
import torch
import skimage
from torchvision import transforms
from skimage import io, transform
from PIL import Image, ImageOps, ImageEnhance
from torch import multiprocessing
import time
# from PIL import Image
import cv2 as cv
from copy import copy


def generate_heatmap(center, size):
    width = 480
    height = 270
    points = np.random.normal(center, scale=(1.5, 1.5, 0.5), size=(size, 3))
    # image = Image.new(mode='F', size=(width, height), color=0)
    image = np.zeros((height, width))
    for p in points:
        line = int(p[0])
        col = int(p[1])
        depth = p[2]
        image[line][col] += depth
        # image.putpixel((line, col), depth)
    # image = cv.resize(image, dsize=(240, 135), interpolation=cv.INTER_CUBIC)
    image = cv.resize(image, dsize=(120, 67), interpolation=cv.INTER_LINEAR)
    return image


class NormalizeImage(object):

    def __init__(self):
        self.max_depth = 5000
        self.max_color = 256

    def __call__(self, sample):
        image = sample['depth']
        image.div_(self.max_depth)
        rgb = sample['rgb']
        rgb.div_(self.max_color)
        return sample

    def __str__(self):
        return "NormalizeImage"


class ResizeImage(object):
    def __init__(self, resize_factor, size=1000):
        self.resize_factor = resize_factor
        self.size = size

    def __call__(self, sample):
        heatmaps = []
        idx = 0
        for joint in sample['skeleton'][0]:
            idx += 1
            y = joint[4]
            x = joint[3]
            depth = 1
            heatmap = generate_heatmap((y, x, depth), self.size)
            # heatmap = heatmap.resize((104, 82), Image.BILINEAR)
            heatmaps.append(np.asarray(heatmap))
        sample['heatmaps'] = heatmaps
        # sample['depth'] = sample['depth'].resize(self.resize_factor, Image.BILINEAR)
        return sample

    def __str__(self):
        return 'ResizeImage({})'.format(self.resize_factor)


class ToTensor(object):

    def __call__(self, sample):
        # image = np.array(sample['depth'])
        skeleton = np.array(sample['target'])
        heatmaps = np.array(sample['heatmaps'])
        # rgb = sample['rgb']
        data = sample['data']

        # return {'depth': torch.from_numpy(image).float(),
        #         'name': sample['name'],
        #         'index': sample['index'],
        #         'skeleton': torch.from_numpy(skeleton).float(),
        #         'heatmaps': torch.from_numpy(heatmaps).float(),
        #         'rgb': torch.from_numpy(rgb).float(),
        #         'data': torch.from_numpy(data).float()}

        return {'name': sample['name'],
                'index': sample['index'],
                'heatmaps': torch.from_numpy(heatmaps).float(),
                'data': torch.from_numpy(data).float(),
                'target': torch.from_numpy(skeleton).float()}

    def __str__(self):
        return 'ToTensor'


class MoveOriginToJoint(object):
    """ Move origin of skeletons in sequences to the selected (by index)
    joint """

    def __init__(self, origin_joint_index=0):
        self.origin_joint_index = origin_joint_index

    def __call__(self, sequence):
        skeleton = sequence['skeleton'][0]
        origin = copy(skeleton[self.origin_joint_index])
        for i in range(len(skeleton)):
            x = skeleton[i][0] - origin[0]
            y = skeleton[i][1] - origin[1]
            z = skeleton[i][2] - origin[2]
            skeleton[i] = [x, y, z]
        sequence['skeleton'] = skeleton
        return sequence

    def __str__(self):
        return 'MoveOriginToJoint({})'.format(self.origin_joint_index)


class ExtractRGB(object):

    def __init__(self, rgb_path):
        self.rgb_path = rgb_path

    def __call__(self, sample):
        name = sample['name']
        index = sample['index']
        video = cv.VideoCapture(self.rgb_path + '/' + name + '_rgb.avi')
        success, image = video.read()
        count = 1
        while success:
            if count == index:
                image = cv.resize(image, (480, 270))
                image_rgb = image
                break
            success, image = video.read()
            count += 1
        sample['rgb'] = image_rgb
        return sample


class AlignDepthRGB(object):

    def __init__(self, width = 480, height = 270, rgb_width=1920, rgb_height=1080, size=1500):
        self.width = width
        self.height = height
        self.rgb_width = rgb_width
        self.rgb_height = rgb_height
        self.size = size

    def __extract_coordinates(self, sample):
        self.pts_src = []
        self.pts_dst = []
        frames = sample['skeleton']
        skeleton = frames[0]
        for joint in skeleton:
            x = joint[3]
            y = joint[4]
            self.pts_src.append((x, y))
            x = joint[5]
            y = joint[6]
            x = (x / self.rgb_width) * self.width
            y = (y / self.rgb_height) * self.height
            self.pts_dst.append((x, y))
        self.pts_src = np.array(self.pts_src)
        self.pts_dst = np.array(self.pts_dst)
        h, status = cv.findHomography(self.pts_src, self.pts_dst)
        return h

    def __call__(self, sample):
        h = self.__extract_coordinates(sample)
        depth = sample['depth']
        rgb = sample['rgb']
        mean = np.load('mean.np.npy')
        depth_image = cv.warpPerspective(depth, h, (self.width, self.height), flags=cv.INTER_NEAREST)
        final_image = np.zeros((self.height, self.width, 4))
        final_image[:, :, 0:3] += rgb / mean
        final_image[:, :, 3] += depth_image / 5000
        sample['data'] = final_image
        heatmaps = []
        idx = 0
        points = []
        for (x, y) in self.pts_dst:
            idx += 1
            # y = joint[4]
            # x = joint[3]
            v = np.array([x, y, 1.0])
            new_v = np.matmul(h, v.transpose())
            y = new_v[1]
            x = new_v[0]
            depth = 1
            heatmap = generate_heatmap((y, x, depth), self.size)
            heatmaps.append(np.asarray(heatmap))
            points.append((y, x))
        sample['heatmaps'] = heatmaps
        sample['target'] = self.pts_dst
        return sample
