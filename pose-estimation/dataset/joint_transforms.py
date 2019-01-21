import numpy as np
import torch
from torchvision import transforms
from copy import copy


class ToTensor(object):

    def __call__(self, sample):
        return torch.from_numpy(np.array(sample)).float()

    def __str__(self):
        return 'ToTensor'


class MoveOriginToJoint(object):
    """ Move origin of skeletons in sequences to the selected (by index)
    joint """

    def __init__(self, origin_joint_index=0):
        self.origin_joint_index = origin_joint_index

    def __call__(self, sample):
        for skeletons in sample:
            for skeleton in skeletons:
                origin = copy(skeleton[self.origin_joint_index])
                for i in range(len(skeleton)):
                    x = skeleton[i][0] - origin[0]
                    y = skeleton[i][1] - origin[1]
                    z = skeleton[i][2] - origin[2]
                    skeleton[i] = [x, y, z]
        return sample

    def __str__(self):
        return 'MoveOriginToJoint({})'.format(self.origin_joint_index)
