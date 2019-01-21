from torch.utils.data.dataset import Dataset
import cv2
from enum import Enum
import os
from random import choice
import torch
import numpy as np
import lintel
from collections import namedtuple
import time
import inspect
import random


class DatasetMode(Enum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3


class VideoDataset(Dataset):

    def __init__(self, rgb_directory, skeleton_directory, image_transforms=None,
                 joint_transforms=None, cross_subject=False, use_validation=True, windows_size=10, validation_fraction=0.1):
        self.rgb_directory = rgb_directory
        self.skeleton_directory = skeleton_directory
        self.validation_fraction = validation_fraction
        self.image_transforms = image_transforms
        self.joint_transforms = joint_transforms
        self.use_validation = use_validation
        self.windows_size = windows_size

        self.train = []
        self.test = []
        self.validation = []
        self.mode = DatasetMode.TRAIN

        if cross_subject:
            self.__cross_subject()
        else:
            self.__cross_view()

        if use_validation:
            size = int(len(self.train) * self.validation_fraction)
            for _ in range(size):
                item = choice(self.train)
                self.validation.append(item)
                self.train.remove(item)

    def __cross_subject(self):
        assert (os.path.isdir(self.rgb_directory))

        train_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        for sample in os.listdir(self.rgb_directory):
            action = int(sample.split('/')[-1].split('A')[1][:3])
            if action >= 50:
                continue
            subject = int(sample.split('/')[-1].split('P')[1][:3])
            sample_name = sample.split('/')[-1].split('_')[0]
            if subject in train_subjects:
                self.train.append(sample_name)
            else:
                self.test.append(sample_name)

    def __cross_view(self):
        assert (os.path.isdir(self.rgb_directory))

        train_cameras = [2, 3]
        for sample in os.listdir(self.rgb_directory):
            action = int(sample.split('/')[-1].split('A')[1][:3])
            if action >= 50:
                continue
            camera = int(sample.split('/')[-1].split('C')[1][:3])
            sample_name = sample.split('/')[-1].split('_')[0]
            if camera in train_cameras:
                self.train.append(sample_name)
            else:
                self.test.append(sample_name)

    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == DatasetMode.VALIDATION:
            sample = self.validation[index]
        elif self.mode == DatasetMode.TRAIN:
            sample = self.train[index]
        else:
            sample = self.test[index]
        skeletons, num_frames = self.__read_skeletons(sample)
        indices = random.sample(range(1, num_frames-1), self.windows_size)
        indices.sort()
        frames = self.__read_video_with_lintel(sample, indices)
        skeletons = skeletons[indices]
        result = {'sample': sample, 'skeletons': skeletons[:, 0, :, :], 'frames': frames}
        return result

    def __len__(self):
        if self.mode == DatasetMode.VALIDATION:
            return len(self.validation)
        elif self.mode == DatasetMode.TRAIN:
            return len(self.train)
        else:
            return len(self.test)

    def __read_skeletons(self, sample_name):
        file = self.skeleton_directory + '/' + sample_name + '.skeleton'
        frames = []
        correct_num_frames = 0
        with open(file) as f:
            num_frames = int(f.readline())
            for i in range(num_frames):
                num_skeletons = int(f.readline())
                skeletons = []
                if num_skeletons == 0:
                    continue
                correct_num_frames += 1
                for j in range(num_skeletons):
                    f.readline()  # skip a line
                    num_joints = int(f.readline())
                    skeleton = []
                    for k in range(num_joints):
                        pose = f.readline()
                        values = pose.split(' ')
                        skeleton.append([float(values[0]), float(values[1]), float(values[2])])
                    skeletons.append(skeleton)
                frames.append(skeletons)
            f.close()
        if self.joint_transforms:
            frames = self.joint_transforms(frames)
        return frames, correct_num_frames

    def __read_video(self, sample_name):
        frames = []
        file = self.rgb_directory + '/' + sample_name + '_rgb.avi'
        cap = cv2.VideoCapture(file)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.num_frames = num_frames
        for i in range(int(num_frames)):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (270, 480), interpolation=cv2.INTER_AREA)
            if self.image_transforms:
                frame = self.image_transforms(frame)
            frames.append(frame)
        skeletons = self.__read_skeletons(sample_name)
        return {'frames': frames, 'skeletons': skeletons}

    def __read_video_with_lintel(self, sample_name, indices=None):
        file = self.rgb_directory + '/' + sample_name + '_rgb.avi'
        fin = open(file, 'rb')
        video = fin.read()
        Dataset = namedtuple('Dataset', 'width height num_frames')
        dataset = Dataset(1920, 1080, None)
        if indices:
            video = lintel.loadvid_frame_nums(video,
                                              frame_nums=indices,
                                              width=dataset.width,
                                              height=dataset.height)
        else:
            video, seek_distance = lintel.loadvid(
                video,
                should_random_seek=True,
                width=dataset.width,
                height=dataset.height)
        video = np.frombuffer(video, dtype=np.uint8)
        video = np.reshape(
            video, newshape=(-1, dataset.height, dataset.width, 3))
        fin.close()
        result = []
        if self.image_transforms:
            for i in range(len(video)):
                result.append(self.image_transforms(video[i]))
        return torch.stack(result)
