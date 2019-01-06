import sys
import os
import torch.utils.data.dataset as dataset
import json
import time
import torch
import numpy as np
import torch.multiprocessing
from enum import Enum


class DatasetMode(Enum):
    TRAIN = 1
    VALIDATION = 2


class NTUSkeletonDataset(dataset.Dataset):
    """NTU RGB+D dataset for 3D skeletons"""

    def __init__(self, raw_dir, transform=None, selected_actions=None,
                 selected_joints=None, use_cache=False, cache_dir=None,
                 pre_apply_transforms=True, use_validation=False,
                 validation_fraction=0.0, parallel_preprocessing=True,
                 preprocessing_threads=None):
        """
        :param raw_dir: Path to the directory containing the downloaded
                        NTU RGB+D dataset containing just skeleton data
        :param transform: Optional transform(s) to be applied on a sample
        :param selected_actions: List describing which actions are selected from
                        the dataset
        :param selected_joints: List describing which joints (by index) are
                        selected from each skeleton
        :param use_cache: Boolean indicating if the cache mechanism is used
        :param cache_dir: Path to the directory containing the cached dataset
        :param pre_apply_transforms: Boolean indicating if transformations
                        should be applied immediately after loading the dataset
                        (before saving to cache, if applicable)
        :param use_validation: Boolean indicating if the dataset should be split
                        in train and validation
        :param validation_fraction: Float from 0.0 (inclusive) to 1.0
                        (exclusive) indicating the proportion of samples to be
                        used for validation (the rest being kept for training)
        :param parallel_preprocessing: Boolean indicating if multiple threads
                        should be used
        :param preprocessing_threads: Integer representing the number of worker
                        threads used for preprocessing
        """

        self.raw_dir = raw_dir
        self.cache_dir = cache_dir
        self.transform = transform
        self.use_cache = use_cache
        self.pre_apply_transforms = pre_apply_transforms
        self.use_validation = use_validation
        self.validation_fraction = validation_fraction
        self.validation_mode = False
        self.use_mode = DatasetMode.TRAIN
        self.parallel_preprocessing = parallel_preprocessing
        self.transform = transform

        # Sanity check for the selected_actions
        if isinstance(selected_actions, list) \
                and all(isinstance(item, int) for item in selected_actions) \
                and all(item > 0 for item in selected_actions) \
                and all(item <= 60 for item in selected_actions):
            self.selected_actions = selected_actions
        else:
            # All the actions are selected
            self.selected_actions = range(1, 61)

        # Sanity check for the selected_joints
        if isinstance(selected_joints, list) \
                and all(isinstance(item, int) for item in selected_joints) \
                and all(item > 0 for item in selected_joints) \
                and all(item <= 25 for item in selected_joints):
            self.selected_joints = selected_joints
        else:
            # All the joints are selected
            self.selected_joints = range(1, 26)

        # Sanity check preprocessing
        if self.parallel_preprocessing:
            if preprocessing_threads is None:
                self.preprocessing_threads = torch.multiprocessing.cpu_count()
            elif preprocessing_threads <= 1:
                sys.exit('Thread count for parallel processing must be higher than 1')
            else:
                self.preprocessing_threads = preprocessing_threads

        # If cache is enabled, first try to load the dataset from cache
        if self.use_cache:
            self.cache_info = self.__load_cache_info()
            self.train_data = None
            self.validation_data = None
            # Search the cache info file for the correct entry
            for entry in self.cache_info:
                if entry['raw_dir'] == raw_dir and \
                                entry['selected_actions'] == str(
                            self.selected_actions) and \
                                entry['selected_joints'] == str(
                            self.selected_joints) and \
                                len(set(
                                    list(map(str, self.transform.transforms))).
                                        intersection(
                                    entry['transforms'])) == \
                                len(self.transform.transforms) and \
                                self.use_validation == entry[
                            'use_validation'] and \
                                self.validation_fraction == \
                                entry['validation_fraction']:
                    self.validation_data = torch.load(
                        entry['validation_filename'])
                    self.train_data = torch.load(entry['train_filename'])
                    print("Loading cache file:\n\t{}\n\t{}".format(
                        entry['train_filename'], entry['validation_filename']))
                    break
            # If no entry was found, we should create one and save the dataset
            # to cache after parsing it
            if not self.train_data:
                # First load the data
                self.train_data, self.validation_data = self.__parse_dir()

                # Cache info values (used for retrieval)
                cache_file_info = {
                    'train_filename': os.path.join(self.cache_dir,
                                                   "train_dataset_" +
                                                   time.strftime(
                                                       "%Y%m%d_%H%M%S") +
                                                   ".t7"),
                    'validation_filename': os.path.join(self.cache_dir,
                                                        "validation_dataset_"
                                                        + time.strftime(
                                                            "%Y%m%d_%H%M%S") +
                                                        ".t7"),
                    'raw_dir': self.raw_dir,
                    'selected_actions': str(self.selected_actions),
                    'selected_joints': str(self.selected_joints),
                    'transforms': [],
                    'use_validation': self.use_validation,
                    'validation_fraction': self.validation_fraction}

                # Check if transforms should be applied immediately after
                # loading the samples (useful when saving the dataset to cache)
                if pre_apply_transforms:
                    for tr in transform.transforms:
                        cache_file_info['transforms'].append(str(tr))
                    cache_file_info['transforms'].sort()
                    processed_train_data = []
                    processed_validation_data = []

                    if not self.parallel_preprocessing:
                        # Single threaded preprocessing
                        for sample in self.train_data:
                            processed_train_data.append(transform(sample))
                        for sample in self.validation_data:
                            processed_validation_data.append(transform(sample))
                    else:
                        # Parallel preprocessing
                        train_pool = torch.multiprocessing.Pool(processes=self.preprocessing_threads)
                        # Measure data loading time for training
                        start = time.time()
                        print("Start parallel preprocessing for train data")
                        train_chunk = int(len(self.train_data) / self.preprocessing_threads)
                        processed_train_data = train_pool.map(transform, self.train_data, chunksize=train_chunk)
                        train_pool.close()
                        train_pool.join()
                        # Measure elapsed time for training
                        end = time.time()
                        print(end - start, "Finished parallel preprocessing for train data")

                        # Measure data loading time for validation
                        start = time.time()
                        validation_pool = torch.multiprocessing.Pool(processes=self.preprocessing_threads)
                        print("Start parallel preprocessing for validation data")
                        validation_chunk = int(len(self.validation_data) / self.preprocessing_threads)
                        processed_validation_data = validation_pool.map(transform, self.validation_data,
                                                                        chunksize=validation_chunk)
                        validation_pool.close()
                        validation_pool.join()
                        # Measure elapsed time for validation
                        end = time.time()
                        print(end - start, "Finished parallel preprocessing for validation data")

                    self.train_data = processed_train_data
                    self.validation_data = processed_validation_data

                self.cache_info.append(cache_file_info)
                with open(os.path.join(self.cache_dir, 'cache.json'),
                          "w") as fp:
                    json.dump(self.cache_info, fp)

                torch.save(self.train_data, cache_file_info['train_filename'])
                torch.save(self.validation_data,
                           cache_file_info['validation_filename'])
        else:
            # If cache is not enabled, just load the data
            self.train_data, self.validation_data = self.__parse_dir()

    def __len__(self):
        if self.use_validation and self.use_mode == DatasetMode.VALIDATION:
            return len(self.validation_data)
        else:
            return len(self.train_data)

    def __getitem__(self, index):
        if self.use_validation and self.use_mode == DatasetMode.VALIDATION:
            sample = self.validation_data[index]
        else:
            sample = self.train_data[index]

        if self.transform and not self.pre_apply_transforms:
            sample = self.transform(sample)

        return sample

    def set_use_mode(self, mode):
        """Uses DatasetMode enum values to switch between training and
        validation mode of usage. For example, if set in validation mode the
        __getitem__ will return samples from the validation subset."""
        self.use_mode = mode

    @staticmethod
    def __select_skeleton(frame, prev_frame):
        """Matches the closest skeleton in frame to the skeleton in prev_frame.
        This assumes that prev_frame contains only one skeleton."""
        differences = []
        for skeleton in frame:
            differences.append(np.linalg.norm(np.subtract(skeleton,
                                                          prev_frame)))
        return np.argmin(differences)

    def __split_frames(self, frames, tag):
        """Filter (some) incorrect skeletons for single user action sequences.
        Create two separate sequences for multi-user action sequences."""
        if tag < 50:
            # Single user action
            num_frames = len(frames)
            new_frames = []
            for idx in range(num_frames):
                frame = frames[idx]
                if len(frame) == 1:
                    new_frames.append(frame)
                else:
                    if idx != 0:
                        frame = [frame[
                                     self.__select_skeleton(frame,
                                                            new_frames[
                                                                idx - 1])]]
                    else:
                        frame = [frame[0]]
                    new_frames.append(frame)
            final_frames = [new_frames]
        else:
            # Multi user action
            new_frames1 = []
            new_frames2 = []
            num_frames = len(frames)
            for idx in range(num_frames):
                frame = frames[idx]
                if len(frame) >= 2:
                    if idx == 0:
                        new_frames1.append([frame[0]])
                        new_frames2.append([frame[1]])
                    else:
                        if not new_frames2:
                            index = self.__select_skeleton(frame,
                                                           new_frames1[-1])
                            new_frames1.append([frame[index]])
                            del frame[index]
                            new_frames2.append([frame[0]])
                        else:
                            frame1 = [frame[self.__select_skeleton(frame,
                                                                   new_frames1[
                                                                       -1])]]
                            frame2 = [frame[self.__select_skeleton(frame,
                                                                   new_frames2[
                                                                       -1])]]
                            new_frames1.append(frame1)
                            new_frames2.append(frame2)
                else:
                    if idx == 0:
                        new_frames1.append(frame)
                    else:
                        if new_frames1 and new_frames2:
                            index = self.__select_skeleton(
                                [new_frames1[-1], new_frames2[-1]], frame)
                            if index == 0:
                                new_frames1.append(frame)
                            else:
                                new_frames2.append(frame)
                        else:
                            frame1 = [frame[self.__select_skeleton(frame,
                                                                   new_frames1[
                                                                       -1])]]
                            new_frames1.append(frame1)
            if new_frames2:
                final_frames = [new_frames1, new_frames2]
            else:
                final_frames = [new_frames1]
        return final_frames

    def __parse_file(self, file):
        frames = []
        with open(file) as f:
            num_frames = int(f.readline())
            for i in range(num_frames):
                num_skeletons = int(f.readline())
                skeletons = []
                if num_skeletons == 0:
                    continue
                for j in range(num_skeletons):
                    # Skip a line
                    f.readline()
                    num_joints = int(f.readline())
                    skeleton = []
                    for k in range(num_joints):
                        pose = f.readline()
                        # Skip unwanted joints
                        if k + 1 not in self.selected_joints:
                            continue
                        values = pose.split(' ')
                        skeleton.append([float(values[0]), float(values[1]),
                                         float(values[2])])
                    skeletons.append(skeleton)
                frames.append(skeletons)
            f.close()
        tag = int(file.replace('.skeleton', '').split('/')[-1].split('A')[1])
        frames = self.__split_frames(frames, tag)
        samples = []
        for sample in frames:
            samples.append({'frames': sample,
                            'tag': self.selected_actions.index(tag),
                            'length': len(frames),
                            'filename': file})
        return samples

    def parse_file(self, file):
        return self.__parse_file(file)

    def __parse_dir(self):
        assert (os.path.isdir(self.raw_dir))
        validation_data = []
        train_data = []
        samples_map = {}
        samples_size = 0
        files = {}
        for idx in range(len(self.selected_actions)):
            samples_map[idx] = []
            if self.parallel_preprocessing:
                files[idx] = []

        # Load each valid sequence and save each filename to the samples_map
        # according to the activity class
        for file in os.listdir(self.raw_dir):
            tag = int(file.replace('.skeleton', '').split('A')[1])
            if tag not in self.selected_actions:
                continue
            if self.parallel_preprocessing:
                files[self.selected_actions.index(tag)].append(os.path.join(self.raw_dir, file))
            else:
                samples = self.__parse_file(os.path.join(self.raw_dir, file))
                samples_map[self.selected_actions.index(tag)] += samples
            samples_size += 1

        if self.parallel_preprocessing:
            pool = torch.multiprocessing.Pool(processes=self.preprocessing_threads)
            for idx in files:
                samples_list = pool.map(self.parse_file, files[idx])
                for sample in samples_list:
                    samples_map[idx] += sample
            pool.close()
            pool.join()
        # When using validation, use the samples_map to split up the dataset
        # into train and validation subsets
        if self.use_validation:
            assert (0.0 <= self.validation_fraction <= 1.0)
            validation_size = int(self.validation_fraction * samples_size)
            validation_per_class = int(
                validation_size / len(self.selected_actions))
            for idx in range(len(self.selected_actions)):
                validation_indices = np.random.randint(low=0, high=len(
                    samples_map[idx]), size=validation_per_class)
                validation_data += [samples_map[idx][i] for i in
                                    validation_indices]
                train_data += [samples_map[idx][i] for i in
                               range(len(samples_map[idx])) if
                               i not in validation_indices]
        else:
            for idx in range(len(self.selected_actions)):
                train_data += samples_map[idx]

        return train_data, validation_data

    def __load_cache_info(self):
        assert (os.path.isfile(os.path.join(self.cache_dir, 'cache.json')))
        filename = os.path.join(self.cache_dir, 'cache.json')
        with open(filename, 'r') as fp:
            cache_info = json.load(fp)
        return cache_info
