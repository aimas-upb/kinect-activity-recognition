import numpy as np
import torch
from copy import copy
from math import sqrt
from skimage import filters
import time


class MoveOriginToJoint(object):
    """ Move origin of skeletons in sequences to the selected (by index)
    joint """

    def __init__(self, origin_joint_index=0):
        self.origin_joint_index = origin_joint_index

    def __call__(self, sequence):
        for frame in sequence['frames']:
            for skeleton in frame:
                origin = copy(skeleton[self.origin_joint_index])
                for i in range(len(skeleton)):
                    x = skeleton[i][0] - origin[0]
                    y = skeleton[i][1] - origin[1]
                    z = skeleton[i][2] - origin[2]
                    skeleton[i] = [x, y, z]
        return sequence

    def __str__(self):
        return 'MoveOriginToJoint({})'.format(self.origin_joint_index)


class NormalizeSkeletonsByFrame(object):
    """ Normalize each skeleton in a sequence using frame-by-frame values """

    def __call__(self, sequence):
        for frame in sequence['frames']:
            for skeleton in frame:
                min_x = min(joint[0] for joint in skeleton)
                max_x = max(joint[0] for joint in skeleton)
                min_y = min(joint[1] for joint in skeleton)
                max_y = max(joint[1] for joint in skeleton)
                min_z = min(joint[2] for joint in skeleton)
                max_z = max(joint[2] for joint in skeleton)
                for joint in skeleton:
                    joint[0] -= min_x
                    joint[0] /= max_x - min_x
                    joint[1] -= min_y
                    joint[1] /= max_y - min_y
                    joint[2] -= min_z
                    joint[2] /= max_z - min_z
        return sequence

    def __str__(self):
        return "NormalizeSkeletonByFrame"


class NormalizeSkeletonsBySequence(object):
    """ Normalize each skeleton in a sequence using the whole sequence"""

    def __init__(self, factor=2.):
        self.factor = factor

    def __call__(self, sequence):
        frames = sequence['frames']
        frames = frames / self.factor
        return {'frames': frames,
                'tag': sequence['tag'],
                'length': sequence['length'],
                'filename': sequence['filename']}

    def __str__(self):
        return 'NormalizeSkeletonsBySequence'


class SelectFirstSkeleton(object):
    """ If multiple skeletons are present in the sequence, select the first
    one """

    def __call__(self, sequence, origin_joint_index=0):
        new_sequence = []
        for frame in sequence.get('frames'):
            if len(frame) > 1:
                new_sequence.append([frame[0]])
            else:
                new_sequence.append(frame)
        return {'frames': new_sequence,
                'tag': sequence['tag'],
                'length': sequence['length'],
                'filename': sequence['filename']}

    def __str__(self):
        return 'SelectFirstSkeleton'


class SelectMaxMotionEnergySkeleton(object):
    """ If multiple skeletons are present in the sequence, select the most
    active one """

    def __call__(self, sequence):
        frames = np.array(sequence['frames'])
        (num_frames, num_skeletons, num_joints, _) = frames.shape
        if num_skeletons == 1:
            return sequence
        energy = np.zeros((num_skeletons, num_joints, 3))
        for i in range(num_frames - 1):
            for j in range(num_skeletons):
                energy[j] = np.add(energy[j], np.abs(np.subtract(
                    frames[i + 1][j], frames[i][j])))
        energy = [np.linalg.norm(energy[i]) for i in range(num_skeletons)]
        max_energy = max(energy)
        skeleton_id = energy.index(max_energy)
        new_frames = []
        for i in range(num_frames):
            new_frames.append([sequence['frames'][i][skeleton_id]])
        return {'frames': new_frames,
                'tag': sequence['tag'],
                'length': sequence['length'],
                'filename': sequence['filename']}

    def __str__(self):
        return 'SelectMaxMotionEnergySkeleton'


class UniformSampleOrPad(object):
    """
    (i) If num_frames is greater than sample_size, select sample_size frames
    using random uniform generator ii) If num_frames is less than
    sample_size, frames are filled with zero-padding iii) If num_frames is
    equals to sample_size, return input sequence with new entry (length)
    """

    def __init__(self, sample_size):
        assert (sample_size > 0)
        self.sample_size = sample_size

    def __call__(self, sequence):
        frames, tag = sequence['frames'], sequence['tag']
        num_frames = len(frames)
        new_sequence = []
        if num_frames > self.sample_size:
            interval_size = num_frames / self.sample_size
            indices = []
            start = 0.0
            end = interval_size
            for i in range(self.sample_size):
                index = int(np.random.uniform(start, end))
                if indices and index == indices[-1]:
                    index += 1
                start = end
                end += interval_size
                new_sequence.append(frames[index])
                indices.append(index)
            num_frames = self.sample_size
        elif num_frames == self.sample_size:
            new_sequence = frames
        else:
            new_sequence = frames
            num_intervals = self.sample_size - num_frames
            for i in range(num_intervals):
                new_sequence.append([np.zeros(np.shape(frames[0][0]))])

        return {'frames': new_sequence,
                'tag': tag,
                'length': num_frames,
                'filename': sequence['filename']}

    def __str__(self):
        return 'UniformSampleOrPad({})'.format(self.sample_size)


class UniformSample(object):
    """
    (i) If num_frames is greater than sample_size, select sample_size frames
    using random uniform generator
    ii) If num_frames is less than sample_size, select first sample_size frames
    """

    def __init__(self, sample_size, use_batch=True, clip_ends_size=0):
        assert (sample_size > 0)
        self.sample_size = sample_size
        self.use_batch = use_batch
        self.clip_size = clip_ends_size

    def __call__(self, batch):
        frames = batch['frames']
        tag = batch['tag']
        length = batch['length']
        new_frames = []
        if self.use_batch:
            num_samples = len(frames)
            for idx in range(num_samples):
                num_frames = length[idx].numpy()
                if num_frames > self.sample_size:
                    interval_size = (num_frames - 2 * self.clip_size) / self.sample_size
                    indices = []
                    start = self.clip_size
                    end = interval_size
                    for i in range(self.sample_size):
                        index = int(np.random.uniform(start, end))
                        if indices and index == indices[-1]:
                            index += 1
                        start = end
                        end += interval_size
                        indices.append(index)
                    sample = torch.index_select(frames[idx], 0,
                                                torch.LongTensor(indices))
                    length[idx] = self.sample_size
                else:
                    sample = torch.index_select(frames[idx], 0,
                                                torch.LongTensor(
                                                    range(self.sample_size)))
                new_frames.append(sample)
            new_frames = torch.cat(new_frames).view(num_samples,
                                                    self.sample_size,
                                                    1, -1)
        else:
            num_frames = length
            if num_frames > self.sample_size:
                interval_size = num_frames / self.sample_size
                indices = []
                start = 0.0
                end = interval_size
                for i in range(self.sample_size):
                    index = int(np.random.uniform(start, end))
                    if indices and index == indices[-1]:
                        index += 1
                    start = end
                    end += interval_size
                    indices.append(index)
                new_frames = frames[indices]
                length = self.sample_size
            else:
                new_frames = frames[range(self.sample_size)]
                length = num_frames
        return {'frames': new_frames,
                'tag': tag,
                'length': length,
                'filename': batch['filename']}

    def __str__(self):
        return 'UniformSample({})'.format(self.sample_size)


class RearrangeSkeletonJoints(object):
    """ Modify order of the joints in the sequence using sequence of indices.
    Duplicates are allowed """

    def __init__(self, joints_map=None):
        self.joints_map = joints_map

    def __call__(self, sequence):
        frames = sequence['frames']
        new_frames = []
        num_frames = len(frames)
        num_skeletons = len(frames[0])
        num_joints = len(frames[0][0])
        if isinstance(self.joints_map, list) \
                and all(isinstance(item, int) for item in self.joints_map) \
                and all(item > 0 for item in self.joints_map) \
                and all(item <= num_joints for item in self.joints_map):
            for i in range(num_frames):
                frame = []
                for j in range(num_skeletons):
                    skeleton = []
                    for joint in self.joints_map:
                        skeleton.append(frames[i][j][joint])
                    frame.append(skeleton)
                new_frames.append(frame)
        else:
            new_frames = frames
        return {'frames': new_frames,
                'tag': sequence['tag'],
                'length': sequence['length'],
                'filename': sequence['filename']}

    def __str__(self):
        return 'RearrangeSkeletonJoints({})'.format(self.joints_map)


def euclidian_distance(p1, p2):
    (xa, ya, za) = p1
    (xb, yb, zb) = p2
    dist = sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2)
    return dist


class ToTensor(object):
    """ Convert sequence to Tensor """

    def __call__(self, sequence):
        frames = np.array(sequence['frames'])
        (dim1, dim2, dim3, dim4) = frames.shape
        return {
            'frames': torch.from_numpy(np.array(sequence['frames'])).view(dim1,
                                                                          dim2,
                                                                          dim3 * dim4),
            'tag': sequence['tag'],
            'length': sequence['length'],
            'filename': sequence['filename']}

    def __str__(self):
        return 'ToTensor'


class MovingPoseDescriptor(object):
    """
    Adding first derivatives and second derivatives.
    If normalize is True, then speed is scalar and otherwise is
    vector.
    """

    def __init__(self, batch_size=300):
        self.batch_size = batch_size

    def __call__(self, batch):
        start = time.time()
        print("Start MovingPoseDescriptor for " + batch['filename'])
        frames = batch['frames']
        joints = torch.split(frames, 3, 2)
        num_joints = len(joints)
        first_derivatives = torch.zeros((num_joints, *joints[0].size()))
        second_derivatives = torch.zeros((num_joints, *joints[0].size()))
        first_indices2 = torch.LongTensor([1] + list(range(2, self.batch_size)) + [self.batch_size - 1])
        first_indices3 = torch.LongTensor([1] + list(range(0, self.batch_size - 2)) + [self.batch_size - 1])
        second_indices1 = torch.LongTensor([1] + list(range(3, self.batch_size)) + [self.batch_size - 2,
                                                                                    self.batch_size - 1])
        second_indices2 = torch.LongTensor([1] + list(range(0, self.batch_size - 3)) + [self.batch_size - 2,
                                                                                        self.batch_size - 1])
        second_indices3 = torch.LongTensor([1] + list(range(1, self.batch_size - 2)) + [self.batch_size - 2,
                                                                                        self.batch_size - 1])
        for i in range(num_joints):
            first_derivatives[i] = joints[i][first_indices2, :, :] - joints[i][first_indices3, :, :]
            second_derivatives[i] = joints[i][second_indices1, :, :] + joints[i][second_indices2, :, :] - \
                                    2 * joints[i][second_indices3, :, :]
        old_frames = torch.cat(joints, 2)
        final_first_derivatives = torch.cat(first_derivatives, 2)
        final_second_derivatives = torch.cat(second_derivatives, 2)
        derivatives = torch.cat((final_first_derivatives, final_second_derivatives), 2)
        batch['frames'] = torch.cat((old_frames, derivatives.type(torch.DoubleTensor)), 2)
        end = time.time()
        print(end - start, "End MovingPoseDescriptor for " + batch['filename'])
        return batch

    def __str__(self):
        return "MovingPoseDescriptor"


class ResizeSkeletonSegments(object):
    """ Normalize the size of segments between adjacent joints based on values
    calculated using the the data in the training set. """

    def __init__(self):
        self.segments = [(1, 17), (17, 18), (18, 19), (19, 20), (1, 2), (2, 21),
                         (21, 9), (9, 10), (10, 11), (11, 12), (12, 24), (12, 25),
                         (21, 3), (3, 4), (21, 5), (5, 6), (6, 7), (7, 8), (8, 22),
                         (8, 23), (1, 13), (13, 14), (14, 15), (15, 16)]
        self.distances = {}
        joints = [(19, 20), (18, 19), (18, 17), (16, 15), (15, 14), (14, 13), (1, 13), (1, 17), (1, 2),
                  (2, 21), (5, 21), (9, 21), (3, 21), (3, 4), (5, 6), (6, 7), (8, 7), (8, 22), (8, 23),
                  (9, 10), (10, 11), (11, 12), (12, 25), (12, 24)]
        values = [0.11001726, 0.33970811, 0.33196634, 0.10967019, 0.3396472, 0.33246291,
                  0.06993549, 0.07035548, 0.27923619, 0.20789889, 0.15656522, 0.15577015,
                  0.06900109, 0.12490234, 0.23540237, 0.28492234, 0.06916196, 0.0578364,
                  0.04320888, 0.23292405, 0.22386113, 0.07093468, 0.04294151, 0.05803406]
        for i in range(len(values)):
            self.distances[joints[i]] = values[i]

    def get_distance(self, x, y):
        if (x, y) not in self.distances:
            return self.distances[(y, x)]
        return self.distances[(x, y)]

    def __call__(self, sequence):
        frames = sequence['frames']
        new_frames = []
        num_frames = len(frames)
        num_skeletons = len(frames[0])
        num_joints = len(frames[0][0])

        for i in range(num_frames):
            new_frame = []
            for j in range(num_skeletons):
                new_skeleton = np.zeros((num_joints, 3))
                new_skeleton[0] = frames[i][j][0]
                for (start, end) in self.segments:
                    dist = euclidian_distance(frames[i][j][start - 1], frames[i][j][end - 1])
                    r_dist = self.get_distance(start, end)
                    d = np.array(frames[i][j][end - 1]) - np.array(frames[i][j][start - 1])
                    if dist > 0:
                        new_d = r_dist * d / dist
                    else:
                        new_d = r_dist * d
                    new_skeleton[end - 1] = new_skeleton[start - 1] + new_d
                new_frame.append(new_skeleton)
            new_frames.append(new_frame)
        sequence['frames'] = new_frames
        return sequence

    def __str__(self):
        return "ResizeSkeletonSegments"


def smooth(x, window_len=5, window='hanning'):
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', "
                         "'bartlett', 'blackman'")
    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]


class SmoothingTimeSeries(object):

    def __init__(self, window='hamming', window_len=5):
        self.window = window
        self.window_len = window_len

    def __call__(self, sequence):
        frames = sequence['frames']
        new_frames = []
        num_skeletons = len(frames[0])
        num_joints = len(frames[0][0])
        coordinates = {}
        for i in range(num_skeletons):
            skeleton_coordinates = {}
            for j in range(num_joints):
                joint_coordinates_x = np.array([x[i][j][0] for x in frames])
                joint_coordinates_y = np.array([x[i][j][1] for x in frames])
                joint_coordinates_z = np.array([x[i][j][2] for x in frames])
                smoothed_joint_coordinates_x = smooth(joint_coordinates_x,
                                                      window_len=self.window_len,
                                                      window=self.window)
                smoothed_joint_coordinates_y = smooth(joint_coordinates_y,
                                                      window_len=self.window_len,
                                                      window=self.window)
                smoothed_joint_coordinates_z = smooth(joint_coordinates_z,
                                                      window_len=self.window_len,
                                                      window=self.window)
                skeleton_coordinates[j] = (smoothed_joint_coordinates_x,
                                           smoothed_joint_coordinates_y,
                                           smoothed_joint_coordinates_z)
            coordinates[i] = skeleton_coordinates
        for i in range(len(frames)):
            frame = []
            for j in range(num_skeletons):
                skeleton = []
                for k in range(num_joints):
                    skeleton.append([coordinates[j][k][0][i],
                                     coordinates[j][k][1][i],
                                     coordinates[j][k][2][i]])
                frame.append(skeleton)
            new_frames.append(frame)
        sequence['frames'] = new_frames
        return sequence

    def __str__(self):
        return "SmoothingTimeSeries"


class GaussianFilter(object):
    """ Apply a Gaussian filter using a 5-frame window """

    def __init__(self, sigma=0.8):
        self.sigma = sigma

    def __call__(self, sequence):
        frames = sequence['frames']
        new_frames = []
        num_skeletons = len(frames[0])
        num_joints = len(frames[0][0])
        num_frames = len(frames)
        for k in range(num_frames):
            if k < 2 or k >= num_frames - 2:
                new_frames.append(frames[k])
                continue
            frame = []
            for i in range(num_skeletons):
                skeleton = []
                for j in range(num_joints):
                    joint_coordinates = np.array([x[i][j] for x in frames[k - 2:k + 3]])
                    smoothed_joint_coordinates = filters.gaussian(joint_coordinates,
                                                                  sigma=self.sigma,
                                                                  multichannel=True)
                    skeleton.append(smoothed_joint_coordinates[2])
                frame.append(skeleton)
            new_frames.append(frame)
        sequence['frames'] = new_frames
        return sequence

    def __str__(self):
        return "GaussianFilter"


class MovingPoseDescriptorForBatch(object):
    """
    Adding first derivatives and second derivatives.
    If normalize is True, then velocity is scalar and otherwise is
    vector.
    """

    def __init__(self, batch_size=300):
        self.batch_size = batch_size

    def __call__(self, batch):
        frames = batch['frames']
        joints = torch.split(frames, 3, 3)
        num_joints = len(joints)
        first_derivatives = torch.zeros((num_joints, *joints[0].size()))
        second_derivatives = torch.zeros((num_joints, *joints[0].size()))
        first_indices2 = torch.LongTensor([1] + list(range(2, self.batch_size)) + [self.batch_size - 1])
        first_indices3 = torch.LongTensor([1] + list(range(0, self.batch_size - 2)) + [self.batch_size - 1])
        second_indices1 = torch.LongTensor(
            [1] + list(range(3, self.batch_size)) + [self.batch_size - 2, self.batch_size - 1])
        second_indices2 = torch.LongTensor(
            [1] + list(range(0, self.batch_size - 3)) + [self.batch_size - 2, self.batch_size - 1])
        second_indices3 = torch.LongTensor(
            [1] + list(range(1, self.batch_size - 2)) + [self.batch_size - 2, self.batch_size - 1])
        for i in range(num_joints):
            first_derivatives[i] = joints[i][:, first_indices2, :, :] - joints[i][:, first_indices3, :, :]
            second_derivatives[i] = joints[i][:, second_indices1, :, :] + joints[i][:, second_indices2, :, :] - 2 * \
                                    joints[i][:, second_indices3, :, :]
        old_frames = torch.cat(joints, 3)
        final_first_derivatives = torch.cat(first_derivatives, 3)
        final_second_derivatives = torch.cat(second_derivatives, 3)
        derivatives = torch.cat((final_first_derivatives, final_second_derivatives), 3)
        batch['frames'] = torch.cat((old_frames, derivatives.type(torch.DoubleTensor)), 3)
        return batch

    def __str__(self):
        return "MovingPoseDescriptorForBatch"
