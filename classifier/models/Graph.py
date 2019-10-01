import numpy as np
import graphTransforms as tools
from scipy.linalg import block_diag

num_node = 25
self_link = [(i, i) for i in range(num_node)]
# For 25 joints
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                   (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                   (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                   (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]

# For 17 joints
# inward_ori_index = [(1, 2), (2, 3), (4, 3), (5, 4), (6, 3), (7, 6), (8, 7),
#                     (9, 3), (10, 9), (11, 10), (12, 1),
#                     (13, 12), (14, 13), (15, 1), (16, 15), (17, 16)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph():
    """ The Graph to model the skeletons in NTU RGB+D
    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            dastance*: Distance Partitioning*
            dastance: Distance Partitioning
            spatial: Spatial Configuration
            DAD: normalized graph adjacency matrix
            DLD: normalized graph laplacian matrix
    For more information, please refer to the section 'Partition Strategies' in our paper.
    """

    def __init__(self):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode='uniform', batch_size=None):
        if labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance*':
            A = tools.get_uniform_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'distance':
            A = tools.get_distance_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'DAD':
            A = tools.get_DAD_graph(num_node, self_link, neighbor)
        elif labeling_mode == 'DLD':
            A = tools.get_DLD_graph(num_node, self_link, neighbor)
        # elif labeling_mode == 'customer_mode':
        #     pass
        else:
            raise ValueError()
        if batch_size is None:
            return A
        else:
             B = []
             for i in range(batch_size):
                 B.append(A)
             C = block_diag(*B)
             return C


def main():
    mode = ['uniform', 'distance*', 'distance', 'spatial', 'DAD', 'DLD']
    np.set_printoptions(threshold=np.nan)
    #for m in mode:
    #    print('=' * 10 + m + '=' * 10)
    #    print(Graph(m).get_adjacency_matrix())
    C = Graph().get_adjacency_matrix('uniform', 4)


if __name__ == '__main__':
    main()
