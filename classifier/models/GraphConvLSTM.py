import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import torch.nn as nn
import torch
from collections import OrderedDict
from Graph import Graph
from SimpleGCN import GCN
from torch.autograd import Variable

class GraphConvLSTM(nn.Module):
    def __init__(self, input_size, joint_size, hidden_dim, target_size, depth=3, sample_size=50, batch_size=32, no_of_joints=25):
        super(GraphConvLSTM, self).__init__()
        self.joint_size = joint_size
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.target_size = target_size
        self.no_of_joints = no_of_joints
        self.sample_size = sample_size
        self.depth = depth
        self.activation = nn.ReLU(inplace=True)
        self.gcn1 = GCN(joint_size * sample_size, 128, 64, 0.5)
        self.gcn2 = GCN(64, 256, 128, 0.5)
        self.gcn3 = GCN(128, 512, 256, 0.5)
        self.gcn4 = GCN(256, joint_size * sample_size, joint_size * sample_size, 0.5)

        self.lstm = nn.LSTM(input_size, hidden_dim, depth, batch_first=True)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.bnh1 = nn.BatchNorm1d(hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, target_size)
        self.graph = Graph().get_adjacency_matrix('DAD', batch_size)
        self.simple_graph = Graph().get_adjacency_matrix('DAD', 1)
        self.adj = Variable(torch.from_numpy(self.graph).type(torch.FloatTensor)).cuda()
        self.simple_adj = Variable(torch.from_numpy(self.simple_graph).type(torch.FloatTensor)).cuda()

    def forward(self, data, batch_len):

        # arrange data for convolution
        t = data.size()
        x = data.view(t[0], t[1], -1, 9)
        x = x[:, :, :, :3]
        y = x.transpose(1, 2).contiguous()
        z = y.view(t[0] * self.no_of_joints, -1)

        # apply convolutions
        if t[0] == 1:
            matrix = self.simple_adj
        else:
            matrix = self.adj

        x = z
        x = self.gcn1(x, matrix)
        x = self.gcn2(x, matrix)
        x = self.gcn3(x, matrix)
        x = self.gcn4(x, matrix)


        # reshape and apply linear
        x = x.view(t[0], self.no_of_joints, t[1], -1)
        y = x.transpose(2, 1).contiguous()

        x = y.view(t[0], t[1], -1)

        # pack and pass through LSTM
        packed_batch = torch.nn.utils.rnn.pack_padded_sequence(
            x, batch_len, batch_first=True)
        packed_lstm_out, hidden = self.lstm(packed_batch)
        unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out)

        idx_sel = torch.LongTensor(batch_len)

        # Get all act:
        feat_size = unpacked.size(2)
        all_act = []
        for i in range(idx_sel.size(0)):
            all_act.append(
                unpacked[:idx_sel[i], i, :].contiguous().view(-1, feat_size))
        all_act = torch.cat(all_act, dim=0)
        l1 = self.activation(self.bnh1(self.hidden1(all_act)))
        tag_space = self.hidden2(l1)

        # Sum act/batch ->
        sum_act = []
        idx = 0
        for i in range(idx_sel.size(0)):
            size = idx_sel[i] / 3
            first_idx = int(2 * size) + idx
            second_idx = idx_sel[i] + idx
            sum_batch = torch.mean(tag_space[first_idx:second_idx], dim=0)
            idx = idx + idx_sel[i]
            sum_act.append(sum_batch.unsqueeze(0))

        tag_space = torch.cat(sum_act)
        return tag_space
