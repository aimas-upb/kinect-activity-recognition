import torch.nn as nn
import torch
from collections import OrderedDict


class ConvLSTM(nn.Module):
    """ 2D Arrangement CNN + LSTM """

    def __init__(self, input_size, joint_size, hidden_dim, target_size, kernel_size = 3, padding = 1, depth=3):
        super(ConvLSTM, self).__init__()
        self.joints = [23, 11, 3, 7, 21, 24, 10, 2, 6, 22, 9, 8, 20, 4, 5, 18,
                       17, 1, 13, 14, 19, 16, 0, 12, 15]
        self.joint_size = joint_size
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.target_size = target_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.depth = depth
        self.activation = nn.ReLU(inplace=True)
        self.first_bn = nn.BatchNorm2d(32)
        self.second_bn = nn.BatchNorm2d(hidden_dim)
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=32,
                              kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(32, out_channels=hidden_dim,
                               kernel_size=kernel_size, padding=padding)
        self.fc = nn.Linear(hidden_dim * 3 * 3, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + input_size, hidden_dim, depth, batch_first=True)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.bnh1 = nn.BatchNorm2d(hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, target_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, data, batch_len):
        (num_batch, sample_size, input_size) = data.size()

        # arrange data for convolution
        t = data.size()
        x = data.view(t[0], t[1], -1, self.joint_size)
        x = x[:, :, self.joints, :]
        x = x.view(-1, self.joint_size, 5, 5)

        # apply convolutions
        x = self.activation(self.first_bn(self.conv1(x)))
        x = self.activation(self.second_bn(self.conv2(x)))

        # reshape and apply linear
        x = x.view(-1, self.hidden_dim * 3 * 3)
        x = self.fc(x)

        # reshape and pack
        x = x.view(t[0], t[1], -1)

        # pack and pass through LSTM
        x = self.dropout(x)
        x = torch.cat([x, data], dim=2)
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
        final_tag_space = self.dropout(tag_space)
        return final_tag_space
