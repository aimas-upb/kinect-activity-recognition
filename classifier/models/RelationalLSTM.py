import torch.nn as nn
import torch
import itertools


class RelationalLSTM(nn.Module):
    """ Relational FCs + LSTM """

    def __init__(self, input_size, joint_size, hidden_dim, target_size,
                 depth=2):
        super(RelationalLSTM, self).__init__()

        self.activation = nn.LeakyReLU()
        self.drop = nn.Dropout(0.3)
        self.input_size = input_size
        self.joint_size = joint_size
        self.joint_count = int(input_size / self.joint_size)

        self.g_size = 256
        self.f_size = 256

        # pairwise combination indexes
        combs = list(
            itertools.combinations(range(0, int(input_size / self.joint_size)),
                                   2))
        self.idx_i = torch.LongTensor([idx[0] for idx in combs]).cuda()
        self.idx_j = torch.LongTensor([idx[1] for idx in combs]).cuda()
        self.comb_cnt = len(combs)

        # g function
        self.g_fc1 = nn.Linear(2 * self.joint_size, self.g_size)
        self.bng1 = nn.BatchNorm1d(self.g_size)
        self.g_fc2 = nn.Linear(self.g_size, self.g_size)
        self.bng2 = nn.BatchNorm1d(self.g_size)
        self.g_fc3 = nn.Linear(self.g_size, self.f_size)

        # f function
        self.f_fc1 = nn.Linear(self.f_size, self.f_size)
        self.bnf1 = nn.BatchNorm1d(self.f_size)
        self.f_fc2 = nn.Linear(self.f_size, self.f_size)
        self.bnf2 = nn.BatchNorm1d(self.f_size)
        self.f_fc3 = nn.Linear(self.f_size, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(hidden_dim + input_size, hidden_dim, depth,
                            batch_first=True)

        # output
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.bnh1 = nn.BatchNorm1d(hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, target_size)

    def forward(self, data, batch_len):
        drop = self.drop
        activation = self.activation
        t = data.size()

        # cast all pairs
        x = data.view(t[0], t[1], self.joint_count, self.joint_size)
        x = torch.cat([x[:, :, self.idx_i, :], x[:, :, self.idx_j, :]], 3)
        x = x.view(-1, 2 * self.joint_size)

        # apply g
        x_g = self.bng1(activation(self.g_fc1(x)))
        x_g = self.bng2(activation(self.g_fc2(x_g)))
        x_g = activation(self.g_fc3(x_g))

        # reshape and sum
        x_g = x_g.view(t[0], t[1], self.comb_cnt, self.f_size).sum(2)

        # apply f
        x_g = x_g.view(-1, self.f_size)
        x_f = self.bnf1(activation(self.f_fc1(x_g)))
        x_f = self.bnf2(activation(self.f_fc2(x_f)))
        x_f = activation(self.f_fc3(x_f))

        # concatenate original input
        x = data.view(-1, self.input_size)
        x_f = torch.cat([x_f, x], 1)
        x_f = x_f.view(t[0], t[1], -1)

        # pack and pass through LSTM
        packed_batch = torch.nn.utils.rnn.pack_padded_sequence(
            x_f, batch_len, batch_first=True)
        packed_lstm_out, hidden = self.lstm(packed_batch)
        unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out)

        # Get all act
        idx_sel = torch.LongTensor(batch_len)
        feat_size = unpacked.size(2)
        all_act = []
        for i in range(idx_sel.size(0)):
            all_act.append(
                unpacked[:idx_sel[i], i, :].contiguous().view(-1, feat_size))
        all_act = torch.cat(all_act, dim=0)

        l1 = activation(self.bnh1(self.hidden1(all_act)))
        tag_space = self.hidden2(l1)

        # Sum act/batch
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
