import torch.nn as nn
import torch
from torch.autograd import Variable


class DenselyConnectedLinearLSTM(nn.Module):
    """ Densely Connected FCs + LSTM """

    def __init__(self, input_size, hidden_dim, target_size, depth=2):
        super(DenselyConnectedLinearLSTM, self).__init__()

        self.input_size = input_size
        self.activation = nn.LeakyReLU()
        interim_layer_size = 256
        self.drop = nn.Dropout(0.3)

        self.ln11 = nn.Linear(input_size, interim_layer_size)
        self.ln12 = nn.Linear(interim_layer_size, interim_layer_size)
        self.ln21 = nn.Linear(input_size + interim_layer_size, interim_layer_size)
        self.ln22 = nn.Linear(interim_layer_size, interim_layer_size)
        self.ln31 = nn.Linear(input_size + interim_layer_size * 2, interim_layer_size * 2)
        self.ln32 = nn.Linear(interim_layer_size * 2, interim_layer_size)
        self.ln41 = nn.Linear(input_size + interim_layer_size * 3, interim_layer_size * 3)
        self.ln42 = nn.Linear(interim_layer_size * 3, interim_layer_size)
        self.ln51 = nn.Linear(input_size + interim_layer_size * 4, interim_layer_size * 4)
        self.ln52 = nn.Linear(interim_layer_size * 4, interim_layer_size)

        self.lnu1 = nn.Linear(interim_layer_size, 512)
        self.bn1 = nn.BatchNorm2d(512)
        self.lnu2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm2d(512)
        self.lnu3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm2d(256)

        self.lstm = nn.LSTM(512, hidden_dim, depth, batch_first=True)
        self.lstm2 = nn.LSTM(512, hidden_dim, depth, batch_first=True)

        self.hidden1 = nn.Linear(512 + hidden_dim, hidden_dim)
        self.bnh1 = nn.BatchNorm2d(hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, target_size)

    def forward(self, data, batch_len):
        in_size = self.input_size
        drop = self.drop

        activation = self.activation
        t = data.size()
        x = data.view(-1, in_size)

        x0 = activation(self.ln11(x))
        x0 = (activation(self.ln12(x0)))

        x1 = torch.cat([x, x0], dim=1)
        x1 = activation(self.ln21(x1))
        x1 = (activation(self.ln22(x1)))

        x2 = torch.cat([x, x0, x1], dim=1)
        x2 = activation(self.ln31(x2))
        x2 = (activation(self.ln32(x2)))

        x3 = torch.cat([x, x0, x1, x2], dim=1)
        x3 = activation(self.ln41(x3))
        x3 = (activation(self.ln42(x3)))

        x4 = torch.cat([x, x0, x1, x2, x3], dim=1)
        x4 = activation(self.ln51(x4))
        x4 = (activation(self.ln52(x4)))

        x = x4
        x = activation(self.bn1(self.lnu1(x)))
        x = activation(self.bn2(self.lnu2(x)))
       
        x = x.view(t[0], t[1], -1)

        packed_batch = torch.nn.utils.rnn.pack_padded_sequence(
            x, batch_len, batch_first=True)

        packed_lstm_out, hidden = self.lstm(packed_batch)
        packed_lstm_out_2, hidden_2 = self.lstm2(packed_batch)
        unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out)
        unpacked2, unpacked_len2 = nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out_2)

        idx_sel = torch.LongTensor(batch_len)

        # Get all act:
        feat_size = unpacked.size(2)
        all_act = []
        all_act2 = []
        for i in range(idx_sel.size(0)):
            all_act.append(
                unpacked[:idx_sel[i], i, :].contiguous().view(-1, feat_size))
            all_act2.append(
                unpacked2[:idx_sel[i], i, :].contiguous().view(-1, feat_size))
        all_act = torch.cat(all_act, dim=0)
        all_act2 = torch.cat(all_act2, dim=0)

        all_act = torch.cat([all_act, all_act2], dim=1)
        l1 = activation(self.bn3(self.lnu3(all_act)))

        l1 = torch.cat([all_act, l1], dim=1)
        l1 = activation(self.bnh1(self.hidden1(l1)))
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
