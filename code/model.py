import torch
import torch.nn as nn
import string
import torch.nn.functional as F

torch.manual_seed(1)

def sort_batch(data, length):
    data_id = torch.arange(data.size(0), dtype=torch.int)

    _, inx = torch.sort(length, descending=True)

    data = data[inx]
    data_id = data_id[inx]
    length = length[inx]

    return data, data_id, length


def recover_batch(data, data_id):
    _, inx = torch.sort(data_id)
    data = data[inx]

    return data


class Network(nn.Module):

    def __init__(self, in_channel, char_dim, hidden_size, height, width, channel_size, rel_len, node_len, linear_dim):
        super(Network, self).__init__()

        n_letters = len(string.printable)
        self.char_embedding = nn.Embedding(n_letters, char_dim)

        num_layers = 1
        biFlag = True
        self.bi_num = 2
        self.rnn = nn.LSTM(
            input_size=char_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=biFlag,
        )

        self.conv = nn.Conv2d(in_channel, channel_size, (height, width))

        self.vec_len = channel_size + hidden_size * self.bi_num
        self.rel_embedding = nn.Embedding(rel_len, self.vec_len)
        self.node_embedding = nn.Embedding(node_len, width)
        
        self.linear = nn.Linear(self.vec_len, linear_dim)
        self.out_linear = nn.Linear(linear_dim, width)

    def forward_text(self, data, length):
        data, data_id, length = sort_batch(data, length)
        data = self.char_embedding(data.long())

        data = nn.utils.rnn.pack_padded_sequence(data, length, batch_first=True)

        out, (h_n, h_c) = self.rnn(data)
        out, length = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = torch.mean(out, 1)
        out = recover_batch(out, data_id)

        return out

    def forward_once(self, data_text, text_len, data_net, rel_id):
        out_text = self.forward_text(data_text, text_len)

        data_net = self.node_embedding(data_net.long())
        out_net = self.conv(data_net)
        out_net = out_net.view(out_net.size(0), -1)

        rout = torch.cat([out_text, out_net], 1)

        weight = self.rel_embedding(rel_id.long())
        rout = rout * weight

        rout = F.relu(self.linear(rout))
        rout = self.out_linear(rout)

        return rout

    def forward(self, data_text, text_len, data_net, rel_id, pos_id, neg_id):
        out_net = self.forward_once(data_text, text_len, data_net, rel_id)
        
        out_pos = self.node_embedding(pos_id.long())
        out_neg = self.node_embedding(neg_id.long())

        return out_net, out_pos, out_neg

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin1, margin2):
        super(ContrastiveLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, out_self, pos_out, neg_out):
        pdist = nn.PairwiseDistance(p=2)
        pos_dist = pdist(out_self, pos_out)
        neg_dist = pdist(out_self, neg_out)
        loss = torch.mean(torch.clamp(pos_dist - self.margin1, min=0.0) + torch.clamp(self.margin2 - neg_dist, min=0.0))

        return loss