from model import *
from data import *

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim

from _collections import defaultdict
import numpy as np
import random


class TrainDataset(Dataset):

    def __init__(self, input_maxlen, file_dir, flag):
        self.input_maxlen = input_maxlen
        self.flag = flag

        self.rel2id, self.id2rel = load_dict(file_dir + '/rel2id.txt')
        self.rel2id_1, self.id2rel_1 = load_dict(file_dir + '/rel2id-1.txt')
        self.rel2id_2, self.id2rel_2 = load_dict(file_dir + '/rel2id-2.txt')
        self.entity2id, self.id2entity = load_dict(file_dir + '/entity2id.txt')
        self.rel_len = len(self.rel2id)
        self.rel_len_1 = len(self.rel2id_1)
        self.rel_len_2 = len(self.rel2id_2)
        self.entity_len = len(self.entity2id)

        self.node2str = load_str(file_dir + '/node2str.txt')

        self.id2e1_data, self.id2e2_data, self.id2rel_data = load_data(file_dir + '/data.txt')
        self.data_len = len(self.id2e1_data)
        
        self.noderel2nodes = defaultdict(list)
        self.rel2nodes = defaultdict(list)
        for i in range(self.data_len):
            e1 = self.id2e1_data[i]
            e2 = self.id2e2_data[i]
            rel = self.id2rel_data[i]
            self.noderel2nodes[(e1, rel)].append(e2)
            self.rel2nodes[rel].append(e2)

        for rel in self.rel2nodes.keys():
            self.rel2nodes[rel] = list(set(self.rel2nodes[rel]))

        self.id2e1_train, self.id2e2_train, self.id2rel_train = load_data(file_dir + '/train.txt')
        self.len = len(self.id2e1_train)

    def get_self_data(self, e1_id, rel_id, e2_id):
        embedding = []

        for rel_name in self.rel2id_2:
            rel = self.rel2id[rel_name]
            nodes = self.noderel2nodes[(e1_id, rel)].copy()
            if rel == rel_id and e2_id in nodes:
                nodes.remove(e2_id)

            if len(nodes) == 0:
                node = self.entity2id['null_' + rel_name]
            else:
                node = random.choice(nodes)
            
            embedding.append(node)

        embedding = np.row_stack(embedding)
        embedding = embedding.reshape(1, -1)

        return embedding

    def get_data(self, e1_id):
        embedding = []

        for rel_name in self.rel2id_2:
            rel = self.rel2id[rel_name]
            nodes = self.noderel2nodes[(e1_id, rel)]

            if len(nodes) == 0:
                node = self.entity2id['null_' + rel_name]
            else:
                node = random.choice(nodes)
            
            embedding.append(node)

        embedding = np.row_stack(embedding)
        embedding = embedding.reshape(1, -1)

        return embedding

    def get_result(self, e1_id, rel_id, e2_id):
        data_rel = []

        if self.flag == True:
            self_data = self.get_self_data(e1_id, rel_id, e2_id)
            data_rel.append(self_data)

        for rel_name in self.rel2id_1:
            rel = self.rel2id[rel_name]
            node_list = self.noderel2nodes[(e1_id, rel)].copy()

            if len(node_list) == 0:
                node = self.entity2id['null_' + rel_name]
            else:
                node = random.choice(node_list)

            embedding = self.get_data(node)
            data_rel.append(embedding)
        
        data_rel = np.row_stack(data_rel)

        return data_rel

    def __getitem__(self, index):
        e1_id = self.id2e1_train[index]
        e2_id = self.id2e2_train[index]
        rel_id = self.id2rel_train[index]

        text = self.node2str[e1_id][ : self.input_maxlen]
        data_text, text_len = lineToTensor(text)
        temp = torch.zeros((self.input_maxlen - text_len), dtype=torch.int)
        data_text = torch.cat((data_text, temp), 0)

        data_net = self.get_result(e1_id, rel_id, e2_id)

        pos_id = e2_id
        neg_id = random.choice(self.rel2nodes[rel_id])

        return data_text, text_len, data_net, rel_id, pos_id, neg_id

    def __len__(self):
        return self.len


def train(input_maxlen, char_dim, hidden_size, node_dim, channel_size, linear_dim, margin1, margin2, nn_lr, batch_size, file_dir, flag):
    train_epochs = 150

    file_name = 'out/' + file_dir + '/' + str(char_dim) + '-' + str(hidden_size) + '-' + str(node_dim) + '-' + str(channel_size) + '-' + str(linear_dim) + '-' + str(margin1) + '-' + str(margin2) + '(' + str(nn_lr) + '-' + str(batch_size) + ')'
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    train_dataset = TrainDataset(input_maxlen, file_dir, flag)
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=1, batch_size=batch_size)
    train_len = train_dataset.len

    if flag == True:
        height = train_dataset.rel_len_1 + 1
    else:
        height = train_dataset.rel_len_1

    net = Network(height, char_dim, hidden_size, train_dataset.rel_len_2, node_dim, channel_size, train_dataset.rel_len, train_dataset.entity_len, linear_dim)
    # net = torch.load(file_name + '/net-' + str(100) + '.pt')
    loss_func = ContrastiveLoss(margin1, margin2)
    optimizer = optim.Adam(net.parameters(), lr=nn_lr)

    if torch.cuda.is_available() == True:
        torch.backends.cudnn.benchmart = True
        torch.cuda.set_device(0)

        net = net.cuda()
        loss_func = loss_func.cuda()
    
    start = time.time()

    for epoch in range(train_epochs):
        epoch_loss = 0
        current_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            data_text, text_len, data_net, rel_id, pos_id, neg_id = data

            if torch.cuda.is_available() == True:
                data_text = data_text.cuda()
                text_len = text_len.cuda()
                data_net = data_net.cuda()
                rel_id = rel_id.cuda()
                pos_id = pos_id.cuda()
                neg_id = neg_id.cuda()

            net_out, pos_out, neg_out = net(data_text, text_len, data_net, rel_id, pos_id, neg_id)

            optimizer.zero_grad()
            loss = loss_func(net_out, pos_out, neg_out)
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available() == True:
                loss = loss.cpu()

            current_loss += loss.data.item()
            epoch_loss += loss.data.item()

            j = i * batch_size
            if j % 10000 == 0:
                print('%d %d %d%% (%s) %.4f' % (epoch, j, j * 100 / train_len, timeSince(start), current_loss))
                current_loss = 0

        j = epoch + 1
        if j % 1 == 0:
            model_name = file_name + '/net-' + str(j) + '.pt'
            torch.save(net, model_name)

        loss_str = '%.4f' % epoch_loss
        write(file_name + '/loss.txt', loss_str)

    print('train done!')
    

if __name__ == '__main__':
    input_maxlen = 100

    char_dim = 10
    hidden_size = 100

    node_dim = 20
    channel_size = 500

    linear_dim = 100

    margin1 = 10
    margin2 = 50

    nn_lr = 0.001
    batch_size = 100

    # file_dir = '../data/DBLP'
    # flag = False
    file_dir = '../data/FB13'
    flag = True

    train(input_maxlen, char_dim, hidden_size, node_dim, channel_size, linear_dim, margin1, margin2, nn_lr, batch_size, file_dir, flag)
    
