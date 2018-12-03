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

    def __init__(self, input_maxlen, file_dir):
        self.input_maxlen = input_maxlen

        self.rel_value2id, self.id2rel_value = load_dict(file_dir + '/rel_value_dict.txt')
        self.rel_node2id, self.id2rel_node = load_dict(file_dir + '/rel_node_dict.txt')
        self.rel_model2id, self.id2rel_model = load_dict(file_dir + '/rel_model_dict.txt')

        self.node2id, self.id2node = load_dict(file_dir + '/node_info.txt')
        self.value2id, self.id2value = load_dict(file_dir + '/value_dict.txt')

        self.rel2value = load_rel2value(file_dir + '/rel2value.txt', self.rel_value2id, self.value2id)
        
        self.noderel2value_list = load_noderel2value(file_dir + '/node2value.txt', self.rel_value2id, self.value2id)
        self.noderel2node_list = load_noderel2node(file_dir + '/node2node.txt', self.rel_node2id)

        self.id2e1_train, self.id2e2_train, self.id2rel_train = load_train_data(file_dir + '/train.txt')
        self.len = len(self.id2e1_train)

        self.height = len(self.rel_value2id)
        self.depth = len(self.rel_node2id) + 1
        self.rel_len = len(self.rel_model2id)
        self.value_len = len(self.value2id)

    def get_self_data(self, name, self_rel, self_value):
        embedding = []

        for rel in self.id2rel_value:
            value_list = self.noderel2value_list[(name, rel)].copy()
            if rel == self_rel:
                if self_value in value_list:
                    value_list.remove(self_value)

            if len(value_list) == 0:
                value = self.value2id['null_' + self.id2rel_value[rel]]
            else:
                flag = random.randint(0, 2)
                if flag == 2:
                    value = self.value2id['null_' + self.id2rel_value[rel]]
                else:
                    value = random.choice(value_list)
            
            embedding.append(value)

        embedding = np.row_stack(embedding)
        embedding = embedding.reshape(1, self.height)

        return embedding

    def get_data(self, name):
        embedding = []

        for rel in self.id2rel_value:
            value_list = self.noderel2value_list[(name, rel)]
            if len(value_list) == 0:
                value = self.value2id['null_' + self.id2rel_value[rel]]
            else:
                flag = random.randint(0, 2)
                if flag == 2:
                    value = self.value2id['null_' + self.id2rel_value[rel]]
                else:
                    value = random.choice(value_list)
            embedding.append(value)

        embedding = np.row_stack(embedding)
        embedding = embedding.reshape(1, self.height)

        return embedding

    def get_neg(self, name_id, rel_id, value_id):
        value_list = self.rel2value[rel_id].copy()
        right_list = self.noderel2value_list[(name_id, rel_id)]

        for value in right_list:
            if value not in value_list:
                continue
            value_list.remove(value)

        if len(value_list) == 0:
            neg_id = self.value2id['null_' + self.id2rel_value[rel_id]]
        else:
            neg_id = random.choice(value_list)

        return neg_id

    def get_result(self, name_id, rel_id, value_id):
        data_self = self.get_self_data(name_id, rel_id, value_id)

        data_rel = []
        data_rel.append(data_self)

        for rel in self.id2rel_node:
            name_list = self.noderel2node_list[(name_id, rel)]
            if len(name_list) == 0:
                name = 0
            else:
                flag = random.randint(0, 2)
                if flag == 2:
                    name = 0
                else:
                    name = random.choice(name_list)

            embedding = self.get_data(name)
            data_rel.append(embedding)
        
        data_rel = np.row_stack(data_rel)

        return data_rel

    def __getitem__(self, index):
        e1 = self.id2e1_train[index]
        e2 = self.id2e2_train[index]
        rel = self.id2rel_train[index]

        name_id = int(e1)
        rel_id = self.rel_model2id[rel]
        value_id = self.value2id[e2]

        name = self.id2node[name_id][ : self.input_maxlen]
        name_result, name_len = lineToTensor(name)
        temp = torch.zeros((self.input_maxlen - name_len), dtype=torch.int)
        data_self = torch.cat((name_result, temp), 0)

        data_other = self.get_result(name_id, rel_id, value_id)

        pos_id = value_id
        neg_id = self.get_neg(name_id, rel_id, value_id)

        return data_self, name_len, data_other, rel_id, pos_id, neg_id

    def __len__(self):
        return self.len


def train(channel_size, out_dim, char_dim, hidden_size, margin1, margin2, train_dataset):
    train_epochs = 200
    nn_lr = 0.001
    batch_size = 100

    file_name = str(channel_size) + '-' + str(out_dim) + '-' + str(char_dim) + '-' + str(hidden_size) + '-' + str(margin1) + '-' + str(margin2)


    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=1, batch_size=batch_size)
    train_len = train_dataset.len

    net = Network(train_dataset.depth, train_dataset.height, channel_size, train_dataset.rel_len, out_dim, train_dataset.value_len, char_dim, hidden_size)
    loss_func = ContrastiveLoss(margin1, margin2)
    optimizer = optim.Adam(net.parameters(), lr=nn_lr)

    start = time.time()

    for epoch in range(train_epochs):
        epoch_loss = 0
        current_loss = 0

        for i, data in enumerate(train_dataloader, 0):
            data_self, name_len, data_other, rel_id, value_id, neg_id = data

            out_self, pos_out, neg_out = net(data_self, name_len, data_other, rel_id, value_id, neg_id)

            optimizer.zero_grad()
            loss = loss_func(out_self, pos_out, neg_out)
            loss.backward()
            optimizer.step()

            current_loss += loss.data.item()
            epoch_loss += loss.data.item()

            j = i * batch_size
            if j % 20000 == 0:
                print('%d %d %d%% (%s) %.4f' % (epoch, j, j * 100 / train_len, timeSince(start), current_loss))
                current_loss = 0
            

        if not os.path.exists('out/' + file_name):
            os.makedirs('out/' + file_name)
        
        j = epoch + 1
        if j % 10 == 0:
            model_name = 'out/' + file_name + '/net-' + str(j) + '.pt'
            torch.save(net, model_name)

        loss_str = '%.4f' % epoch_loss
        write('out/' + file_name + '/loss.txt', loss_str)

    print('train done!')

    

if __name__ == '__main__':
    input_maxlen = 50

    channel_size = 500
    out_dim = 50

    char_dim = 10
    hidden_size = 50
    
    margin1 = 10
    margin2 = 50

    file_dir = '../FB13'
    train_dataset = TrainDataset(input_maxlen, file_dir)
    train(channel_size, out_dim, char_dim, hidden_size, margin1, margin2, train_dataset)
    
