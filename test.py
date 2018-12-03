from model import *
from data import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from _collections import defaultdict
import numpy as np
import random


class TrainDataset(Dataset):

    def __init__(self, file_dir, input_maxlen, file_path):
        self.input_maxlen = input_maxlen

        self.rel_value2id, self.id2rel_value = load_dict(file_dir + '/rel_value_dict.txt')
        self.rel_node2id, self.id2rel_node = load_dict(file_dir + '/rel_node_dict.txt')
        self.rel_model2id, self.id2rel_model = load_dict(file_dir + '/rel_model_dict.txt')

        self.node2id, self.id2node = load_dict(file_dir + '/node_info.txt')
        self.value2id, self.id2value = load_dict(file_dir + '/value_dict.txt')

        self.rel2value = load_rel2value(file_dir + '/rel2value.txt', self.rel_value2id, self.value2id)
        
        self.noderel2value_list = load_noderel2value(file_dir + '/node2value.txt', self.rel_value2id, self.value2id)
        self.noderel2node_list = load_noderel2node(file_dir + '/node2node.txt', self.rel_node2id)

        self.height = len(self.rel_value2id)
        self.depth = len(self.rel_node2id)
        self.rel_len = len(self.rel_model2id)
        self.value_len = len(self.value2id)
        

        self.id2e1_test, self.id2e2_test, self.id2rel_test, self.id2flag_test = load_test_data(file_dir + file_path)
        self.len = len(self.id2e1_test)


    
    def get_data(self, name):
        embedding = []

        for rel in self.id2rel_value:
            value_list = self.noderel2value_list[(name, rel)]
            if len(value_list) == 0:
                value = self.value2id['null_' + self.id2rel_value[rel]]
            else:
                value = random.choice(value_list)
            embedding.append(value)

        embedding = np.row_stack(embedding)
        embedding = embedding.reshape(1, self.height)

        return embedding

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
                value = random.choice(value_list)
            
            embedding.append(value)

        embedding = np.row_stack(embedding)
        embedding = embedding.reshape(1, self.height)

        return embedding
    

    def get_result(self, name_id, rel_id, value_id):
        data_self = self.get_self_data(name_id, rel_id, value_id)

        data_rel = []
        data_rel.append(data_self)

        for rel in self.id2rel_node:
            name_list = self.noderel2node_list[(name_id, rel)]
            if len(name_list) == 0:
                name = 0
            else:
                name = random.choice(name_list)

            embedding = self.get_data(name)
            data_rel.append(embedding)
        
        data_rel = np.row_stack(data_rel)

        return data_rel

    def __getitem__(self, index):
        e1 = self.id2e1_test[index]
        e2 = self.id2e2_test[index]
        rel = self.id2rel_test[index]
        flag = self.id2flag_test[index]

        name_id = int(e1)
        rel_id = self.rel_model2id[rel]
        value_id = self.value2id[e2]

        name = self.id2node[name_id][ : self.input_maxlen]
        name_result, name_len = lineToTensor(name)
        temp = torch.zeros((self.input_maxlen - name_len), dtype=torch.int)
        data_self = torch.cat((name_result, temp), 0)

        data_other = self.get_result(name_id, rel_id, value_id)

        flag = int(flag)

        return data_self, name_len, data_other, rel_id, value_id, flag

    def __len__(self):
        return self.len

def test(file_dir, net_path, file_path, out_path, margin1, margin2, epoch):
    net = torch.load(net_path)

    input_maxlen = 50

    train_dataset = TrainDataset(file_dir, input_maxlen, file_path)
    train_dataloader = DataLoader(train_dataset, shuffle=False, num_workers=1, batch_size=1000)


    dict_right1 = defaultdict(int)
    dict_right2 = defaultdict(int)
    dict_count = defaultdict(int)
    right1 = 0
    right2 = 0
    count = 0

    d = (margin1 + margin2) / 2

    pdist = nn.PairwiseDistance(p=2)

    for i, data in enumerate(train_dataloader, 0):
        data_self, name_len, data_other, rel_id, value_id, flag = data

        output = net.forward_once(data_self, name_len, data_other, rel_id)

        value_out = net.value_out(value_id.long())

        dist = pdist(output, value_out)

        data_len = name_len.size(0)
        for idx in range(data_len):        
            if dist[idx] < d and flag[idx] == 1:
                right1 += 1
            if dist[idx] >= d and flag[idx] == -1:
                right2 += 1

        count += data_len
        if count % 20000 == 0:
            line = '%d %d %d : %.4f%%' % (right1, right2, count, (right1 + right2 ) / count)
            print(line)
    line = '%d : %d %d %d : %.4f%%' % (epoch, right1, right2, count, (right1 + right2 ) / count)
    print(line)
    write(out_path, line)

if __name__ == '__main__':
    for i in range(1, 21):
        epoch = i * 10

        file_dir = '../FB13'
        file_name = '500-50-10-50-10-50'
        net_path = 'out/' + file_name + '/net-' + str(epoch) + '.pt'
        file_path = '/test.txt'
        out_path = 'out/' + file_name + '/test.txt'
        margin1 = 10
        margin2 = 50
        
        test(file_dir, net_path, file_path, out_path, margin1, margin2, epoch)
    
