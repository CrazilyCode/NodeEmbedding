from model import *
from data import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from _collections import defaultdict
import numpy as np
import random


class TestDataset(Dataset):

    def __init__(self, input_maxlen, file_dir, test_name, flag):
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

        self.id2e1_test, self.id2e2_test, self.id2rel_test, self.id2flag_test = load_test_data(file_dir + '/' + test_name)
        self.len = len(self.id2e1_test)

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
        e1_id = self.id2e1_test[index]
        e2_id = self.id2e2_test[index]
        rel_id = self.id2rel_test[index]
        flag = self.id2flag_test[index]

        text = self.node2str[e1_id][ : self.input_maxlen]
        data_text, text_len = lineToTensor(text)
        temp = torch.zeros((self.input_maxlen - text_len), dtype=torch.int)
        data_text = torch.cat((data_text, temp), 0)

        data_net = self.get_result(e1_id, rel_id, e2_id)

        return data_text, text_len, data_net, rel_id, e2_id, flag


    def __len__(self):
        return self.len

def test(input_maxlen, file_dir, test_name, net_path, out_path, epoch, margin1, margin2, flag):
    net = torch.load(net_path)

    test_dataset = TestDataset(input_maxlen, file_dir, test_name, flag)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=1, batch_size=1000)

    dict_right1 = defaultdict(int)
    dict_right2 = defaultdict(int)
    dict_count = defaultdict(int)
    right1 = 0
    right2 = 0
    count = 0

    pdist = nn.PairwiseDistance(p=2)
    d = (margin1 + margin2) / 2

    for i, data in enumerate(test_dataloader, 0):
        data_text, text_len, data_net, rel_id, node_id, flag = data
        if torch.cuda.is_available() == True:
            data_text = data_text.cuda()
            text_len = text_len.cuda()
            data_net = data_net.cuda()
            rel_id = rel_id.cuda()
            node_id = node_id.cuda()

        output = net.forward_once(data_text, text_len, data_net, rel_id)
        # if torch.cuda.is_available() == True:
        #     output = output.cpu()

        embedding = net.node_embedding(node_id)
        
        dist = pdist(output, embedding)

        data_len = text_len.size(0)
        for idx in range(data_len):        
            if dist[idx] < d and flag[idx] == 1:
                right1 += 1
            if dist[idx] >= d and flag[idx] == -1:
                right2 += 1

        count += data_len
        if count % 10000 == 0:
            line = '%d %d %d : %.4f' % (right1, right2, count, (right1 + right2 ) / count)
            print(line)
        
    line = '%d : %d %d %d : %.4f' % (epoch, right1, right2, count, (right1 + right2 ) / count)
    print(line)
    write(out_path, line)

if __name__ == '__main__':
    input_maxlen = 100
    file_dir = '../data/FB13'
    file_name = '10-100-20-500-100-10-50(0.001-100)'
    test_name = 'test.txt'
    margin1 = 10
    margin2 = 50
    flag = True

    out_path = 'out/' + file_dir + '/' + file_name + '/' + test_name

    for i in range(110, 111):
        epoch = i
        net_path = 'out/' + file_dir + '/' + file_name + '/net-' + str(epoch) + '.pt'
        test(input_maxlen, file_dir, test_name, net_path, out_path, epoch, margin1, margin2, flag)
    
