from model import *
from data import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from _collections import defaultdict
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_confusion(confusion):
    n_categories = len(confusion)
    all_categories = []
    for i in range(n_categories):
        all_categories.append(i)

    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)


    xlabels = ['']
    ylabels = ['']

    for i in range(6):
        for j in range(5):
            xlabels.append(str(i) + '-' + str(j))
            ylabels.append(str(i) + '-' + str(j))


    # Set up axes
    ax.set_xticklabels(xlabels, rotation=90)
    # ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xlabel('guess node')
    ax.set_ylabel('real node')

    plt.show()
    # plt.savefig(pngpath) 
    plt.close()

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
        for i in range(self.data_len):
            e1 = self.id2e1_data[i]
            e2 = self.id2e2_data[i]
            rel = self.id2rel_data[i]
            self.noderel2nodes[(e1, rel)].append(e2)
        
        self.id2e1_test, self.id2e2_test, self.id2rel_test = load_data(file_dir + '/' + test_name + '.txt')
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

        text = self.node2str[e1_id][ : self.input_maxlen]
        data_text, text_len = lineToTensor(text)
        temp = torch.zeros((self.input_maxlen - text_len), dtype=torch.int)
        data_text = torch.cat((data_text, temp), 0)

        data_net = self.get_result(e1_id, rel_id, e2_id)

        return data_text, text_len, data_net, rel_id, e2_id


    def __len__(self):
        return self.len

class Hit_meeting():
    def __init__(self, entity2id, embedding):
        meeting_names = []
        with open('../data/DBLP/meeting.txt') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                meeting_names.append(line[0])

        self.idx2id = {}
        with open('../data/DBLP/meeting2label.txt') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                self.idx2id[entity2id[line[0]]] = len(self.idx2id)

        meeting_ids = []
        self.meeting_id2index = {}
        self.index2meeting_id = {}
        for i in range(len(meeting_names)):
            meeting_name = meeting_names[i]

            meeting_id = entity2id[meeting_name]
            meeting_ids.append(meeting_id)

            self.meeting_id2index[meeting_id] = i
            self.index2meeting_id[i] = meeting_id

        meeting_matrix = torch.from_numpy(np.array(meeting_ids))
        self.embedding = embedding(meeting_matrix.long().cuda())


    def predict(self, out, meeting_id):
        pdist = nn.PairwiseDistance(p=2)

        dist = pdist(out, self.embedding)
        values, indices = torch.sort(dist)
        if torch.cuda.is_available() == True:
            indices = indices.cpu()
        indice_list = list(indices.numpy())
        indice = indice_list.index(self.meeting_id2index[meeting_id])

        # print(meeting_id)
        # print(dist)
        # print(values)
        # print(indices)
        # print(indice)
        idx_x = self.idx2id[meeting_id]
        min_dist_indice = indices[0].item()
        min_meeting_id = self.index2meeting_id[min_dist_indice]
        idx_y = self.idx2id[min_meeting_id]

        # line = '%d-%d &[ ' % (int(idx_x/5), int(idx_x%5))
        # for i in range(5):
        #     dist_indice = indices[i].item()
        #     meeting_id = self.index2meeting_id[dist_indice]
        #     idx = self.idx2id[meeting_id]
        #     line = line + '%d-%d, ' % (int(idx/5), int(idx%5))
        # line = line[0:-2] + ']\\\\'
        # print(line)

        return indice, idx_x, idx_y

def test(input_maxlen, file_dir, net_path, out_path, epoch, test_name, flag):
    net = torch.load(net_path)

    test_dataset = TestDataset(input_maxlen, file_dir, test_name, flag)
    test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=1, batch_size=1)

    hit_meeting = Hit_meeting(test_dataset.entity2id, net.node_embedding)

    sum_rank = 0
    sum_hit1 = 0
    sum_hit5 = 0
    count = 0

    pdist = nn.PairwiseDistance(p=2)

    matrix = np.zeros((30, 30))

    for i, data in enumerate(test_dataloader, 0):
        data_text, text_len, data_net, rel_id, node_id = data
        if torch.cuda.is_available() == True:
            data_text = data_text.cuda()
            text_len = text_len.cuda()
            data_net = data_net.cuda()
            rel_id = rel_id.cuda()

        output = net.forward_once(data_text, text_len, data_net, rel_id)
        node_id = node_id.numpy()

        data_len = text_len.size(0)
        for idx in range(data_len): 
            rank, idx_x, idx_y = hit_meeting.predict(output[idx], node_id[idx])
            matrix[idx_x][idx_y] += 1
            count += 1
            sum_rank += rank
            if rank < 1:
                sum_hit1 += 1
            if rank < 5:
                sum_hit5 += 1

    line = '%d : %d: %.2f %.4f %.4f' % (epoch, count, sum_rank / count, sum_hit1 / count, sum_hit5 / count)
    print(line)
    write(out_path, line)
    print(matrix)
    plot_confusion(matrix)

if __name__ == '__main__':
    input_maxlen = 100
    flag = False
    file_dir = '../data/DBLP'
    file_name = '5-100-10-100-50-1-50(0.005-100)'
    test_name = 'test'
    out_path = 'data/DBLP/' + file_name + '/' + test_name + '-hit.txt'

    for i in range(20, 21):
        epoch = i
        net_path = 'data/DBLP/' + file_name + '/net-' + str(epoch) + '.pt'
        test(input_maxlen, file_dir, net_path, out_path, epoch, test_name, flag)
    
