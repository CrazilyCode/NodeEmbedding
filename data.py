from _collections import defaultdict
import numpy as np
import torch
import string
import math
import time

def lineToTensor(name):
    all_letters = string.printable

    name = name.lower()
    tensor = torch.zeros((len(name)), dtype=torch.int)
    for li, letter in enumerate(name):
        if letter not in all_letters:
            index = 0
        else:
            index = all_letters.find(letter)
        tensor[li] = index

    return tensor, len(name)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)


def load_dict(file_path):
    str2id = defaultdict(int)
    id2str = defaultdict(str)

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            idx = int(line[0])
            string = line[1]
            str2id[string] = idx
            id2str[idx] = string

    return str2id, id2str


def load_rel2value(file_path, rel2id, value2id):
    rel2value = defaultdict(list)

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            rel_id = rel2id[line[0]]
            value_id = value2id[line[1]]
            rel2value[rel_id].append(value_id)

    return rel2value


def load_noderel2value(file_path, rel2id, value2id):
    noderel2value_list = defaultdict(list)

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            node_id = int(line[0])
            rel_id = rel2id[line[1]]
            value_id = value2id[line[2]]
            noderel2value_list[(node_id, rel_id)].append(value_id)

    return noderel2value_list

def load_noderel2node(file_path, rel2id):
    noderel2node_list = defaultdict(list)

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            node1_id = int(line[0])
            rel_id = rel2id[line[1]]
            node2_id = int(line[2])
            noderel2node_list[(node1_id, rel_id)].append(node2_id)

    return noderel2node_list

def load_train_data(file_path):
    id2e1 = []
    id2e2 = []
    id2rel = []

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            e1, rel, e2 = line[0], line[1], line[2]
            id2e1.append(e1)
            id2e2.append(e2)
            id2rel.append(rel)

    return id2e1, id2e2, id2rel


def load_test_data(file_path):
    id2e1 = []
    id2e2 = []
    id2rel = []
    id2flag = []

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            e1, rel, e2, flag = line[0], line[1], line[2], line[3]
            id2e1.append(e1)
            id2e2.append(e2)
            id2rel.append(rel)
            id2flag.append(flag)

    return id2e1, id2e2, id2rel, id2flag


def write(file_path, result):
    with open(file_path, 'a+') as f:
        f.write(result + '\n')
