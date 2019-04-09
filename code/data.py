from _collections import defaultdict
import torch
import string
import math
import time


def lineToTensor(text):
    all_letters = string.printable

    tensor = torch.zeros((len(text)), dtype=torch.int)
    for li, letter in enumerate(text):
        if letter not in all_letters:
            index = 0
        else:
            index = all_letters.find(letter)
        tensor[li] = index

    return tensor, len(text)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60

    return '%dm %ds' % (m, s)

def load_str(file_path):
    node2str = defaultdict(str)
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            node2str[int(line[0])] = line[1]
    return node2str

def load_dict(file_path):
    str2id = defaultdict(int)
    id2str = defaultdict(str)

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            string = line[0]
            idx = int(line[1])
            str2id[string] = idx
            id2str[idx] = string

    return str2id, id2str

def load_data(file_path):
    id2e1 = []
    id2e2 = []
    id2rel = []

    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            e1, rel, e2 = line[0], line[1], line[2]
            id2e1.append(int(e1))
            id2e2.append(int(e2))
            id2rel.append(int(rel))

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
            id2e1.append(int(e1))
            id2e2.append(int(e2))
            id2rel.append(int(rel))
            id2flag.append(int(flag))

    return id2e1, id2e2, id2rel, id2flag

def write(file_path, result):
    with open(file_path, 'a+') as f:
        f.write(result + '\n')
