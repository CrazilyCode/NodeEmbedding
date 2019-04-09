from _collections import defaultdict

entity2id = {}
id2entity = {}
entity_len = 0

entity2id['null'] = entity_len
id2entity[entity_len] = 'null'
entity_len += 1

relation2id = {}
id2relation = {}
with open('../data_original/FB13/relation2id.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		rel = line[0]
		relation2id[rel] = int(line[1])
		id2relation[int(line[1])] = rel
		entity2id[rel] = entity_len
		id2entity[entity_len] = rel
		entity_len += 1

with open('../data_original/FB13/entity2id.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		entity = line[0]
		entity2id[entity] = entity_len
		id2entity[entity_len] = entity
		entity_len += 1

with open('FB13/entity2id.txt', 'w') as f:
	for i in range(entity_len):
		f.write('%s\t%d\n' % (id2entity[i], i))

data = []
with open('../data_original/FB13/train.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		data.append('%d\t%d\t%d\n' % (entity2id[line[0]], relation2id[line[1]], entity2id[line[2]]))
with open('FB13/data.txt', 'w') as f:
	f.writelines(data)

entity_nodes = []
attribute_nodes = []
with open('../data_original/FB13/train.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		entity_nodes.append(line[0])
		if relation2id[line[1]] in range(0, 7):
			attribute_nodes.append(line[2])
		else:
			entity_nodes.append(line[2])
with open('../data_original/FB13/valid.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		entity_nodes.append(line[0])
		if relation2id[line[1]] in range(0, 7):
			attribute_nodes.append(line[2])
		else:
			entity_nodes.append(line[2])
with open('../data_original/FB13/test.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		entity_nodes.append(line[0])
		if relation2id[line[1]] in range(0, 7):
			attribute_nodes.append(line[2])
		else:
			entity_nodes.append(line[2])

print('#entity:%d' % len(set(entity_nodes)))
print('#attribute:%d' % len(set(attribute_nodes)))

node2str = []
for node in set(entity_nodes):
	node2str.append('%d\t%s\n' % (entity2id[node], node))
with open('FB13/node2str.txt', 'w') as f:
	f.writelines(node2str)


train_lines = []
with open('../data_original/FB13/train.txt') as f:
	lines = f.readlines()
	print('#train:%d' % (len(lines)))
	for line in lines:
		line = line.strip().split('\t')
		if relation2id[line[1]] in range(0, 7):
			train_lines.append('%d\t%d\t%d\n' % (entity2id[line[0]], relation2id[line[1]], entity2id[line[2]]))
with open('FB13/train.txt', 'w') as f:
	f.writelines(train_lines)

valid_lines = []
with open('../data_original/FB13/valid.txt') as f:
	lines = f.readlines()
	print('#valid:%d' % (len(lines)))
	for line in lines:
		line = line.strip().split('\t')
		valid_lines.append('%d\t%d\t%d\t%s\n' % (entity2id[line[0]], relation2id[line[1]], entity2id[line[2]], line[3]))
with open('FB13/valid.txt', 'w') as f:
	f.writelines(valid_lines)

test_lines = []
with open('../data_original/FB13/test.txt') as f:
	lines = f.readlines()
	print('#test:%d' % (len(lines)))
	for line in lines:
		line = line.strip().split('\t')
		test_lines.append('%d\t%d\t%d\t%s\n' % (entity2id[line[0]], relation2id[line[1]], entity2id[line[2]], line[3]))
with open('FB13/test.txt', 'w') as f:
	f.writelines(test_lines)

