from _collections import defaultdict
import numpy as np

def writes(file_path, lines):
	with open(file_path, 'w') as f:
		f.writelines(lines)


paper_list = []
author_list = []
meeting_list = []
with open('../data_original/DBLP/paper.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		paper = line[0]
		titile = line[1]
		authors = line[2].split(';')
		meeting = line[5]
		paper_list.append('paper_' + paper)
		for author in authors:
			if author.strip() != '':
				author_list.append(author.strip())
		meeting_list.append(meeting)

paper_set = set(paper_list)
author_set = set(author_list)
meeting_set = set(meeting_list)

print('#entity:%d' % (len(paper_set) + len(author_set)))
print('#attribute:%d' % (len(meeting_set)))

rel2id = {}
id2rel = {}
with open('DBLP/rel2id.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		rel2id[line[0]] = int(line[1])
		id2rel[int(line[1])] = line[0]

entity2id = {}
entity2id['null'] = 0
entity_num = 1
for rel in rel2id.keys():
	entity2id['null_' + rel] = entity_num
	entity_num += 1

for paper in paper_set:
	entity2id[paper] = entity_num
	entity_num += 1

for author in author_set:
	entity2id[author] = entity_num
	entity_num += 1

for meeting in meeting_set:
	entity2id[meeting] = entity_num
	entity_num += 1

with open('DBLP/entity2id.txt', 'w') as f:
	for entity in entity2id.keys():
		f.write('%s\t%d\n' % (entity, entity2id[entity]))


rel_id = rel2id['paper_meeting']

node2str = []
train_data = []
valid_data = []
test_data = []
with open('../data_original/DBLP/train.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		paper = 'paper_' + line[0]
		titile = line[1]
		meeting = line[5]
		year = line[4]

		paper_id = entity2id[paper]
		meeting_id = entity2id[meeting]

		node2str.append('%d\t%s\n' % (paper_id, titile))
		train_data.append('%d\t%d\t%d\n' % (paper_id, rel_id, meeting_id))
with open('../data_original/DBLP/valid.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		paper = 'paper_' + line[0]
		titile = line[1]
		meeting = line[5]
		year = line[4]

		paper_id = entity2id[paper]
		meeting_id = entity2id[meeting]

		node2str.append('%d\t%s\n' % (paper_id, titile))
		valid_data.append('%d\t%d\t%d\n' % (paper_id, rel_id, meeting_id))
with open('../data_original/DBLP/test.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		paper = 'paper_' + line[0]
		titile = line[1]
		meeting = line[5]
		year = line[4]

		paper_id = entity2id[paper]
		meeting_id = entity2id[meeting]

		node2str.append('%d\t%s\n' % (paper_id, titile))
		test_data.append('%d\t%d\t%d\n' % (paper_id, rel_id, meeting_id))
writes('DBLP/node2str.txt', node2str)
print('#train:%d' % len(train_data))
writes('DBLP/train.txt', train_data)
print('#valid:%d' % len(valid_data))
writes('DBLP/valid.txt', test_data)
print('#test:%d' % len(test_data))
writes('DBLP/test.txt', test_data)



data = []
author2meetings = defaultdict(list)
author_list = []

with open('../data_original/DBLP/train.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		paper = 'paper_' + line[0]
		authors = line[2].split(';')
		meeting = line[5]
		year = int(line[4])

		paper_id = entity2id[paper]

		data.append('%d\t%d\t%d\n' % (paper_id, rel2id['paper_meeting'], entity2id[meeting]))

		author_ids = []
		for author in authors:
			if author.strip() != '':
				author_ids.append(entity2id[author.strip()])
		meeting_id = entity2id[meeting]

		for idx in range(0, 5):
			if len(author_ids) > idx:
				author_id = author_ids[idx]
			else:
				author_id = entity2id['null_paper_author' + str(idx)]

			rel_id = rel2id['paper_author' + str(idx)]
			data.append('%d\t%d\t%d\n' % (paper_id, rel_id, author_id))

		for idx in range(5, 10):
			if len(author_ids) >= 10 - idx:
				author_id = author_ids[idx - 10]
			else:
				author_id = entity2id['null_paper_author' + str(idx)]

			rel_id = rel2id['paper_author' + str(idx)]
			data.append('%d\t%d\t%d\n' % (paper_id, rel_id, author_id))

		for author_id in author_ids:
			author2meetings[author_id].append(meeting_id)
			author_list.append(author_id)

with open('../data_original/DBLP/valid.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		paper = 'paper_' + line[0]
		authors = line[2].split(';')
		meeting = line[5]
		year = int(line[4])

		paper_id = entity2id[paper]

		data.append('%d\t%d\t%d\n' % (paper_id, rel2id['paper_meeting'], entity2id[meeting]))

		author_ids = []
		for author in authors:
			if author.strip() != '':
				author_ids.append(entity2id[author.strip()])
		meeting_id = entity2id[meeting]

		for idx in range(0, 5):
			if len(author_ids) > idx:
				author_id = author_ids[idx]
			else:
				author_id = entity2id['null_paper_author' + str(idx)]

			rel_id = rel2id['paper_author' + str(idx)]
			data.append('%d\t%d\t%d\n' % (paper_id, rel_id, author_id))

		for idx in range(5, 10):
			if len(author_ids) >= 10 - idx:
				author_id = author_ids[idx - 10]
			else:
				author_id = entity2id['null_paper_author' + str(idx)]

			rel_id = rel2id['paper_author' + str(idx)]
			data.append('%d\t%d\t%d\n' % (paper_id, rel_id, author_id))

with open('../data_original/DBLP/test.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip().split('\t')
		paper = 'paper_' + line[0]
		authors = line[2].split(';')
		meeting = line[5]
		year = int(line[4])

		paper_id = entity2id[paper]

		data.append('%d\t%d\t%d\n' % (paper_id, rel2id['paper_meeting'], entity2id[meeting]))

		author_ids = []
		for author in authors:
			if author.strip() != '':
				author_ids.append(entity2id[author.strip()])
		meeting_id = entity2id[meeting]

		for idx in range(0, 5):
			if len(author_ids) > idx:
				author_id = author_ids[idx]
			else:
				author_id = entity2id['null_paper_author' + str(idx)]

			rel_id = rel2id['paper_author' + str(idx)]
			data.append('%d\t%d\t%d\n' % (paper_id, rel_id, author_id))

		for idx in range(5, 10):
			if len(author_ids) >= 10 - idx:
				author_id = author_ids[idx - 10]
			else:
				author_id = entity2id['null_paper_author' + str(idx)]

			rel_id = rel2id['paper_author' + str(idx)]
			data.append('%d\t%d\t%d\n' % (paper_id, rel_id, author_id))

for author_id in set(author_list):
	meeting_list = author2meetings[author_id]
	meeting2count = defaultdict(int)
	for meeting in meeting_list:
		meeting2count[meeting] += 1
	meetings = []
	counts = []
	for meeting in set(meeting_list):
		meetings.append(meeting)
		counts.append(meeting2count[meeting])

	sort_ids = np.array(counts).argsort()
	for i in range(0, 10):
		if len(counts) > i:
			idx = sort_ids[i]
			meeting_id = meetings[idx]
		else:
			meeting_id = entity2id['null_author_meeting' + str(i)]

		rel_id = rel2id['author_meeting' + str(i)]

		data.append('%d\t%d\t%d\n' % (author_id, rel_id, meeting_id))

writes('DBLP/data.txt', data)