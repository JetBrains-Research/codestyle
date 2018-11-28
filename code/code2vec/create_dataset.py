import pandas as pd
import numpy as np
from collections import Counter
import pickle

data_file = "dataset/idea-changes/file_changes_{}.csv"
data_count = 72

count = 0
creations = 0
deletions = 0
changes = 0

author_counts = Counter()
email_counts = Counter()


def read_tokens(token_file):
    with open(token_file, 'r') as f:
        indices = []
        counts = []
        tokens = []
        for line in f:
            if not ',' in line:
                tokens[-1] += line
                continue
            ind, cnt, token = line.split(',', 2)
            indices.append(ind)
            counts.append(cnt)
            tokens.append(token)
        for i in range(len(tokens)):
            tokens[i] = tokens[i][:-1]
        return np.array(indices), np.array(counts), tokens



_indices, _counts, _tokens = read_tokens('dataset/idea-changes/tokens.csv')
tokens = pd.DataFrame(data={_tokens[0]: _tokens[1:], _counts[0]: _counts[1:].astype(int)}, 
                      index=_indices[1:].astype(int))

methods = pd.read_csv('dataset/idea-changes/method_ids.csv', sep=';', index_col=0)
nodes = pd.read_csv('dataset/idea-changes/node_types.csv', sep=',', index_col=0)
paths = pd.read_csv('dataset/idea-changes/path_ids.csv', sep=',', index_col=0)
meta = pd.read_csv('dataset/idea-changes/change_metadata.csv', index_col=0)
entities = pd.read_csv('dataset/entities.csv', sep=';')

entities['names'] = entities['names'].map(lambda s: str(s).split(','))
entities['emails'] = entities['emails'].map(lambda s: str(s).split(','))
entities['count'] = np.max([entities['authorCount'], entities['emailCount']], axis=0)
entities = entities.drop(['authorCount', 'emailCount'], axis=1)

with open('dataset/entities.dict', 'rb') as fin:
    saved_entities = pickle.load(fin)

random_member = {}
for key in saved_entities:
    if saved_entities[key] not in random_member:
        random_member[saved_entities[key]] = key


def saved_entity(author, email):
    if (author, email) in saved_entities:
        return saved_entities[(author, email)]
    raise ValueError('No entity saved for:', author, email)


changes_by_entity = {}
for index, row in meta.iterrows():
    try:
        entity = saved_entity(row['authorName'], row['authorEmail'])
    except ValueError as err:
        print(err.args)
        continue
    if entity not in changes_by_entity:
        changes_by_entity[entity] = []
    changes_by_entity[entity].append((row['authorTime'], index))

for entity in changes_by_entity:
    changes_by_entity[entity] = sorted(changes_by_entity[entity])

entity_size = []
for entity in changes_by_entity:
    entity_size.append((len(changes_by_entity[entity]), entity))
entity_size = reversed(sorted(entity_size))

entity2k = {}
for i, (_, entity) in enumerate(entity_size):
    entity2k[entity] = i + 1

valid = np.zeros(len(meta), dtype=np.bool)
for entity in changes_by_entity:
    changes_by_entity[entity] = sorted(changes_by_entity[entity]) 
    size = len(changes_by_entity[entity])
    for i in range(size // 3, size * 2 // 3):
        valid[changes_by_entity[entity][i][1]] = True

meta['valid'] = valid

buckets = [(1, 2), (3, 4), (5, 6), (7, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 100), (101, 500)]


def get_bucket(k):
    for b in buckets:
        if b[0] <= k <= b[1]:
            return b


#  Write headers
# fout = {b: open('changes/changes_{}_{}.csv'.format(b[0], b[1]), 'w') for b in buckets}
# for bucket in buckets:
#     fout[bucket].write('id,changeId,entity,methodBeforeId,methodAfterId,pathsCountBefore,pathsCountAfter,pathsBefore,pathsAfter\n')

cnt = 0

entity2ids = {entity: [] for entity in changes_by_entity}
id2path = {}

for file_id in range(data_count):
    print("Processing #{}".format(file_id + 1))
    df = pd.read_csv(data_file.format(file_id))
    for index, row in df.iterrows():
        if not valid[row['changeId']]:
            continue
        entity = saved_entity(row['authorName'], row['authorEmail'])
        bucket = get_bucket(entity2k[entity])
        cnt += 1
        entity2ids[entity].append(cnt)
        old_path = meta.loc[row['changeId']]['oldPath']
        new_path = meta.loc[row['changeId']]['newPath']
        id2path[cnt] = old_path if old_path != 'null' else new_path
        # fout[bucket].write('{},{},{},{},{},{},{},{},{}\n'.format(cnt,
        #                                            row['changeId'],
        #                                            entity,
        #                                            row['methodBeforeId'], row['methodAfterId'],
        #                                            row['pathsCountBefore'], row['pathsCountAfter'],
        #                                            row['pathsBefore'], row['pathsAfter']))
    del df

# for b in buckets:
#     fout[b].close()

fout_train = {b: open('dataset/batches_separated/train_batches_{}_{}.csv'.format(b[0], b[1]), 'w') for b in buckets}
fout_test = {b: open('dataset/batches_separated/test_batches_{}_{}.csv'.format(b[0], b[1]), 'w') for b in buckets}

np.random.seed(42)
test_size = 0.3
train_size = 1 - test_size
batch_multiplier = 1
batch_size = 16

for i, entity in enumerate(entity2ids):
    print('Processing {}/{}'.format(i + 1, len(entity2ids)))
    k = entity2k[entity]
    bucket = get_bucket(k)
    ids = entity2ids[entity]
    np.random.shuffle(ids)
    train_ids = []
    test_ids = []
    train_methods = set()
    test_methods = set()
    for ind in ids:
        path = id2path[ind]
        if path in train_methods:
            train_ids.append(ind)
        elif path in test_methods:
            test_ids.append(ind)
        else:
            if len(train_ids) == 0:
                train_methods.add(path)
                train_ids.append(ind)
            elif len(test_ids) == 0:
                test_methods.add(path)
                test_ids.append(ind)
            elif len(train_ids) / len(test_ids) < train_size / test_size:
                train_methods.add(path)
                train_ids.append(ind)
            else:
                test_methods.add(path)
                test_ids.append(ind)

    for _ in range(len(train_ids) * batch_multiplier):
        indices = np.random.choice(train_ids, batch_size)
        fout_train[bucket].write(str(entity) + ',' + ','.join(str(ind) for ind in indices) + '\n')
            
    for _ in range(len(test_ids) * batch_multiplier):
        indices = np.random.choice(test_ids, batch_size)
        fout_test[bucket].write(str(entity) + ',' + ','.join(str(ind) for ind in indices)+ '\n')
