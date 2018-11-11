import numpy as np


path2dataset = "dataset/"
path2changes = path2dataset + 'changes/changes_'
path2metachanges = path2dataset + 'changes_meta/meta_changes_'
path2deletions = path2dataset + 'deletions/deletions_'
path2metadeletions = path2dataset + 'deletions_meta/meta_deletions_'
path2creations = path2dataset + 'creations/creations_'
path2metacreations = path2dataset + 'creations_meta/meta_creations_'


def get_file(path2file, l, r):
    return path2file + str(l) + '_' + str(r) + '.csv'


np.random.seed(42)

buckets = [(1, 2), (3, 4), (5, 6), (7, 10)]
path2train = path2dataset + 'train_creations_{}_{}.csv'.format(buckets[0][0], buckets[-1][1])
path2test = path2dataset + 'test_creations_{}_{}.csv'.format(buckets[0][0], buckets[-1][1])

count_max = 1000

train_data = []
test_data = []

for l, r in buckets:
    with open(get_file(path2creations, l, r), 'r') as fin:
        for line in fin:
            line = line.replace(';', ',')
            if line.count(',') > count_max:
                continue
            line = line[:-1] + ',' * (count_max - line.count(',')) + '\n'
            if np.random.uniform() < 0.3:
                test_data.append(line)
            else:
                train_data.append(line)

np.random.shuffle(train_data)
np.random.shuffle(test_data)

with open(path2train, 'w') as f_train:
    for line in train_data:
        f_train.write(line)

with open(path2test, 'w') as f_test:
    for line in test_data:
        f_test.write(line)
