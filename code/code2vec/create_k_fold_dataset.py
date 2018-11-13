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


def path2fold(k):
    return path2dataset + 'fold_all_{}_{}_{}.csv'.format(buckets[0][0], buckets[-1][1], k)

path_limit = 500
count_max = 200

data = []


def create_line(entity, added, deleted):
    return str(entity) + \
           ',' + \
           ','.join(added) + ',' * (count_max - max(1, len(added))) + \
           ',' + \
           ','.join(deleted) + ',' * (count_max - max(1, len(deleted))) + \
           '\n'


for l, r in buckets:
    with open(get_file(path2creations, l, r), 'r') as fin:
        for line in fin:
            line = line[:-1]
            entity, paths = line.split(',')
            paths = paths.split(';')
            if len(paths) > path_limit or len(paths) < 6:
                continue
            if len(paths) > count_max:
                paths = paths[:count_max]
            line = create_line(entity, paths, [])
            if np.random.uniform() < 0.3:
                data.append(line)
            else:
                data.append(line)

for l, r in buckets:
    with open(get_file(path2deletions, l, r), 'r') as fin:
        for line in fin:
            line = line[:-1]
            entity, paths = line.split(',')
            paths = paths.split(';')
            if len(paths) > path_limit or len(paths) < 6:
                continue
            if len(paths) > count_max:
                paths = paths[:count_max]
            line = create_line(entity, [], paths)
            if np.random.uniform() < 0.3:
                data.append(line)
            else:
                data.append(line)

for l, r in buckets:
    with open(get_file(path2changes, l, r), 'r') as fin:
        for line in fin:
            line = line[:-1]
            entity, added, deleted = line.split(',')
            added = added.split(';')
            deleted = deleted.split(';')
            added_set = set(added)
            deleted_set = set(deleted)
            filtered_added = []
            filtered_deleted = []
            for item in added:
                if item not in deleted_set:
                    filtered_added.append(item)
            for item in deleted:
                if item not in added_set:
                    filtered_deleted.append(item)

            if len(filtered_added) > path_limit or len(filtered_deleted) > path_limit or \
                    len(filtered_deleted) + len(filtered_added) < 6:
                continue

            if len(filtered_added) > count_max:
                filtered_added = filtered_added[:count_max]

            if len(filtered_deleted) > count_max:
                filtered_deleted = filtered_deleted[:count_max]

            line = create_line(entity, filtered_added, filtered_deleted)
            if np.random.uniform() < 0.3:
                data.append(line)
            else:
                data.append(line)

np.random.shuffle(data)

for k in range(1, 11):
    with open(path2fold(k), 'w') as fout:
        for line in data[int(len(data) * (k - 1) / 10):int(len(data) * k / 10)]:
            fout.write(line)
