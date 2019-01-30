import pandas as pd
import numpy as np
import pickle
import argparse


def get_prefix(path, depth):
    path = path[::-1]
    while depth != 0:
        ind = path.find('/')
        if ind == -1:
            break
        path = path[ind + 1:]
        depth -= 1
    return path[::-1]


def valid_path(path):
    return type(path) is str and path != 'null'


def create_dataset(changes_path, meta_file, entities_dict_path, batches_path, test_size, pack_size, path_depth, data_count=72):
    data_file = changes_path + 'file_changes_{}.csv'
    meta = pd.read_csv(meta_file, index_col=0)

    with open(entities_dict_path, 'rb') as fin:
        saved_entities = pickle.load(fin)

    entities = []

    random_member = {}
    for key in saved_entities:
        if saved_entities[key] not in random_member:
            random_member[saved_entities[key]] = key
            entities.append(saved_entities[key])

    def saved_entity(author, email):
        if (author, email) in saved_entities:
            return saved_entities[(author, email)]
        raise ValueError('No entity saved for:', author, email)

    change2entity = {}
    for index, row in meta.iterrows():
        try:
            entity = saved_entity(row['authorName'], row['authorEmail'])
        except ValueError as err:
            print(err.args)
            continue
        change2entity[index] = (entity, row['authorTime'])

    cnt = 0
    entity2methods = {entity: [] for entity in entities}
    id2path = {}

    for file_id in range(data_count):
        print("Processing file changes {}/{}".format(file_id + 1, data_count))
        df = pd.read_csv(data_file.format(file_id))
        for index, row in df.iterrows():
            cnt += 1
            entity, author_time = change2entity[row['changeId']]
            entity2methods[entity].append((author_time, cnt))
            new_path = meta.loc[row['changeId']]['newPath']
            old_path = meta.loc[row['changeId']]['oldPath']
            if valid_path(old_path):
                id2path[cnt] = old_path
            elif valid_path(new_path):
                id2path[cnt] = new_path
            else:
                id2path[cnt] = ""

        del df

    entities_count = [(len(entity2methods[entity]), entity) for entity in entities]

    entity2k = {entity: i + 1 for i, (_, entity) in enumerate(sorted(entities_count, reverse=True))}

    buckets = [(1, 2), (3, 4), (5, 6), (7, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 100), (101, 500)]

    def get_bucket(k):
        for b in buckets:
            if b[0] <= k <= b[1]:
                return b

    fout_train = {b: open(batches_path + 'train_batches_{}_{}.csv'.format(b[0], b[1]), 'w') for b in buckets}
    fout_test = {b: open(batches_path + 'test_batches_{}_{}.csv'.format(b[0], b[1]), 'w') for b in buckets}

    np.random.seed(42)
    train_size = 1 - test_size
    pack_multiplier = 1

    for i, entity in enumerate(entities):
        print('Processing person {}/{}'.format(i + 1, len(entities)))
        k = entity2k[entity]
        bucket = get_bucket(k)
        all_ids = sorted(entity2methods[entity])
        ids = []
        size = len(all_ids)
        for j in range(size // 3, size * 2 // 3):
            ids.append(all_ids[j][1])
        print("Rank is {}, got {} ids".format(k, len(ids)))
        np.random.shuffle(ids)
        train_ids = []
        test_ids = []
        if path_depth != -1:
            train_paths = set()
            test_paths = set()
            for ind in ids:
                path = get_prefix(id2path[ind], path_depth)
                if path in train_paths:
                    train_ids.append(ind)
                elif path in test_paths:
                    test_ids.append(ind)
                else:
                    if len(train_ids) == 0:
                        train_paths.add(path)
                        train_ids.append(ind)
                    elif len(test_ids) == 0:
                        test_paths.add(path)
                        test_ids.append(ind)
                    elif len(train_ids) / len(test_ids) < train_size / test_size:
                        train_paths.add(path)
                        train_ids.append(ind)
                    else:
                        test_paths.add(path)
                        test_ids.append(ind)
            print(len(test_paths))
            for i, p in enumerate(test_paths):
                if i >= 10:
                    break
                print(p)
            print(len(train_paths))
            for i, p in enumerate(train_paths):
                if i >= 10:
                    break
                print(p)
        else:
            train_ids = ids[:int(train_size * len(ids))]
            test_ids = ids[len(train_ids):]

        if len(test_ids) > 0:
            print('train size: {}, test size: {}, ratio: {}'.format(len(train_ids), len(test_ids), len(train_ids) / len(test_ids)))
        else:
            print('train size: {}, test size: {}, ratio: {}'.format(len(train_ids), len(test_ids), "INFINITY"))

        for _ in range(len(train_ids) * pack_multiplier):
            indices = np.random.choice(train_ids, pack_size)
            fout_train[bucket].write(str(entity) + ',' + ','.join(str(ind) for ind in indices) + '\n')

        for _ in range(len(test_ids) * pack_multiplier):
            indices = np.random.choice(test_ids, pack_size)
            fout_test[bucket].write(str(entity) + ',' + ','.join(str(ind) for ind in indices)+ '\n')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path-depth', dest='path_depth', required=False, default=0, type=int,
    #                     help='depth of paths that should be separated between train and test')
    # parser.add_argument('--time-separated', dest='time_separated', action='store_true',
    #                     help='if passed then train and test will be from different parts of history for each person')
    # args = parser.parse_args()

    create_dataset(changes_path='dataset/idea-changes/',
                   meta_file='dataset/idea-changes/change_metadata.csv',
                   entities_dict_path='dataset/entities.dict',
                   batches_path='dataset/batches_separated_2/',
                   test_size=0.3,
                   pack_size=16,
                   path_depth=2,
                   data_count=72)
