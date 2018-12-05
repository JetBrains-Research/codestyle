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


def create_dataset(changes_path, meta_file, entities_dict_path, compressed_changes_path, data_count=72):
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

    entity2methods = {entity: 0 for entity in entities}

    for file_id in range(data_count):
        print("Processing file changes {}/{}".format(file_id + 1, data_count))
        df = pd.read_csv(data_file.format(file_id))
        for index, row in df.iterrows():
            entity, author_time = change2entity[row['changeId']]
            entity2methods[entity] += 1
        del df

    entities_count = [(entity2methods[entity], entity) for entity in entities]

    entity2k = {entity: i + 1 for i, (_, entity) in enumerate(sorted(entities_count, reverse=True))}

    buckets = [(1, 2), (3, 4), (5, 6), (7, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 100), (101, 500)]

    def get_bucket(k):
        for b in buckets:
            if b[0] <= k <= b[1]:
                return b

    fout = {b: open(compressed_changes_path + 'changes_{}_{}.csv'.format(b[0], b[1]), 'w') for b in buckets}
    for bucket in buckets:
        fout[bucket].write('id,changeId,entity,methodBeforeId,methodAfterId,pathsCountBefore,pathsCountAfter,pathsBefore,pathsAfter\n')

    cnt = 0
    for file_id in range(data_count):
        print("Processing #{}".format(file_id + 1))
        df = pd.read_csv(data_file.format(file_id))
        for index, row in df.iterrows():
            cnt += 1
            entity = saved_entity(row['authorName'], row['authorEmail'])
            bucket = get_bucket(entity2k[entity])
            cnt += 1
            fout[bucket].write('{},{},{},{},{},{},{},{},{}\n'.format(cnt,
                                                                     row['changeId'],
                                                                     entity,
                                                                     row['methodBeforeId'], row['methodAfterId'],
                                                                     row['pathsCountBefore'], row['pathsCountAfter'],
                                                                     row['pathsBefore'], row['pathsAfter']))
        del df


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
                   compressed_changes_path='dataset/changes/',
                   data_count=72)
