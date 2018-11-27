import tensorflow as tf
import numpy as np
import pandas as pd

empty_context = [0, 0, 0]


class ContextsLoader:

    def __init__(self, config, files, seed=42):
        np.random.seed(seed)
        self.config = config
        self.ids, self.paths_before, self.paths_after, self.dim_ids = self.read_files(files)
        print('Loaded all files')
        self.default_value = tf.constant([empty_context for _ in range(self.config.MAX_CONTEXTS)], dtype=tf.int32)
        self.size = len(self.ids)
        self.ids_table = self.ids_mapping(self.ids)
        self.before_table = self.create_table(self.paths_before)
        self.after_table = self.create_table(self.paths_after)

    def read_files(self, files):
        size = sum(len(open(filename).readlines()) - 1 for filename in files)

        ids = np.zeros(size)
        paths_before = np.zeros((size, 200, 3))
        paths_after = np.zeros((size, 200, 3))
        cnt = 0

        for i, file in enumerate(files):
            print('Reading file #{}/{}'.format(i + 1, len(files)))
            df = pd.read_csv(file, index_col=0)
            for index, row in df.iterrows():
                cnt_before = row['pathsCountBefore']
                cnt_after = row['pathsCountAfter']
                # if cnt_before + cnt_after < self.config.PATH_MIN or \
                #         self.config.PATH_MAX < cnt_before or self.config.PATH_MAX < cnt_after:
                #     continue
                ids[cnt] = index
                set_restricted = set(row['pathsBefore'].split(';')) &  set(row['pathsAfter'].split(';'))
                self.unpack_and_trim(row['pathsBefore'], paths_before[cnt], set_restricted)
                self.unpack_and_trim(row['pathsAfter'], paths_after[cnt], set_restricted)
                cnt += 1
            del df

        return ids, paths_before, paths_after, max(ids) + 1

    def unpack_and_trim(self, paths, output, restricted):
        if type(paths) is str:
            paths = paths.split(';')
            paths = [path for path in paths if path not in restricted]
            paths = list(map(lambda p: list(map(int, p.split())), paths))
            # if len(paths) > self.config.MAX_CONTEXTS:
            #     np.random.shuffle(paths)
            #     paths = paths[:self.config.MAX_CONTEXTS]
        else:
            paths = []
        for i in range(min(len(paths), self.config.MAX_CONTEXTS)):
            output[i] = paths[i]

    def ids_mapping(self, ids):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(
                ids, np.arange(self.size), key_dtype=tf.int32, value_dtype=tf.int32
            ),
            -1,
            name='IDS_LOOKUP'
        )

    def create_table(self, paths):
        return tf.constant(paths, dtype=tf.int32)

    def get(self, indices):
        indices = self.ids_table.lookup(indices)
        return tf.gather(self.before_table, indices, name='GATHER_BEFORE'), \
               tf.gather(self.after_table, indices, name='GATHER_AFTER')
