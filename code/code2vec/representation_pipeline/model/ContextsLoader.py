import tensorflow as tf
import numpy as np
import pandas as pd

empty_context = [0, 0, 0]


class ContextsLoader:
    ADDED = 0
    DELETED = 1
    CHANGED = 2

    def __init__(self, config, files, seed=42, mask_tokens=False):
        np.random.seed(seed)
        self.config = config
        self.ids, self.rev_ids, self.paths_before, self.paths_after, self.dim_ids, self.types, \
        self.change_ids, self.method_before_ids, self.method_after_ids \
            = self.read_files(files, mask_tokens)
        print('Loaded all files')
        # self.default_value = tf.constant([empty_context for _ in range(self.config.MAX_CONTEXTS)], dtype=tf.int32)
        self.size = len(self.ids)

    def read_files(self, files, mask_tokens):
        size = 0
        for filename in files:
            print("Processing", filename)
            with open(filename, 'r', encoding='utf-8') as f:
                size += len(f.readlines()) - 1

        # sum(len(open(filename, 'r').readlines()) - 1 for filename in files)
        ids = np.zeros(size, dtype=np.int32)
        change_ids = np.zeros(size, dtype=np.int32)
        method_before_ids = np.zeros(size, dtype=np.int32)
        method_after_ids = np.zeros(size, dtype=np.int32)
        paths_before = np.zeros((size, self.config.MAX_CONTEXTS, 3), np.int32)
        paths_after = np.zeros((size, self.config.MAX_CONTEXTS, 3), np.int32)
        types = np.zeros(size)
        cnt = 0

        for i, file in enumerate(files):
            print('Reading file #{}/{}'.format(i + 1, len(files)))
            df = pd.read_csv(file)
            for index, row in df.iterrows():
                cnt_before = row['pathsCountBefore']
                cnt_after = row['pathsCountAfter']
                if cnt_before == 0 and cnt_after != 0:
                    types[cnt] = self.ADDED
                elif cnt_before != 0 and cnt_after == 0:
                    types[cnt] = self.DELETED
                else:
                    types[cnt] = self.CHANGED

                # if cnt_before + cnt_after < self.config.PATH_MIN or \
                #         self.config.PATH_MAX < cnt_before or self.config.PATH_MAX < cnt_after:
                #     continue
                # ids[cnt] = index
                ids[cnt] = cnt
                change_ids[cnt] = row['changeId']
                # change_ids[cnt] = cnt
                method_before_ids[cnt] = row['methodBeforeId']
                method_after_ids[cnt] = row['methodAfterId']
                set_restricted = set(row['pathsBefore'].split(';')) & set(row['pathsAfter'].split(';')) \
                    if type(row['pathsBefore']) is str \
                       and type(row['pathsAfter']) is str \
                    else set()

                self.unpack_and_trim(row['pathsBefore'], paths_before[cnt], set_restricted)
                self.unpack_and_trim(row['pathsAfter'], paths_after[cnt], set_restricted)
                cnt += 1
            del df

        if mask_tokens:
            paths_before[:, :, 0] = 0
            paths_before[:, :, 2] = 0
            paths_after[:, :, 0] = 0
            paths_after[:, :, 2] = 0

        reverse_ids = np.ones(max(ids) + 1, dtype=np.int32) * -1
        for i, ind in enumerate(ids):
            reverse_ids[ind] = i

        return ids, reverse_ids, paths_before, paths_after, max(ids) + 1, types, \
               change_ids, method_before_ids, method_after_ids

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

    def get(self, batched_indices):
        batch_size = len(batched_indices)
        paths_before = np.zeros((batch_size, self.config.PACK_SIZE, self.config.MAX_CONTEXTS, 3))
        paths_after = np.zeros((batch_size, self.config.PACK_SIZE, self.config.MAX_CONTEXTS, 3))
        for i, indices in enumerate(batched_indices):
            indices = self.rev_ids[indices]
            paths_before[i] = self.paths_before[indices]
            paths_after[i] = self.paths_after[indices]
        return paths_before, paths_after

    def is_added(self, index):
        return self.types[index] == self.ADDED

    def get_paths(self, index):
        return self.paths_before[index], self.paths_after[index]
