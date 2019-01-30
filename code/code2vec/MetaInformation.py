import pandas as pd
import numpy as np
import os


class MetaInformation:

    def __init__(self, dataset_folder, logger=None):
        self.dataset_folder = dataset_folder
        self.logger = logger
        self.methods = self._load_methods()
        self.nodes = self._load_nodes()
        self.tokens = self._load_tokens()
        self.paths = self._load_paths()

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def _load_methods(self):
        self._log('Loading methods')
        methods = pd.read_csv(os.path.join(self.dataset_folder, 'method_ids.csv'), sep=';', index_col=0)
        return methods

    def _load_nodes(self):
        self._log('Loading nodes')
        nodes = pd.read_csv(os.path.join(self.dataset_folder, 'node_types.csv'), sep=',', index_col=0)
        return nodes

    def _load_tokens(self):
        self._log('Loading tokens')
        with open(os.path.join(self.dataset_folder, 'tokens.csv'), 'r') as f:
            indices = []
            counts = []
            tokens = []
            for line in f:
                if ',' not in line:
                    tokens[-1] += line
                    continue
                ind, cnt, token = line.split(',', 2)
                indices.append(ind)
                counts.append(cnt)
                tokens.append(token)
            for i in range(len(tokens)):
                tokens[i] = tokens[i][:-1]
            indices, counts = np.array(indices), np.array(counts)
            return pd.DataFrame(data={tokens[0]: tokens[1:], counts[0]: counts[1:].astype(int)},
                                index=indices[1:].astype(int))

    def _load_paths(self):
        self._log('Loading paths')
        paths = pd.read_csv(os.path.join(self.dataset_folder, 'path_ids.csv'), sep=',', index_col=0)
        return paths

    def get_method_name(self, ind):
        return self.methods.loc[ind]['methodName']

    def get_method_args(self, ind):
        return self.methods.loc[ind]['argTypes']
